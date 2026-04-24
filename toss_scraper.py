"""
toss_scraper.py — Cricbuzz Edition
====================================
Scrapes today's IPL match toss from Cricbuzz, then runs pre/post-toss
prediction using the ensemble model.

Usage:
    python toss_scraper.py --dry-run        # test without saving
    python toss_scraper.py --db ipl.db      # scrape + predict + log
    python toss_scraper.py --db ipl.db --manual  # manual override mode
"""

import argparse
import csv
import logging
import re
import sqlite3
import sys
import time
from datetime import datetime, date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from model.train import SoftEnsemble  # noqa: F401 — required for pickle

DB_PATH       = Path("ipl.db")
TOSS_LOG_PATH = Path("toss_predictions.csv")

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

TEAM_ALIASES = {
    "mi":   "Mumbai Indians",
    "csk":  "Chennai Super Kings",
    "rcb":  "Royal Challengers Bengaluru",
    "dc":   "Delhi Capitals",
    "kkr":  "Kolkata Knight Riders",
    "pbks": "Punjab Kings",
    "rr":   "Rajasthan Royals",
    "srh":  "Sunrisers Hyderabad",
    "gt":   "Gujarat Titans",
    "lsg":  "Lucknow Super Giants",
    # full name variants
    "mumbai indians":              "Mumbai Indians",
    "chennai super kings":         "Chennai Super Kings",
    "royal challengers bengaluru": "Royal Challengers Bengaluru",
    "royal challengers bangalore": "Royal Challengers Bengaluru",
    "delhi capitals":              "Delhi Capitals",
    "kolkata knight riders":       "Kolkata Knight Riders",
    "punjab kings":                "Punjab Kings",
    "rajasthan royals":            "Rajasthan Royals",
    "sunrisers hyderabad":         "Sunrisers Hyderabad",
    "gujarat titans":              "Gujarat Titans",
    "lucknow super giants":        "Lucknow Super Giants",
}

def norm(name: str) -> str | None:
    if not name:
        return None
    return TEAM_ALIASES.get(name.strip().lower(), name.strip().title())


# ─── STEP 1: FIND TODAY'S IPL MATCH ON CRICBUZZ ──────────────────────────────

def get_todays_ipl_match() -> dict | None:
    """
    Scrapes Cricbuzz live scores page to find the current/upcoming IPL match.
    Returns dict with url, team1, team2, venue — or None if no match.
    """
    url  = "https://www.cricbuzz.com/live-cricket-scores"
    log.info(f"Fetching: {url}")

    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
    except Exception as e:
        log.error(f"Failed to fetch live scores page: {e}")
        return None

    soup    = BeautifulSoup(r.text, "html.parser")
    anchors = soup.select("a[href*='/live-cricket-scores/']")

    for a in anchors:
        href = a.get("href", "")
        text = a.get_text(" ", strip=True)

        if "indian-premier-league" not in href.lower():
            continue

        full_url = "https://www.cricbuzz.com" + href
        log.info(f"IPL match found: {full_url}")

        # Try to extract team names from the link text or URL slug
        # URL format: /live-cricket-scores/NNNNN/team1-vs-team2-match-N-...
        slug_match = re.search(r"/live-cricket-scores/\d+/([^/]+)", href)
        team1, team2, venue = None, None, None

        if slug_match:
            slug  = slug_match.group(1)
            parts = slug.split("-vs-")
            if len(parts) == 2:
                t1_raw = parts[0].replace("-", " ").strip()
                # team2 may have extra words like "match-22" — take first 2-3 words
                t2_raw = " ".join(parts[1].replace("-", " ").split()[:3])
                team1  = norm(t1_raw)
                team2  = norm(t2_raw)

        return {"url": full_url, "team1": team1, "team2": team2, "venue": venue}

    log.info("No IPL match found on Cricbuzz live scores page.")
    return None


# ─── STEP 2: GET MATCH DETAILS + TOSS FROM MATCH PAGE ────────────────────────

def get_match_details(match_url: str) -> dict | None:
    """
    Fetches the match page and extracts:
    - team1, team2 (from scoreboard headers — more reliable than URL slug)
    - toss_winner, toss_decision
    - venue
    Returns None if toss not yet announced.
    """
    log.info(f"Fetching match page: {match_url}")
    try:
        r = requests.get(match_url, headers=HEADERS, timeout=10)
        r.raise_for_status()
    except Exception as e:
        log.error(f"Failed to fetch match page: {e}")
        return None

    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text(" ", strip=True)

    # ── Team names from page ──────────────────────────────────────────────────
    team1, team2 = None, None

    # Cricbuzz match page has team names in <h1> or title
    title = soup.find("title")
    if title:
        t = title.get_text()
        # "MI vs CSK, IPL 2026 ..." or "Mumbai Indians vs Chennai Super Kings..."
        vs_match = re.search(r"([A-Za-z ]+?) vs ([A-Za-z ]+?)[,|]", t)
        if vs_match:
            team1 = norm(vs_match.group(1).strip())
            team2 = norm(vs_match.group(2).strip())

    # ── Venue ─────────────────────────────────────────────────────────────────
    venue = None
    venue_patterns = [
        r"at ([A-Za-z ,]+(?:Stadium|Ground|Oval|Gardens|Arena|Park))",
        r"Venue[:\s]+([A-Za-z ,]+(?:Stadium|Ground|Oval|Gardens|Arena|Park))",
    ]
    for pat in venue_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            venue = m.group(1).strip()
            break

    # ── Toss ──────────────────────────────────────────────────────────────────
    toss_winner, toss_decision = None, None

    toss_patterns = [
        r"([A-Za-z ]+?) won the toss and (?:opt(?:ed)? to|elected to) (bat|bowl|field)",
        r"([A-Za-z ]+?) won the toss.*?(bat|bowl|field)",
        r"([A-Za-z ]+?) opt(?:ed)? to (bat|bowl|field).*?after winning the toss",
        r"Toss[:\s]+([A-Za-z ]+?),\s*(batting|bowling|fielding)",
    ]
    for pat in toss_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            toss_winner   = norm(m.group(1).strip())
            raw_decision  = m.group(2).strip().lower()
            # toss_decision = "field" if raw_decision in ("bowl", "bowling", "fielding") else "bat"
            if raw_decision in ("bowl", "bowling", "field", "fielding"):
                toss_decision = "field"
            else:
                toss_decision = "bat"
            break

    # Also try the specific Cricbuzz toss div
    toss_div = soup.find("div", class_="text-cbLive") or soup.find(string=re.compile("won the toss", re.I))
    if toss_div and not toss_winner:
        t = toss_div if isinstance(toss_div, str) else toss_div.get_text()
        m = re.search(r"([A-Za-z ]+?) won the toss", t, re.IGNORECASE)
        if m:
            toss_winner = norm(m.group(1).strip())
        m2 = re.search(r"elect(?:ed)? to (bat|bowl|field)", t, re.IGNORECASE)
        if m2:
            raw = m2.group(1).lower()
            toss_decision = "field" if raw in ("bowl", "field") else "bat"

    if not toss_winner or not toss_decision:
        log.info("Toss not yet announced on this page.")
        return None

    log.info(f"Teams:  {team1} vs {team2}")
    log.info(f"Venue:  {venue}")
    log.info(f"Toss:   {toss_winner} elected to {toss_decision}")

    return {
        "team1":         team1,
        "team2":         team2,
        "venue":         venue,
        "toss_winner":   toss_winner,
        "toss_decision": toss_decision,
        "date":          date.today().isoformat(),
    }


# ─── STEP 3: POLL UNTIL TOSS IS AVAILABLE ────────────────────────────────────

# def wait_for_toss(match_url: str, max_retries: int = 3, interval: int = 300) -> dict | None:
#     """
#     Retry a few times with longer intervals.
#     Total wait ~15 minutes max.
#     """
#     log.info(f"Polling for toss (max {max_retries} retries, every {interval}s)...")
#     for attempt in range(1, max_retries + 1):
#         details = get_match_details(match_url)
#         if details:
#             return details
#         log.info(f"  [{attempt}/{max_retries}] Toss not yet... retrying in {interval}s")
#         time.sleep(interval)

#     log.error("Toss not found after max retries.")
#     return None

def wait_for_toss(match_url: str, max_retries: int = 6, interval: int = 60) -> dict | None:
    log.info(f"Polling toss every {interval}s (max {max_retries} tries)...")

    for i in range(max_retries):
        details = get_match_details(match_url)
        if details:
            return details

        log.info(f"[{i+1}/{max_retries}] Toss not yet... retrying...")
        time.sleep(interval)

    log.warning("Toss not found after retries.")
    return None


# ─── STEP 4: RUN PREDICTION ──────────────────────────────────────────────────

def run_prediction(match_info: dict, db_path: Path) -> tuple:
    from model.predict import predict_match
    conn = sqlite3.connect(db_path)
    try:
        pre  = predict_match(team1=match_info["team1"], team2=match_info["team2"],
                             season="2026", venue=match_info.get("venue"), conn=conn)
        post = predict_match(team1=match_info["team1"], team2=match_info["team2"],
                             season="2026", venue=match_info.get("venue"),
                             toss_winner=match_info["toss_winner"],
                             toss_decision=match_info["toss_decision"], conn=conn)
        return pre, post
    finally:
        conn.close()


# ─── STEP 5: DISPLAY + LOG ───────────────────────────────────────────────────

def display(match_info: dict, pre: dict, post: dict):
    t1    = match_info["team1"]
    t2    = match_info["team2"]
    shift = post["p_team1_wins"] - pre["p_team1_wins"]
    impact = "High" if abs(shift) > 0.05 else "Moderate" if abs(shift) > 0.02 else "Low"
    post_fav = t1 if post["p_team1_wins"] >= 0.5 else t2

    print("\n" + "="*60)
    print(f"  IPL 2026 TOSS PREDICTION  —  {match_info['date']}")
    print(f"  {t1} vs {t2}")
    if match_info.get("venue"):
        print(f"  {match_info['venue']}")
    print(f"  Toss: {match_info['toss_winner']} elected to {match_info['toss_decision']}")
    print("="*60)
    print(f"  {'Team':<38} Pre-Toss  Post-Toss")
    print(f"  {'-'*56}")
    print(f"  {t1:<38} {pre['p_team1_wins']:>7.1%}   {post['p_team1_wins']:>7.1%}")
    print(f"  {t2:<38} {pre['p_team2_wins']:>7.1%}   {post['p_team2_wins']:>7.1%}")
    print(f"  Toss shift: {shift:+.1%}  ({impact} impact)")
    print(f"  Post-toss favourite: {post_fav}")
    print("="*60)

    print("\n  Per-model breakdown (post-toss):")
    for mname, mprob in post["model_probs"].items():
        bar = "█" * int(mprob * 30)
        print(f"    {mname:<22}: {mprob:.1%}  {bar}")


def log_to_csv(match_info: dict, pre: dict, post: dict):
    write_header = not TOSS_LOG_PATH.exists()
    shift = post["p_team1_wins"] - pre["p_team1_wins"]

    with open(TOSS_LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "date", "team1", "team2", "venue",
                        "toss_winner", "toss_decision",
                        "pre_p_team1", "post_p_team1", "toss_shift",
                        "post_favourite"])
        post_fav = match_info["team1"] if post["p_team1_wins"] >= 0.5 else match_info["team2"]
        w.writerow([
            datetime.utcnow().isoformat(),
            match_info.get("date", date.today().isoformat()),
            match_info["team1"], match_info["team2"],
            match_info.get("venue", ""),
            match_info["toss_winner"], match_info["toss_decision"],
            round(pre["p_team1_wins"], 4), round(post["p_team1_wins"], 4),
            round(shift, 4), post_fav,
        ])
    log.info(f"Logged → {TOSS_LOG_PATH}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Automated IPL toss scraper + predictor")
    parser.add_argument("--db",      default=str(DB_PATH))
    parser.add_argument("--dry-run", action="store_true")
    # Manual override flags
    parser.add_argument("--team1",         default=None)
    parser.add_argument("--team2",         default=None)
    parser.add_argument("--venue",         default=None)
    parser.add_argument("--toss-winner",   default=None)
    parser.add_argument("--toss-decision", default=None, choices=["bat", "field"])
    parser.add_argument("--date",          default=date.today().isoformat())
    args = parser.parse_args()

    db_path = Path(args.db)

    # Manual override
    if args.team1 and args.team2 and args.toss_winner and args.toss_decision:
        log.info("Manual override mode.")
        match_info = {
            "team1":         norm(args.team1),
            "team2":         norm(args.team2),
            "venue":         args.venue,
            "toss_winner":   norm(args.toss_winner),
            "toss_decision": args.toss_decision,
            "date":          args.date,
        }

    else:
        # Auto scrape mode
        match = get_todays_ipl_match()
        if not match:
            log.info("No IPL match today. Exiting.")
            return

        match_info = wait_for_toss(match["url"])
        if not match_info:
            log.info("Could not get toss. Try manual override with --team1 --team2 --toss-winner --toss-decision")
            return

        # If team names weren't parsed from URL, try to get from match page
        if not match_info.get("team1") and match.get("team1"):
            match_info["team1"] = match["team1"]
        if not match_info.get("team2") and match.get("team2"):
            match_info["team2"] = match["team2"]

    if not match_info.get("team1") or not match_info.get("team2"):
        log.error("Could not determine team names. Use --team1 and --team2 flags.")
        return

    pre, post = run_prediction(match_info, db_path)
    display(match_info, pre, post)

    if not args.dry_run:
        log_to_csv(match_info, pre, post)
    else:
        log.info("[DRY RUN] Not saving to CSV.")


if __name__ == "__main__":
    main()