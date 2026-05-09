"""
scrape_standings.py — Robust IPL 2026 Points Table Scraper
==========================================================
HTML (Cricbuzz) → fallback → API
Form computed from ipl.db and merged into scraped standings.

Usage:
    python scrape_standings.py
    python scrape_standings.py --dry-run
    python scrape_standings.py --db ipl.db
"""

import argparse
import json
import logging
import sqlite3
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

STANDINGS_PATH = Path("standings.json")
DB_PATH        = Path("ipl.db")

SERIES_URL = "https://www.cricbuzz.com/cricket-series/9241/indian-premier-league-2026/points-table"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.cricbuzz.com/",
}

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
}


def norm(name):
    if not name:
        return name
    return TEAM_ALIASES.get(name.strip().lower(), name.strip())


# ─────────────────────────────────────────────────────────────
# FORM COMPUTATION FROM ipl.db
# ─────────────────────────────────────────────────────────────

def compute_form(db_path: Path, n: int = 5) -> dict:
    """
    Compute last N match form string per team from ipl.db.
    Returns dict: { "Mumbai Indians": "WLWWL", ... }

    Uses only league matches (stage='League') to avoid playoff
    results distorting form display during group stage.
    W = win, L = loss, N = no result
    """
    if not db_path.exists():
        log.warning(f"DB not found at {db_path} — skipping form computation")
        return {}

    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute("""
            SELECT team1, team2, winner, result, date
            FROM matches
            WHERE season = '2026'
              AND stage = 'League'
              AND (winner IS NOT NULL OR result IS NOT NULL)
            ORDER BY date ASC, match_id ASC
        """).fetchall()
        conn.close()
    except Exception as e:
        log.warning(f"DB query failed: {e}")
        return {}

    # Build per-team match history in order
    team_history = {}  # team -> list of 'W'/'L'/'N' in chronological order

    for team1, team2, winner, result, date in rows:
        result_lower = (result or "").lower()
        is_no_result = result_lower in ("no result", "abandoned")

        for team in [team1, team2]:
            if team not in team_history:
                team_history[team] = []

            if is_no_result:
                team_history[team].append("N")
            elif winner == team:
                team_history[team].append("W")
            else:
                team_history[team].append("L")

    # Take last N and join into form string
    form_map = {}
    for team, history in team_history.items():
        last_n = history[-n:]
        form_map[team] = "".join(last_n)

    log.info(f"Form computed for {len(form_map)} teams (last {n} league matches):")
    for team, form in sorted(form_map.items()):
        log.info(f"  {team:<35} {form}")

    return form_map


# ─────────────────────────────────────────────────────────────
# FETCH HTML
# ─────────────────────────────────────────────────────────────

def fetch_page(url):
    try:
        log.info(f"Fetching HTML: {url}")
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        time.sleep(1.2)
        return BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        log.warning(f"HTML fetch failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# PARSE HTML (NEW GRID LAYOUT)
# ─────────────────────────────────────────────────────────────

def parse_html_standings(soup):
    rows = []

    grid_rows = soup.select("div.point-table-grid.p-2")

    if not grid_rows:
        log.warning("No grid rows found.")
        return []

    log.info(f"Found {len(grid_rows)} rows in HTML")

    for row in grid_rows:
        cols = row.find_all("div", recursive=False)

        try:
            team_tag = row.select_one("a div.text-xs")
            if not team_tag:
                continue

            team = norm(team_tag.get_text(strip=True))
            nums = [c.get_text(strip=True) for c in cols]

            rows.append({
                "team": team,
                "M":    int(nums[2]),
                "W":    int(nums[3]),
                "L":    int(nums[4]),
                "NR":   int(nums[5]),
                "Pts":  int(nums[6]),
                "NRR":  float(nums[7].replace("+", "")),
                "Form": "",
            })

        except Exception as e:
            log.debug(f"Skipping row: {e}")
            continue

    log.info(f"Parsed {len(rows)} teams from HTML")
    return rows


# ─────────────────────────────────────────────────────────────
# API FALLBACK
# ─────────────────────────────────────────────────────────────

def fetch_api_standings():
    url = "https://www.cricbuzz.com/api/cricket-series/9241/points-table"

    try:
        log.info("Trying API fallback...")
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()

        rows = []
        for group in data.get("pointsTable", []):
            for t in group.get("pointsTableInfo", []):
                rows.append({
                    "team": norm(t.get("teamName")),
                    "M":    t.get("matchesPlayed"),
                    "W":    t.get("matchesWon"),
                    "L":    t.get("matchesLost"),
                    "NR":   t.get("noResult"),
                    "Pts":  t.get("points"),
                    "NRR":  round(float(t.get("netRunRate", 0)), 3),
                    "Form": "",
                })

        log.info(f"Parsed {len(rows)} teams from API")
        return rows

    except Exception as e:
        log.warning(f"API failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# MERGE FORM INTO STANDINGS
# ─────────────────────────────────────────────────────────────

def merge_form(rows: list, form_map: dict) -> list:
    """
    Inject form string into each standing row.
    Falls back to empty string if team not found in form_map.
    """
    if not form_map:
        log.warning("No form data to merge — Form column will be empty")
        return rows

    for row in rows:
        team = row.get("team", "")
        row["Form"] = form_map.get(team, "")

    matched = sum(1 for r in rows if r.get("Form"))
    log.info(f"Form merged: {matched}/{len(rows)} teams have form data")
    return rows


# ─────────────────────────────────────────────────────────────
# MAIN SCRAPER
# ─────────────────────────────────────────────────────────────

def scrape_standings():
    # 1. Try HTML
    soup = fetch_page(SERIES_URL)
    if soup:
        rows = parse_html_standings(soup)
        if rows:
            log.info("Using HTML data ✔")
            return rows

    # 2. Fallback → API
    rows = fetch_api_standings()
    if rows:
        log.info("Using API data ✔")
        return rows

    # 3. Fail
    log.error("All methods failed ❌")
    return []


# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────

def save_standings(rows):
    data = {
        "scraped_at": __import__("datetime").datetime.utcnow().isoformat(),
        "standings":  rows,
    }
    STANDINGS_PATH.write_text(json.dumps(data, indent=2))
    log.info(f"Saved → {STANDINGS_PATH}")


# ─────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────

def display(rows):
    print(f"\n{'#':>3} {'Team':<30} {'M':>3} {'W':>3} {'L':>3} {'NR':>3} {'Pts':>4} {'NRR':>7}  {'Form'}")
    print("-" * 78)
    for i, r in enumerate(rows, 1):
        print(
            f"{i:>3}. {r['team']:<30} {r['M']:>3} {r['W']:>3} {r['L']:>3} "
            f"{r['NR']:>3} {r['Pts']:>4} {r['NRR']:>+7.3f}  {r.get('Form','')}"
        )


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--db", default=str(DB_PATH), help="Path to ipl.db")
    args = parser.parse_args()

    # Step 1: Scrape NRR + standings from Cricbuzz
    rows = scrape_standings()
    if not rows:
        log.error("No standings scraped.")
        return

    # Step 2: Compute form from DB and merge
    form_map = compute_form(Path(args.db))
    rows = merge_form(rows, form_map)

    # Step 3: Display + save
    display(rows)

    if not args.dry_run:
        save_standings(rows)
    else:
        log.info("[DRY RUN] Not saving.")


if __name__ == "__main__":
    main()