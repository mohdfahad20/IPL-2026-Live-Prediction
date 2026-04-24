import sqlite3
import logging
import argparse
from pathlib import Path
import time

from cricdata import CricinfoClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("ipl.db")
SERIES_SLUG = "ipl-2026-1510719"
SEASON = "2026"

TEAM_ALIASES = {
    "MI": "Mumbai Indians",
    "CSK": "Chennai Super Kings",
    "RCB": "Royal Challengers Bengaluru",
    "DC": "Delhi Capitals",
    "KKR": "Kolkata Knight Riders",
    "PBKS": "Punjab Kings",
    "RR": "Rajasthan Royals",
    "SRH": "Sunrisers Hyderabad",
    "GT": "Gujarat Titans",
    "LSG": "Lucknow Super Giants",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}


def normalise_team(name):
    if not name:
        return None
    return TEAM_ALIASES.get(name.strip(), name.strip())


# ─── FETCH ─────────────────────────────────────────────

def fetch_fixtures():
    log.info(f"Fetching fixtures for: {SERIES_SLUG}")

    ci = CricinfoClient()

    for attempt in range(3):
        try:
            data = ci.series_fixtures(SERIES_SLUG)
            matches = data.get("content", {}).get("matches", [])
            log.info(f"Fetched {len(matches)} matches")
            return matches
        except Exception as e:
            log.warning(f"Retry {attempt+1}: {e}")
            time.sleep(2)

    log.error("Failed to fetch data")
    return []


# ─── RESULT PARSE ───────────────────────────────────────

def parse_result(m):
    status = (m.get("status") or m.get("statusText") or "").lower()

    if "no result" in status or "abandon" in status:
        return "no result"
    if "tie" in status or "super over" in status:
        return "super over"
    if "wicket" in status:
        return "wickets"
    if "run" in status:
        return "runs"

    return "unknown"


# ─── PARSE ─────────────────────────────────────────────

def build_match_record(m):
    try:
        teams = m.get("teams", [])
        if len(teams) < 2:
            return None

        team1 = normalise_team(teams[0]["team"]["longName"])
        team2 = normalise_team(teams[1]["team"]["longName"])

        # winner (ALLOW None)
        winner = None
        winner_id = m.get("winnerTeamId")

        for t in teams:
            if t["team"]["id"] == winner_id:
                winner = normalise_team(t["team"]["longName"])

        # date
        date_raw = m.get("startDate")
        date_str = date_raw[:10] if date_raw else None
        if not date_str:
            return None

        # toss
        toss = m.get("toss") or {}
        toss_winner = normalise_team(toss.get("winner", {}).get("longName"))
        toss_decision = toss.get("decision")

        # venue
        ground = m.get("ground", {})
        venue = ground.get("longName") or ground.get("name")
        city = ground.get("town", {}).get("name")

        # pom
        pom_list = m.get("playerOfMatch", [])
        pom = pom_list[0]["longName"] if pom_list else None

        match_id_raw = m.get("objectId") or m.get("id")
        if not match_id_raw:
            return None

        match_id = f"{SEASON}_{match_id_raw}"

        return {
            "match_id": match_id,
            "season": SEASON,
            "date": date_str,
            "city": city,
            "venue": venue,
            "team1": team1,
            "team2": team2,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "winner": winner,  # ✅ IMPORTANT
            "result": parse_result(m),
            "result_margin": None,
            "player_of_match": pom,
            "method": None,
            "stage": m.get("stage", "Group"),
            "event_match_no": m.get("number"),
            "source": "cricdata_2026",
        }

    except Exception as e:
        log.warning(f"Parse error: {e}")
        return None


COLS = [
    "match_id", "season", "date", "city", "venue",
    "team1", "team2", "toss_winner", "toss_decision",
    "winner", "result", "result_margin", "player_of_match",
    "method", "stage", "event_match_no", "source",
]


# ─── UPSERT ─────────────────────────────────────────────

def insert_matches(matches, conn, dry_run=False):
    inserted = 0

    for m in matches:
        if not m:
            continue

        values = [m.get(c) for c in COLS]

        if dry_run:
            log.info(f"[DRY RUN] {m['team1']} vs {m['team2']}")
            continue

        conn.execute(
            f"""
            INSERT INTO matches ({', '.join(COLS)})
            VALUES ({', '.join(['?']*len(COLS))})
            ON CONFLICT(match_id) DO UPDATE SET
                winner=excluded.winner,
                result=excluded.result,
                toss_winner=excluded.toss_winner,
                toss_decision=excluded.toss_decision,
                player_of_match=excluded.player_of_match
            """,
            values
        )

        inserted += 1

    conn.commit()
    log.info(f"Upserted total: {inserted}")


# ─── MAIN ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    log.info(f"Using DB: {db_path}")

    matches_raw = fetch_fixtures()

    records = [build_match_record(m) for m in matches_raw]
    records = [r for r in records if r]

    log.info(f"Parsed matches (including NR): {len(records)}")

    conn = sqlite3.connect(db_path)

    try:
        insert_matches(records, conn, dry_run=args.dry_run)

        # DEBUG CHECK
        cnt = conn.execute(
            "SELECT COUNT(*) FROM matches WHERE season='2026'"
        ).fetchone()[0]

        log.info(f"Total matches in DB after scrape: {cnt}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()