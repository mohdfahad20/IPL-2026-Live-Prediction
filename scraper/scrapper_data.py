# import sqlite3
# import logging
# import argparse
# from pathlib import Path
# import time

# from cricdata import CricinfoClient

# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# log = logging.getLogger(__name__)

# DEFAULT_DB_PATH = Path("ipl.db")
# SERIES_SLUG = "ipl-2026-1510719"
# SEASON = "2026"

# TEAM_ALIASES = {
#     "MI": "Mumbai Indians",
#     "CSK": "Chennai Super Kings",
#     "RCB": "Royal Challengers Bengaluru",
#     "DC": "Delhi Capitals",
#     "KKR": "Kolkata Knight Riders",
#     "PBKS": "Punjab Kings",
#     "RR": "Rajasthan Royals",
#     "SRH": "Sunrisers Hyderabad",
#     "GT": "Gujarat Titans",
#     "LSG": "Lucknow Super Giants",
#     "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
# }

# COLS = [
#     "match_id", "season", "date", "city", "venue",
#     "team1", "team2", "toss_winner", "toss_decision",
#     "winner", "result", "result_margin", "player_of_match",
#     "method", "stage", "event_match_no", "source",
# ]


# def normalise_team(name):
#     if not name:
#         return None
#     return TEAM_ALIASES.get(name.strip(), name.strip())


# # ─── FETCH ─────────────────────────────────────────────

# def fetch_fixtures():
#     log.info(f"Fetching fixtures for: {SERIES_SLUG}")
#     ci = CricinfoClient()
#     for attempt in range(3):
#         try:
#             data = ci.series_fixtures(SERIES_SLUG)
#             matches = data.get("content", {}).get("matches", [])
#             log.info(f"Fetched {len(matches)} matches")
#             return matches
#         except Exception as e:
#             log.warning(f"Retry {attempt+1}: {e}")
#             time.sleep(2)
#     log.error("Failed to fetch data")
#     return []


# # ─── RESULT PARSE ───────────────────────────────────────

# def parse_result(m):
#     status = (m.get("status") or m.get("statusText") or "").lower()
#     if "no result" in status or "abandon" in status:
#         return "no result"
#     if "tie" in status or "super over" in status:
#         return "super over"
#     if "wicket" in status:
#         return "wickets"
#     if "run" in status:
#         return "runs"
#     return "unknown"


# # ─── PARSE ─────────────────────────────────────────────

# def build_match_record(m):
#     try:
#         status = (m.get("status") or m.get("statusText") or "").lower()

#         is_completed = any(x in status for x in [
#             "result", "won", "tie", "no result", "abandon", "super over"
#         ])

#         if not is_completed:
#             return None

#         teams = m.get("teams", [])
#         if len(teams) < 2:
#             return None

#         team1 = normalise_team(teams[0]["team"]["longName"])
#         team2 = normalise_team(teams[1]["team"]["longName"])

#         is_no_result = "no result" in status or "abandon" in status
#         is_super_over = "super over" in status

#         # ─── WINNER EXTRACTION ───
#         winner = None

#         # 1. Primary winner
#         winner_id = m.get("winnerTeamId")
#         if winner_id:
#             for t in teams:
#                 if t["team"]["id"] == winner_id:
#                     winner = normalise_team(t["team"]["longName"])

#         # 2. Super over fallback
#         if not winner and is_super_over:
#             so_winner_id = (
#                 m.get("superoverWinnerTeamId") or
#                 m.get("superOverWinnerTeamId") or
#                 m.get("superover_winner_team_id") or
#                 (m.get("winner") or {}).get("id")
#             )
#             if so_winner_id:
#                 for t in teams:
#                     if t["team"]["id"] == so_winner_id:
#                         winner = normalise_team(t["team"]["longName"])

#         # 3. Status text fallback
#         if not winner and not is_no_result:
#             status_text = (m.get("statusText") or "").lower()
#             for t in teams:
#                 name = t["team"]["longName"]
#                 if name and name.lower() in status_text:
#                     winner = normalise_team(name)
#                     break
#             if not winner:
#                 for short, full in TEAM_ALIASES.items():
#                     if short.lower() in status_text or full.lower() in status_text:
#                         winner = full
#                         break

#         # ─── DATE ───
#         date_raw = m.get("startDate")
#         date_str = date_raw[:10] if date_raw else None
#         if not date_str:
#             return None

#         # ─── TOSS ───
#         toss          = m.get("toss") or {}
#         toss_winner   = normalise_team(toss.get("winner", {}).get("longName"))
#         toss_decision = toss.get("decision")

#         # ─── VENUE ───
#         ground = m.get("ground", {})
#         venue  = ground.get("longName") or ground.get("name")
#         city   = ground.get("town", {}).get("name")

#         # ─── PLAYER OF MATCH ───
#         pom_list = m.get("playerOfMatch", [])
#         pom      = pom_list[0]["longName"] if pom_list else None

#         # ─── MATCH ID ───
#         match_id_raw = m.get("objectId") or m.get("id")
#         if not match_id_raw:
#             return None

#         return {
#             "match_id":        f"{SEASON}_{match_id_raw}",
#             "season":          SEASON,
#             "date":            date_str,
#             "city":            city,
#             "venue":           venue,
#             "team1":           team1,
#             "team2":           team2,
#             "toss_winner":     toss_winner,
#             "toss_decision":   toss_decision,
#             "winner":          winner,
#             "result":          parse_result(m),
#             "result_margin":   None,
#             "player_of_match": pom,
#             "method":          None,
#             "stage":           m.get("stage", "Group"),
#             "event_match_no":  m.get("number"),
#             "source":          "cricdata_2026",
#         }

#     except Exception as e:
#         log.warning(f"Parse error: {e}")
#         return None


# # ─── UPSERT ─────────────────────────────────────────────

# def insert_matches(matches, conn, dry_run=False):
#     inserted = 0

#     for m in matches:
#         if not m:
#             continue

#         values = [m.get(c) for c in COLS]

#         if dry_run:
#             log.info(f"[DRY RUN] {m['team1']} vs {m['team2']}")
#             continue

#         conn.execute(
#             f"""
#             INSERT INTO matches ({', '.join(COLS)})
#             VALUES ({', '.join(['?']*len(COLS))})
#             ON CONFLICT(match_id) DO UPDATE SET
#                 winner          = excluded.winner,
#                 result          = excluded.result,
#                 toss_winner     = excluded.toss_winner,
#                 toss_decision   = excluded.toss_decision,
#                 player_of_match = excluded.player_of_match
#             """,
#             values
#         )
#         inserted += 1

#     conn.commit()
#     log.info(f"Upserted total: {inserted}")


# # ─── MAIN ─────────────────────────────────────────────

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
#     parser.add_argument("--dry-run", action="store_true")
#     args = parser.parse_args()

#     db_path = Path(args.db).resolve()
#     log.info(f"Using DB: {db_path}")

#     matches_raw = fetch_fixtures()

#     records = [build_match_record(m) for m in matches_raw]
#     records = [r for r in records if r]

#     log.info(f"Parsed matches (including NR): {len(records)}")

#     conn = sqlite3.connect(db_path)
#     try:
#         insert_matches(records, conn, dry_run=args.dry_run)

#         cnt = conn.execute(
#             "SELECT COUNT(*) FROM matches WHERE season='2026'"
#         ).fetchone()[0]

#         log.info(f"Total matches in DB after scrape: {cnt}")
#     finally:
#         conn.close()


# if __name__ == "__main__":
#     main()

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

# ─── IPL 2026 STAGE MAP ─────────────────────────────────
# Match numbers 1-70: League stage
# Match numbers 71-74: Playoffs (fixed by IPL schedule)
IPL_2026_STAGE_MAP = {
    71: "Qualifier 1",
    72: "Eliminator",
    73: "Qualifier 2",
    74: "Final",
}

COLS = [
    "match_id", "season", "date", "city", "venue",
    "team1", "team2", "toss_winner", "toss_decision",
    "winner", "result", "result_margin", "player_of_match",
    "method", "stage", "event_match_no", "source",
]


def normalise_team(name):
    if not name:
        return None
    return TEAM_ALIASES.get(name.strip(), name.strip())


def get_ipl_stage(event_match_no) -> str:
    """
    Map match number to correct IPL stage label.
    - Matches 1-70  → 'League'
    - Matches 71-74 → Playoff stage name
    - Unknown       → 'League' (safe default)
    """
    try:
        num = int(event_match_no)
        return IPL_2026_STAGE_MAP.get(num, "League")
    except (TypeError, ValueError):
        return "League"


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
        status = (m.get("status") or m.get("statusText") or "").lower()

        is_completed = any(x in status for x in [
            "result", "won", "tie", "no result", "abandon", "super over"
        ])

        if not is_completed:
            return None

        teams = m.get("teams", [])
        if len(teams) < 2:
            return None

        team1 = normalise_team(teams[0]["team"]["longName"])
        team2 = normalise_team(teams[1]["team"]["longName"])

        is_no_result  = "no result" in status or "abandon" in status
        is_super_over = "super over" in status

        # ─── WINNER EXTRACTION ───
        winner = None

        # 1. Primary winner
        winner_id = m.get("winnerTeamId")
        if winner_id:
            for t in teams:
                if t["team"]["id"] == winner_id:
                    winner = normalise_team(t["team"]["longName"])

        # 2. Super over fallback
        if not winner and is_super_over:
            so_winner_id = (
                m.get("superoverWinnerTeamId") or
                m.get("superOverWinnerTeamId") or
                m.get("superover_winner_team_id") or
                (m.get("winner") or {}).get("id")
            )
            if so_winner_id:
                for t in teams:
                    if t["team"]["id"] == so_winner_id:
                        winner = normalise_team(t["team"]["longName"])

        # 3. Status text fallback
        if not winner and not is_no_result:
            status_text = (m.get("statusText") or "").lower()
            for t in teams:
                name = t["team"]["longName"]
                if name and name.lower() in status_text:
                    winner = normalise_team(name)
                    break
            if not winner:
                for short, full in TEAM_ALIASES.items():
                    if short.lower() in status_text or full.lower() in status_text:
                        winner = full
                        break

        # ─── DATE ───
        date_raw = m.get("startDate")
        date_str = date_raw[:10] if date_raw else None
        if not date_str:
            return None

        # ─── TOSS ───
        toss          = m.get("toss") or {}
        toss_winner   = normalise_team(toss.get("winner", {}).get("longName"))
        toss_decision = toss.get("decision")

        # ─── VENUE ───
        ground = m.get("ground", {})
        venue  = ground.get("longName") or ground.get("name")
        city   = ground.get("town", {}).get("name")

        # ─── PLAYER OF MATCH ───
        pom_list = m.get("playerOfMatch", [])
        pom      = pom_list[0]["longName"] if pom_list else None

        # ─── MATCH ID ───
        match_id_raw = m.get("objectId") or m.get("id")
        if not match_id_raw:
            return None

        # ─── EVENT MATCH NUMBER ───
        event_match_no = m.get("number")

        # ─── STAGE (KEY FIX) ───
        # cricdata returns status values like FINISHED/RUNNING, not stage names.
        # We derive the correct IPL stage from the match number instead.
        stage = get_ipl_stage(event_match_no)

        return {
            "match_id":        f"{SEASON}_{match_id_raw}",
            "season":          SEASON,
            "date":            date_str,
            "city":            city,
            "venue":           venue,
            "team1":           team1,
            "team2":           team2,
            "toss_winner":     toss_winner,
            "toss_decision":   toss_decision,
            "winner":          winner,
            "result":          parse_result(m),
            "result_margin":   None,
            "player_of_match": pom,
            "method":          None,
            "stage":           stage,
            "event_match_no":  event_match_no,
            "source":          "cricdata_2026",
        }

    except Exception as e:
        log.warning(f"Parse error: {e}")
        return None


# ─── UPSERT ─────────────────────────────────────────────

def insert_matches(matches, conn, dry_run=False):
    inserted = 0

    for m in matches:
        if not m:
            continue

        values = [m.get(c) for c in COLS]

        if dry_run:
            log.info(f"[DRY RUN] {m['team1']} vs {m['team2']} | stage={m['stage']} | match_no={m['event_match_no']}")
            continue

        conn.execute(
            f"""
            INSERT INTO matches ({', '.join(COLS)})
            VALUES ({', '.join(['?']*len(COLS))})
            ON CONFLICT(match_id) DO UPDATE SET
                winner          = excluded.winner,
                result          = excluded.result,
                toss_winner     = excluded.toss_winner,
                toss_decision   = excluded.toss_decision,
                player_of_match = excluded.player_of_match,
                stage           = excluded.stage,
                event_match_no  = excluded.event_match_no
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

    # ─── Stage breakdown log ───
    from collections import Counter
    stage_counts = Counter(r['stage'] for r in records)
    for stage, cnt in sorted(stage_counts.items()):
        log.info(f"  Stage '{stage}': {cnt} matches")

    conn = sqlite3.connect(db_path)
    try:
        insert_matches(records, conn, dry_run=args.dry_run)

        cnt = conn.execute(
            "SELECT COUNT(*) FROM matches WHERE season='2026'"
        ).fetchone()[0]

        # ─── Post-scrape stage verification ───
        stage_rows = conn.execute("""
            SELECT stage, COUNT(*) as cnt FROM matches
            WHERE season='2026'
            GROUP BY stage ORDER BY stage
        """).fetchall()
        log.info("2026 stage breakdown in DB:")
        for row in stage_rows:
            log.info(f"  stage='{row[0]}' → {row[1]} matches")

        log.info(f"Total matches in DB after scrape: {cnt}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()