import json
import logging
import sqlite3
import pandas as pd
from pathlib import Path
from collections import defaultdict
from api.core.database import db_connection
from api.core.config import RUNS_PER_WICKET

log = logging.getLogger(__name__)

STANDINGS_PATH = Path("standings.json")


def get_standings() -> list:
    """
    Primary: scrape_standings.py writes standings.json → read from it.
    Fallback: compute from ipl.db match results.
    """
    if STANDINGS_PATH.exists():
        try:
            data = json.loads(STANDINGS_PATH.read_text())
            rows = data.get("standings", [])
            if rows:
                log.info("Standings loaded from standings.json")
                # Normalize column names (scraper uses M, we expose M)
                return [
                    {
                        "team": r.get("team"),
                        "M":    r.get("M", 0),
                        "W":    r.get("W", 0),
                        "L":    r.get("L", 0),
                        "NR":   r.get("NR", 0),
                        "Pts":  r.get("Pts", 0),
                        "NRR":  r.get("NRR", 0.0),
                        "Form": r.get("Form", ""),
                    }
                    for r in rows
                ]
        except Exception as e:
            log.warning(f"Failed to read standings.json: {e} — falling back to DB")

    log.info("standings.json not found — computing from ipl.db")
    return _compute_from_db()


def _compute_from_db() -> list:
    """Fallback: compute points table from match results in ipl.db."""
    with db_connection() as conn:
        matches = pd.read_sql("""
            SELECT team1, team2, winner, result, result_margin, date, venue
            FROM matches
            WHERE season = '2026'
              AND (winner IS NOT NULL OR result IN ('no result', 'abandoned'))
            ORDER BY date
        """, conn)

    stats = defaultdict(lambda: {
        "P": 0, "W": 0, "L": 0, "NR": 0,
        "Pts": 0, "nrr_sum": 0.0, "form": []
    })

    for _, r in matches.iterrows():
        t1, t2 = r["team1"], r["team2"]
        w      = r["winner"]
        result = (r.get("result") or "").lower()

        if result in ("no result", "abandoned") or (pd.isna(w) and "tie" not in result):
            for t in [t1, t2]:
                stats[t]["P"]   += 1
                stats[t]["NR"]  += 1
                stats[t]["Pts"] += 1
            continue

        loser = t2 if w == t1 else t1
        for t in [t1, t2]:
            stats[t]["P"] += 1
        stats[w]["W"]     += 1
        stats[w]["Pts"]   += 2
        stats[loser]["L"] += 1

        margin = 0.0
        if pd.notna(r["result_margin"]):
            try:
                margin_val = float(r["result_margin"])
                if margin_val > 0:
                    if "run" in result:
                        margin = margin_val
                    elif "wicket" in result:
                        margin = margin_val * RUNS_PER_WICKET
                    elif "super over" in result:
                        margin = 1.0
            except (ValueError, TypeError):
                pass
        if w:
            stats[w]["nrr_sum"]    += margin
            stats[loser]["nrr_sum"] -= margin

    rows = []
    for team, s in stats.items():
        nrr = round(s["nrr_sum"] / s["P"], 3) if s["P"] > 0 else 0.0
        rows.append({
            "team": team,
            "M":    s["P"],
            "W":    s["W"],
            "L":    s["L"],
            "NR":   s["NR"],
            "Pts":  s["Pts"],
            "NRR":  nrr,
            "Form": "",
        })

    rows.sort(key=lambda x: (-x["Pts"], -x["NRR"]))
    return rows


def get_recent_matches() -> list:
    """Last 8 completed matches."""
    with db_connection() as conn:
        df = pd.read_sql("""
            SELECT date, team1, team2, winner,
                   result, result_margin, venue
            FROM matches
            WHERE season = '2026' AND winner IS NOT NULL
            ORDER BY date DESC
            LIMIT 8
        """, conn)
    return df.to_dict(orient="records")


def get_venues() -> dict:
    """All venue-team pair combinations for dropdown."""
    with db_connection() as conn:
        df = pd.read_sql(
            "SELECT team1, team2, venue FROM matches WHERE venue IS NOT NULL",
            conn
        )

    venue_map = defaultdict(set)
    for _, row in df.iterrows():
        key = f"{row['team1']}|||{row['team2']}"
        venue_map[key].add(row["venue"])

    return {k: sorted(v) for k, v in venue_map.items()}