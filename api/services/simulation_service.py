import json
import logging
from api.core.database import db_connection

log = logging.getLogger(__name__)


def get_latest_probabilities() -> dict:
    """Latest simulation run from simulation_results table."""
    with db_connection() as conn:
        row = conn.execute("""
            SELECT run_id, run_at, n_simulations,
                   matches_played, matches_remaining, results_json
            FROM simulation_results
            ORDER BY run_at DESC
            LIMIT 1
        """).fetchone()

    if not row:
        return {}

    return {
        "run_id":            row[0],
        "run_at":            row[1],
        "n_simulations":     row[2],
        "matches_played":    row[3],
        "matches_remaining": row[4],
        "results":           json.loads(row[5]),
    }


def get_probability_history() -> list:
    """
    All simulation runs for trend chart.
    One entry per unique matches_played value (latest run only).
    """
    with db_connection() as conn:
        rows = conn.execute("""
            SELECT run_id, run_at, n_simulations,
                   matches_played, matches_remaining, results_json
            FROM simulation_results
            WHERE matches_played <= 70
            ORDER BY run_at ASC
        """).fetchall()

    # Deduplicate — keep latest run per matches_played count
    seen: dict = {}
    for row in rows:
        key = row[3]  # matches_played
        seen[key] = {
            "run_id":            row[0],
            "run_at":            row[1],
            "n_simulations":     row[2],
            "matches_played":    row[3],
            "matches_remaining": row[4],
            "results":           json.loads(row[5]),
        }

    return list(seen.values())