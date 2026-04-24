import json
import logging
from api.core.database import db_connection

log = logging.getLogger(__name__)


def get_latest_probabilities() -> dict:
    """
    Return ONLY the latest simulation run.
    Uses run_id (not run_at) to avoid ordering bugs.
    """
    with db_connection() as conn:
        row = conn.execute("""
            SELECT run_id, run_at, n_simulations,
                   matches_played, matches_remaining, results_json
            FROM simulation_results
            ORDER BY run_id DESC
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
    Return history of simulation runs.
    Keeps ONLY latest run per matches_played value.
    Uses run_id ordering (safe).
    """
    with db_connection() as conn:
        rows = conn.execute("""
            SELECT run_id, run_at, n_simulations,
                   matches_played, matches_remaining, results_json
            FROM simulation_results
            WHERE matches_played <= 70
            ORDER BY run_id ASC
        """).fetchall()

    # Deduplicate → latest per matches_played
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