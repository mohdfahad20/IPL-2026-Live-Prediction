"""
logger.py — Prediction & Tournament Winner Logger
===================================================
Reads the latest simulation results and match predictions from the DB
and appends a structured log entry to prediction_log.csv and winner_log.csv.

Run this after every simulation:
    python logger.py --db ipl.db

Or chain it in the pipeline:
    python -m simulate.simulate --db ipl.db && python logger.py --db ipl.db
"""

import argparse
import csv
import json
import pickle
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
from model.train import SoftEnsemble  # noqa: F401 — required for pickle

DB_PATH          = Path("ipl.db")
WINNER_LOG_PATH  = Path("winner_log.csv")
MATCH_LOG_PATH   = Path("match_predictions_log.csv")


def log_tournament_winner(conn, run_id=None):
    """Append latest simulation result to winner_log.csv."""
    query = "SELECT * FROM simulation_results ORDER BY run_at DESC LIMIT 1"
    import pandas as pd
    df = pd.read_sql(query, conn)
    if df.empty:
        print("No simulation results found.")
        return

    row      = df.iloc[0]
    probs    = json.loads(row["results_json"])
    leader   = list(probs.keys())[0]
    leader_p = list(probs.values())[0]

    write_header = not WINNER_LOG_PATH.exists()
    with open(WINNER_LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "logged_at", "run_id", "matches_played", "matches_remaining",
                "n_simulations", "predicted_winner", "win_probability",
                "top3_json"
            ])
        top3 = {k: v for k, v in list(probs.items())[:3]}
        w.writerow([
            datetime.utcnow().isoformat(),
            row["run_id"],
            row["matches_played"],
            row["matches_remaining"],
            row["n_simulations"],
            leader,
            round(leader_p, 4),
            json.dumps(top3),
        ])

    print(f"Tournament winner log → {WINNER_LOG_PATH}")
    print(f"  Leader: {leader} at {leader_p:.1%}")
    print("  Full standings:")
    for i, (team, p) in enumerate(probs.items(), 1):
        bar = "█" * int(p * 30)
        print(f"    {i:2d}. {team:38s} {p:5.1%}  {bar}")


def log_match_predictions(conn):
    """
    Log predictions for all 2026 matches that have an actual result,
    comparing what the model would have predicted (from features table)
    vs what actually happened.
    """
    import pandas as pd
    import numpy as np

    # Select ALL feature columns from features table + match metadata
    df = pd.read_sql("""
        SELECT f.*, m.winner, m.result, m.result_margin, m.venue, m.stage
        FROM features f
        JOIN matches m ON f.match_id = m.match_id
        WHERE f.season = '2026' AND f.target IS NOT NULL
        ORDER BY f.date ASC
    """, conn)

    if df.empty:
        print("No 2026 match features found.")
        return

    FEATURE_COLS = [
        "toss_won_team1", "toss_bat_team1", "is_playoff",
        "team1_form_last5", "team2_form_last5",
        "team1_season_winrate", "team2_season_winrate",
        "team1_season_played", "team2_season_played",
        "team1_season_points", "team2_season_points",
        "h2h_winrate_team1", "h2h_matches",
        "team1_venue_winrate", "team2_venue_winrate",
        "opponent_strength", "team1_avg_margin", "team2_avg_margin",
        "form_diff", "winrate_diff", "points_diff", "venue_diff", "margin_diff",
    ]

    # Add derived difference features (same as train.py)
    df["form_diff"]    = df["team1_form_last5"]     - df["team2_form_last5"]
    df["winrate_diff"] = df["team1_season_winrate"]  - df["team2_season_winrate"]
    df["points_diff"]  = df["team1_season_points"]   - df["team2_season_points"]
    df["venue_diff"]   = df["team1_venue_winrate"]   - df["team2_venue_winrate"]
    df["margin_diff"]  = df["team1_avg_margin"]      - df["team2_avg_margin"]

    # Load model
    with open(Path("models/model.pkl"), "rb") as f:
        model = pickle.load(f)

    X = df[FEATURE_COLS].fillna(0.5)

    probs  = model.predict_proba(X)[:, 1]
    preds  = (probs >= 0.5).astype(int)
    actual = df["target"].astype(int).values
    correct = (preds == actual).sum()

    write_header = not MATCH_LOG_PATH.exists()
    with open(MATCH_LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "logged_at", "match_id", "date", "team1", "team2",
                "predicted_winner", "actual_winner", "p_team1",
                "correct", "result", "result_margin", "venue"
            ])
        for i, row in df.iterrows():
            idx = df.index.get_loc(i)
            p   = round(float(probs[idx]), 4)
            predicted_winner = row["team1"] if p >= 0.5 else row["team2"]
            w.writerow([
                datetime.utcnow().isoformat(),
                row["match_id"], row["date"],
                row["team1"], row["team2"],
                predicted_winner, row["winner"],
                p,
                int(preds[idx] == actual[idx]),
                row.get("result", ""),
                row.get("result_margin", ""),
                row.get("venue", ""),
            ])

    accuracy = correct / len(df) if len(df) > 0 else 0
    print(f"\nMatch prediction log → {MATCH_LOG_PATH}")
    print(f"  2026 matches logged : {len(df)}")
    print(f"  Model accuracy      : {accuracy:.1%} ({correct}/{len(df)} correct)")


def main():
    parser = argparse.ArgumentParser(description="Log IPL predictions and tournament winner")
    parser.add_argument("--db", default=str(DB_PATH))
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        print("── Tournament Winner Log ────────────────────────────────────────")
        log_tournament_winner(conn)

        print("\n── Match Prediction Log ─────────────────────────────────────────")
        log_match_predictions(conn)
    finally:
        conn.close()

    print("\nDone. Check winner_log.csv and match_predictions_log.csv")


if __name__ == "__main__":
    main()