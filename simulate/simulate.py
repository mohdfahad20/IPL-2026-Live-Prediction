"""
Phase 4 — Monte Carlo Simulation Engine
=========================================
Simulates the remaining IPL 2026 season 10,000 times and computes
tournament win probabilities per team.

Key fix from v1: always generate the full remaining schedule from the
round-robin template, subtracting already-completed matches.
This ensures all 10 teams appear in every simulation even early in
the season.

Usage:
    python simulate/simulate.py
    python simulate/simulate.py --n 10000 --db ipl.db
"""

import argparse
import json
import logging
import sqlite3
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(".").resolve()))
from model.train import SoftEnsemble  # noqa: F401 — required for pickle to load ensemble

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DB_PATH  = Path("ipl.db")
N_SIMS   = 10_000
SEASON   = "2026"

# All 10 IPL 2026 teams — used as ground truth for schedule generation
IPL_2026_TEAMS = [
    "Mumbai Indians",
    "Chennai Super Kings",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Delhi Capitals",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Punjab Kings",
    "Gujarat Titans",
    "Lucknow Super Giants",
]

# Each pair plays twice → 10C2 * 2 = 90 group matches
# IPL actual group stage = 70 (each team plays 14), not full round-robin
# We approximate: each pair plays once = 45 matches, close to real schedule
# Adjust GROUP_MATCHES to match the actual 2026 format once known
GROUP_MATCHES_TOTAL = 70   # standard IPL group stage


# ─── SCHEMA ──────────────────────────────────────────────────────────────────

RESULTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS simulation_results (
    run_id              TEXT    PRIMARY KEY,
    run_at              TEXT    NOT NULL,
    n_simulations       INTEGER NOT NULL,
    matches_played      INTEGER,
    matches_remaining   INTEGER,
    results_json        TEXT    NOT NULL
);
"""


# ─── LOAD COMPLETED 2026 MATCHES ─────────────────────────────────────────────

def load_completed(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load ALL played matches for IPL 2026:
    - includes wins/losses
    - includes no result / abandoned matches
    """

    df = pd.read_sql("""
        SELECT match_id, date, team1, team2, winner, venue, stage, result, method
        FROM matches
        WHERE season = '2026'
        AND (
            winner IS NOT NULL
            OR result IS NOT NULL
            OR method IN ('no result', 'abandoned')
        )
        ORDER BY date ASC, match_id ASC
    """, conn)

    log.info(f"Completed (including NR) 2026 matches in DB: {len(df)}")
    return df


# ─── BUILD REMAINING FIXTURE LIST ────────────────────────────────────────────

def build_remaining(completed: pd.DataFrame) -> pd.DataFrame:
    """
    Generate remaining fixtures ensuring:
    - Each team ends with exactly 14 matches
    - Each pair plays at most twice
    - Total schedule remains consistent
    """

    from collections import Counter, defaultdict
    from itertools import combinations
    import random
    import pandas as pd

    MATCHES_PER_TEAM = 14

    # ─── STEP 1: Count played matches ───
    team_played = defaultdict(int)
    pair_counts = Counter()

    for _, r in completed.iterrows():
        t1, t2 = r["team1"], r["team2"]

        # COUNT ALL matches (including NR)
        team_played[t1] += 1
        team_played[t2] += 1
        pair_counts[tuple(sorted([t1, t2]))] += 1

    # ─── STEP 2: Remaining matches per team ───
    team_remaining = {
        t: max(0, MATCHES_PER_TEAM - team_played[t])
        for t in IPL_2026_TEAMS
    }

    # ─── STEP 3: Generate fixtures (greedy) ───
    all_pairs = list(combinations(IPL_2026_TEAMS, 2))
    random.seed(42)
    random.shuffle(all_pairs)

    fixtures = []
    idx = 0

    for t1, t2 in all_pairs:
        pair = tuple(sorted([t1, t2]))
        already_played = pair_counts.get(pair, 0)
        max_pair_plays = 2

        games_to_add = min(
            max_pair_plays - already_played,
            team_remaining[t1],
            team_remaining[t2],
        )

        for _ in range(max(0, games_to_add)):
            fixtures.append({
                "match_id": f"2026_gen_{idx:04d}",
                "date": "2026-05-01",
                "team1": t1,
                "team2": t2,
                "winner": None,
                "venue": None,
                "stage": "Unknown",
            })
            team_remaining[t1] -= 1
            team_remaining[t2] -= 1
            idx += 1

    remaining = pd.DataFrame(fixtures)

    # ─── STEP 4: REPAIR (CRITICAL FIX) ───
    team_sim = defaultdict(int)
    for _, r in remaining.iterrows():
        team_sim[r["team1"]] += 1
        team_sim[r["team2"]] += 1

    deficit = {}
    for t in IPL_2026_TEAMS:
        total = team_played[t] + team_sim[t]
        if total < MATCHES_PER_TEAM:
            deficit[t] = MATCHES_PER_TEAM - total

    if deficit:
        log.warning(f"Repairing incomplete schedule: {deficit}")

        teams = list(deficit.keys())

        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                t1, t2 = teams[i], teams[j]

                pair = tuple(sorted([t1, t2]))
                if pair_counts[pair] >= 2:
                    continue

                while deficit[t1] > 0 and deficit[t2] > 0:
                    remaining = pd.concat([
                        remaining,
                        pd.DataFrame([{
                            "match_id": f"2026_fix_{idx:04d}",
                            "date": "2026-05-01",
                            "team1": t1,
                            "team2": t2,
                            "winner": None,
                            "venue": None,
                            "stage": "Unknown",
                        }])
                    ], ignore_index=True)

                    deficit[t1] -= 1
                    deficit[t2] -= 1
                    pair_counts[pair] += 1
                    idx += 1

    # ─── STEP 5: FINAL LOGGING ───
    log.info(f"Remaining group fixtures to simulate: {len(remaining)}")

    team_sim = defaultdict(int)
    for _, r in remaining.iterrows():
        team_sim[r["team1"]] += 1
        team_sim[r["team2"]] += 1

    for t in sorted(IPL_2026_TEAMS):
        total = team_played[t] + team_sim[t]
        log.info(f"  {t:<35} played={team_played[t]:2d}  simulated={team_sim[t]:2d}  total={total}")

    return remaining


# ─── GET MATCH PROBABILITIES ─────────────────────────────────────────────────

def get_match_probs(remaining: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    from model.predict import predict_match

    log.info(f"Computing model probabilities for {len(remaining)} fixtures...")

    probs = []
    for _, row in remaining.iterrows():
        res = predict_match(
            team1  = row["team1"],
            team2  = row["team2"],
            season = SEASON,
            venue  = row.get("venue"),
            conn   = conn,
        )
        probs.append(res["p_team1_wins"])

    remaining = remaining.copy()
    remaining["p_team1"] = probs
    log.info("  Probabilities computed.")
    return remaining


# ─── PLAYOFF SIMULATOR ───────────────────────────────────────────────────────

def simulate_playoff(ranked: list, prob_lookup: dict, rng: np.random.Generator) -> str:
    """
    IPL playoff bracket:
      Q1:    #1 vs #2  → winner → Final
      EL:    #3 vs #4  → loser eliminated
      Q2:    Q1-loser vs EL-winner → winner → Final
      Final: Q1-winner vs Q2-winner → champion
    """
    def play(a, b):
        p = prob_lookup.get((a, b), 1.0 - prob_lookup.get((b, a), 0.5))
        return a if rng.random() < p else b

    t1, t2, t3, t4 = ranked[0], ranked[1], ranked[2], ranked[3]

    q1_win  = play(t1, t2)
    q1_lose = t2 if q1_win == t1 else t1
    el_win  = play(t3, t4)
    q2_win  = play(q1_lose, el_win)
    return play(q1_win, q2_win)


# ─── SINGLE SEASON SIMULATION ────────────────────────────────────────────────

def simulate_one(
    completed: pd.DataFrame,
    remaining: pd.DataFrame,
    all_teams: list,
    prob_lookup: dict,
    rng: np.random.Generator,
) -> str:
    """Simulate one full season. Returns champion team name."""

    # Initialise win/played counters from real completed matches
    wins   = {t: 0 for t in all_teams}
    played = {t: 0 for t in all_teams}

    for _, r in completed.iterrows():
        t1, t2, w = r["team1"], r["team2"], r["winner"]
        if t1 in played: played[t1] += 1
        if t2 in played: played[t2] += 1
        if pd.notna(w) and w in wins:
            wins[w] += 1

    # Sample remaining group matches
    p_arr  = remaining["p_team1"].values
    t1_arr = remaining["team1"].values
    t2_arr = remaining["team2"].values
    rolls  = rng.random(len(remaining))

    for i in range(len(remaining)):
        t1, t2 = t1_arr[i], t2_arr[i]
        winner = t1 if rolls[i] < p_arr[i] else t2
        if t1 in played: played[t1] += 1
        if t2 in played: played[t2] += 1
        if winner in wins: wins[winner] += 1

    # Sort by points (2 per win) then win-rate as NRR proxy
    ranked = sorted(
        all_teams,
        key=lambda t: (wins[t] * 2, wins[t] / max(played[t], 1)),
        reverse=True
    )

    return simulate_playoff(ranked, prob_lookup, rng)


# ─── MONTE CARLO RUNNER ───────────────────────────────────────────────────────

def run_monte_carlo(
    completed: pd.DataFrame,
    remaining: pd.DataFrame,
    all_teams: list,
    n: int = N_SIMS,
) -> dict:
    rng = np.random.default_rng(seed=42)

    # Pre-build probability lookup dict for O(1) access inside the loop
    prob_lookup = {}
    for _, r in remaining.iterrows():
        prob_lookup[(r["team1"], r["team2"])] = r["p_team1"]

    win_counts = {t: 0 for t in all_teams}

    log.info(f"Running {n:,} simulations...")
    for i in range(n):
        champion = simulate_one(completed, remaining, all_teams, prob_lookup, rng)
        if champion in win_counts:
            win_counts[champion] += 1
        if (i + 1) % 2000 == 0:
            log.info(f"  {i+1:,} / {n:,} done...")

    probs = {t: round(win_counts[t] / n, 4) for t in all_teams}
    return dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))


# ─── SAVE + DISPLAY ──────────────────────────────────────────────────────────

def save_results(probs, n, n_completed, n_remaining, conn):
    conn.executescript(RESULTS_SCHEMA)
    conn.commit()
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    conn.execute(
        "INSERT INTO simulation_results VALUES (?,?,?,?,?,?)",
        (run_id, datetime.utcnow().isoformat(), n, n_completed, n_remaining, json.dumps(probs))
    )
    conn.commit()
    log.info(f"Saved → run_id: {run_id}")
    return run_id


def display(probs, n_completed, n_remaining):
    log.info("\n" + "=" * 58)
    log.info("  IPL 2026 TOURNAMENT WIN PROBABILITIES")
    log.info(f"  {n_completed} matches played  |  {n_remaining} remaining")
    log.info("=" * 58)
    for i, (team, p) in enumerate(probs.items(), 1):
        bar = "█" * int(p * 40)
        log.info(f"  {i:2d}. {team:38s} {p:5.1%}  {bar}")
    log.info("=" * 58)


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--n",  type=int, default=N_SIMS)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        completed = load_completed(conn)
        remaining = build_remaining(completed)
        remaining = get_match_probs(remaining, conn)

        probs = run_monte_carlo(completed, remaining, IPL_2026_TEAMS, n=args.n)

        save_results(probs, args.n, len(completed), len(remaining), conn)
        display(probs, len(completed), len(remaining))

    finally:
        conn.close()

    log.info("\nPhase 4 complete. Ready for Phase 5 (dashboard).")


if __name__ == "__main__":
    main()