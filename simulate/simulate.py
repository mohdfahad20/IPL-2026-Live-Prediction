"""
Phase 4 — Monte Carlo Simulation Engine
=========================================
Simulates the remaining IPL 2026 season 10,000 times and computes
tournament win probabilities per team.

v3 changes (playoff-aware):
  - League and playoff matches are handled separately throughout
  - build_remaining() only generates LEAGUE fixtures (cap=14 per team)
  - simulate_playoff() uses real results for already-completed playoff matches
  - simulate_one() only counts league matches for standings
  - play_or_lookup() prevents re-simulating real playoff outcomes

Usage:
    python simulate/simulate.py
    python simulate/simulate.py --n 10000 --db ipl.db
"""

import argparse
import json
import logging
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import random

import sys
sys.path.insert(0, str(Path(".").resolve()))
from model.train import SoftEnsemble  # noqa: F401 — required for pickle to load ensemble

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DB_PATH  = Path("ipl.db")
N_SIMS   = 10_000
SEASON   = "2026"

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

# Playoff stage labels — must match what scraper stores after the fix
PLAYOFF_STAGES = {"Qualifier 1", "Qualifier 2", "Eliminator", "Final"}

# Fixed bracket: match_no → stage
IPL_2026_PLAYOFF_MAP = {
    71: "Qualifier 1",
    72: "Eliminator",
    73: "Qualifier 2",
    74: "Final",
}

GROUP_MATCHES_TOTAL = 70   # matches 1–70 are league stage
MATCHES_PER_TEAM    = 14   # each team plays 14 league matches


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


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def is_playoff_match(row) -> bool:
    """True if this match is a playoff match (not a league match)."""
    stage = str(row.get("stage") or "").strip()
    # Also catch by match number in case stage wasn't set correctly
    try:
        match_no = int(row.get("event_match_no") or 0)
        if match_no >= 71:
            return True
    except (TypeError, ValueError):
        pass
    return stage in PLAYOFF_STAGES


# ─── LOAD COMPLETED 2026 MATCHES ─────────────────────────────────────────────

def load_completed(conn: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all played 2026 matches and split into:
      - league_completed  : matches 1–70 (used for standings)
      - playoff_completed : matches 71–74 (used to short-circuit bracket)

    Returns (league_completed, playoff_completed)
    """
    df = pd.read_sql("""
        SELECT match_id, date, team1, team2, winner, venue,
               stage, result, method, event_match_no
        FROM matches
        WHERE season = '2026'
        AND (
            winner IS NOT NULL
            OR result IS NOT NULL
            OR method IN ('no result', 'abandoned')
        )
        ORDER BY date ASC, match_id ASC
    """, conn)

    # Split
    league_mask  = ~df.apply(is_playoff_match, axis=1)
    league_done  = df[league_mask].copy().reset_index(drop=True)
    playoff_done = df[~league_mask].copy().reset_index(drop=True)

    log.info(f"Completed league matches  : {len(league_done)}")
    log.info(f"Completed playoff matches : {len(playoff_done)}")

    if not playoff_done.empty:
        for _, r in playoff_done.iterrows():
            log.info(f"  [{r['stage']}] {r['team1']} vs {r['team2']} → winner={r['winner']}")

    return league_done, playoff_done


# ─── BUILD REMAINING LEAGUE FIXTURES ─────────────────────────────────────────

def build_remaining(league_completed: pd.DataFrame) -> pd.DataFrame:
    """
    Generate remaining LEAGUE fixtures only (no playoffs — bracket handles those).
    Ensures each team ends with exactly MATCHES_PER_TEAM=14 league matches.
    """

    # Count played league matches per team and per pair
    team_played = defaultdict(int)
    pair_counts = Counter()

    for _, r in league_completed.iterrows():
        t1, t2 = r["team1"], r["team2"]
        team_played[t1] += 1
        team_played[t2] += 1
        pair_counts[tuple(sorted([t1, t2]))] += 1

    # Remaining league matches per team
    team_remaining = {
        t: max(0, MATCHES_PER_TEAM - team_played[t])
        for t in IPL_2026_TEAMS
    }

    # Log current state
    log.info("League match counts per team:")
    for t in sorted(IPL_2026_TEAMS):
        log.info(f"  {t:<35} played={team_played[t]:2d}  remaining={team_remaining[t]:2d}")

    # If all teams have played 14, group stage is complete
    if all(v == 0 for v in team_remaining.values()):
        log.info("Group stage complete — no league fixtures to simulate.")
        return pd.DataFrame(columns=["match_id", "date", "team1", "team2", "winner", "venue", "stage"])

    # Greedy fixture generation
    all_pairs = list(combinations(IPL_2026_TEAMS, 2))
    random.seed(42)
    random.shuffle(all_pairs)

    fixtures = []
    idx = 0

    for t1, t2 in all_pairs:
        pair = tuple(sorted([t1, t2]))
        already_played = pair_counts.get(pair, 0)
        games_to_add = min(
            2 - already_played,          # each pair plays max 2 times
            team_remaining[t1],
            team_remaining[t2],
        )
        for _ in range(max(0, games_to_add)):
            fixtures.append({
                "match_id": f"2026_gen_{idx:04d}",
                "date":     "2026-05-01",
                "team1":    t1,
                "team2":    t2,
                "winner":   None,
                "venue":    None,
                "stage":    "League",
            })
            team_remaining[t1] -= 1
            team_remaining[t2] -= 1
            idx += 1

    remaining = pd.DataFrame(fixtures)

    # ─── REPAIR: fill any deficit from greedy not filling all slots ───
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
        teams_with_deficit = list(deficit.keys())
        for i in range(len(teams_with_deficit)):
            for j in range(i + 1, len(teams_with_deficit)):
                t1, t2 = teams_with_deficit[i], teams_with_deficit[j]
                pair = tuple(sorted([t1, t2]))
                while deficit.get(t1, 0) > 0 and deficit.get(t2, 0) > 0:
                    if pair_counts.get(pair, 0) >= 2:
                        break
                    remaining = pd.concat([remaining, pd.DataFrame([{
                        "match_id": f"2026_fix_{idx:04d}",
                        "date":     "2026-05-01",
                        "team1":    t1,
                        "team2":    t2,
                        "winner":   None,
                        "venue":    None,
                        "stage":    "League",
                    }])], ignore_index=True)
                    deficit[t1] -= 1
                    deficit[t2] -= 1
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
                    idx += 1

    # Final summary
    log.info(f"Remaining league fixtures to simulate: {len(remaining)}")
    team_sim_final = defaultdict(int)
    for _, r in remaining.iterrows():
        team_sim_final[r["team1"]] += 1
        team_sim_final[r["team2"]] += 1
    for t in sorted(IPL_2026_TEAMS):
        total = team_played[t] + team_sim_final[t]
        log.info(f"  {t:<35} played={team_played[t]:2d}  simulated={team_sim_final[t]:2d}  total={total}")

    return remaining


# ─── GET MATCH PROBABILITIES ─────────────────────────────────────────────────

def get_match_probs(remaining: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    from model.predict import predict_match

    if remaining.empty:
        log.info("No remaining fixtures — skipping probability computation.")
        return remaining

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


# ─── PLAYOFF BRACKET SIMULATOR ────────────────────────────────────────────────

def play_or_lookup(
    stage: str,
    team_a: str,
    team_b: str,
    playoff_done: pd.DataFrame,
    prob_lookup: dict,
    rng: np.random.Generator,
) -> tuple[str, str]:
    """
    If this playoff stage is already in DB with a real result, use it.
    Otherwise simulate it using model probabilities.

    Returns (winner, loser).
    """
    # Check if this stage has a real result
    if not playoff_done.empty:
        match = playoff_done[playoff_done["stage"] == stage]
        if not match.empty:
            real_winner = match.iloc[0]["winner"]
            real_team1  = match.iloc[0]["team1"]
            real_team2  = match.iloc[0]["team2"]
            if pd.notna(real_winner):
                real_loser = real_team2 if real_winner == real_team1 else real_team1
                log.debug(f"[{stage}] Using real result: {real_winner} beat {real_loser}")
                return real_winner, real_loser

    # Not yet played — simulate
    p = prob_lookup.get((team_a, team_b),
        1.0 - prob_lookup.get((team_b, team_a), 0.5))
    winner = team_a if rng.random() < p else team_b
    loser  = team_b if winner == team_a else team_a
    return winner, loser


def simulate_playoff(
    ranked: list,
    playoff_done: pd.DataFrame,
    prob_lookup: dict,
    rng: np.random.Generator,
) -> str:
    """
    IPL playoff bracket:
      Q1: #1 vs #2  → winner → Final directly
      EL: #3 vs #4  → loser eliminated
      Q2: Q1-loser vs EL-winner → winner → Final
      Final: Q1-winner vs Q2-winner → champion

    Uses real results for already-completed playoff matches.
    """
    t1, t2, t3, t4 = ranked[0], ranked[1], ranked[2], ranked[3]

    q1_win,  q1_lose = play_or_lookup("Qualifier 1", t1, t2, playoff_done, prob_lookup, rng)
    el_win,  _       = play_or_lookup("Eliminator",  t3, t4, playoff_done, prob_lookup, rng)
    q2_win,  _       = play_or_lookup("Qualifier 2", q1_lose, el_win, playoff_done, prob_lookup, rng)
    champion, _      = play_or_lookup("Final",        q1_win, q2_win, playoff_done, prob_lookup, rng)

    return champion


# ─── SINGLE SEASON SIMULATION ────────────────────────────────────────────────

def simulate_one(
    league_completed: pd.DataFrame,
    remaining: pd.DataFrame,
    playoff_done: pd.DataFrame,
    all_teams: list,
    prob_lookup: dict,
    rng: np.random.Generator,
) -> str:
    """
    Simulate one full season. Returns champion team name.

    League standings are computed from LEAGUE matches only.
    Playoff bracket runs separately after standings are finalized.
    """

    # Init counters from real LEAGUE matches only
    wins   = {t: 0 for t in all_teams}
    played = {t: 0 for t in all_teams}

    for _, r in league_completed.iterrows():
        t1, t2, w = r["team1"], r["team2"], r["winner"]
        if t1 in played: played[t1] += 1
        if t2 in played: played[t2] += 1
        if pd.notna(w) and w in wins:
            wins[w] += 1

    # Simulate remaining LEAGUE matches
    if not remaining.empty:
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

    # Sort by points (2 per win) then win-rate as NRR proxy — LEAGUE only
    ranked = sorted(
        all_teams,
        key=lambda t: (wins[t] * 2, wins[t] / max(played[t], 1)),
        reverse=True
    )

    # Run playoff bracket (top 4 only)
    return simulate_playoff(ranked, playoff_done, prob_lookup, rng)


# ─── MONTE CARLO RUNNER ───────────────────────────────────────────────────────

def run_monte_carlo(
    league_completed: pd.DataFrame,
    remaining: pd.DataFrame,
    playoff_done: pd.DataFrame,
    all_teams: list,
    n: int = N_SIMS,
) -> dict:
    rng = np.random.default_rng(seed=42)

    # Pre-build probability lookup for O(1) access in simulation loop
    prob_lookup = {}
    if not remaining.empty:
        for _, r in remaining.iterrows():
            prob_lookup[(r["team1"], r["team2"])] = r["p_team1"]

    win_counts = {t: 0 for t in all_teams}

    log.info(f"Running {n:,} simulations...")
    for i in range(n):
        champion = simulate_one(
            league_completed, remaining, playoff_done,
            all_teams, prob_lookup, rng
        )
        if champion in win_counts:
            win_counts[champion] += 1
        if (i + 1) % 2000 == 0:
            log.info(f"  {i+1:,} / {n:,} done...")

    probs = {t: round(win_counts[t] / n, 4) for t in all_teams}
    return dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))


# ─── SAVE + DISPLAY ──────────────────────────────────────────────────────────

def save_results(probs, n, n_league_completed, n_remaining, n_playoff_done, conn):
    conn.executescript(RESULTS_SCHEMA)
    conn.commit()
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    matches_played    = n_league_completed + n_playoff_done
    matches_remaining = n_remaining + (4 - n_playoff_done)  # remaining league + remaining playoffs
    conn.execute(
        "INSERT INTO simulation_results VALUES (?,?,?,?,?,?)",
        (run_id, datetime.utcnow().isoformat(), n,
         matches_played, matches_remaining, json.dumps(probs))
    )
    conn.commit()
    log.info(f"Saved → run_id: {run_id}")
    return run_id


def display(probs, n_league_completed, n_remaining, n_playoff_done):
    total_played = n_league_completed + n_playoff_done
    total_remaining = n_remaining + (4 - n_playoff_done)
    log.info("\n" + "=" * 58)
    log.info("  IPL 2026 TOURNAMENT WIN PROBABILITIES")
    log.info(f"  League: {n_league_completed}/70  |  Playoffs: {n_playoff_done}/4")
    log.info(f"  Total played: {total_played}  |  Remaining: {total_remaining}")
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
        # Split completed matches into league vs playoff
        league_completed, playoff_done = load_completed(conn)

        # Build remaining LEAGUE fixtures only
        remaining = build_remaining(league_completed)

        # Get model probabilities for remaining league fixtures
        remaining = get_match_probs(remaining, conn)

        # Run Monte Carlo
        probs = run_monte_carlo(
            league_completed, remaining, playoff_done,
            IPL_2026_TEAMS, n=args.n
        )

        save_results(
            probs, args.n,
            len(league_completed), len(remaining), len(playoff_done),
            conn
        )
        display(probs, len(league_completed), len(remaining), len(playoff_done))

    finally:
        conn.close()

    log.info("\nPhase 4 complete. Ready for Phase 5 (dashboard).")


if __name__ == "__main__":
    main()