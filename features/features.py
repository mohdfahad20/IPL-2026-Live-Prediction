"""
Phase 2 — Feature Engineering
==============================
Reads the `matches` table from SQLite and computes match-level features
for every completed match. Saves results to a `features` table.

Features computed (all from the PERSPECTIVE of team1 in that match row):
  - Recent form      : win rate in last 5 matches (within same season)
  - Season form      : overall win rate in current season so far
  - H2H win rate     : historical head-to-head win rate vs opponent
  - Toss advantage   : did team win toss + elected to bat/field
  - Venue win rate   : team's win rate at this specific venue
  - Season points    : points accumulated so far this season (2 per win)
  - Opponent strength: opponent's win rate in current season so far
  - Is playoff       : whether match is a knockout stage match
  - Relative NRR proxy: approximated from result_margin (runs) so far

All features are computed using only data PRIOR to that match
(no data leakage — features at match time, not after).

Usage:
    python features.py               # reads ipl.db, writes features table
    python features.py --db ipl.db
    python features.py --export      # also exports features.csv for inspection
"""

import argparse
import sqlite3
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DB_PATH = Path("ipl.db")

# Playoff stage labels — used to create is_playoff flag
PLAYOFF_STAGES = {
    "Final", "Qualifier 1", "Qualifier 2", "Eliminator",
    "Semi Final", "Elimination Final", "3rd Place Play-Off"
}

# ─── SCHEMA ──────────────────────────────────────────────────────────────────

FEATURES_SCHEMA = """
CREATE TABLE IF NOT EXISTS features (
    match_id                TEXT    PRIMARY KEY,
    season                  TEXT,
    date                    TEXT,
    team1                   TEXT,
    team2                   TEXT,
    winner                  TEXT,
    target                  INTEGER,    -- 1 if team1 won, 0 if team2 won, NULL if no result

    -- Toss
    toss_won_team1          INTEGER,    -- 1 if team1 won toss
    toss_bat_team1          INTEGER,    -- 1 if team1 elected to bat

    -- Is playoff
    is_playoff              INTEGER,

    -- Recent form (last 5 matches within season)
    team1_form_last5        REAL,       -- win rate in last 5
    team2_form_last5        REAL,

    -- Season win rate so far (before this match)
    team1_season_winrate    REAL,
    team2_season_winrate    REAL,

    -- Season matches played so far (proxy for experience/fatigue)
    team1_season_played     INTEGER,
    team2_season_played     INTEGER,

    -- Season points so far (2 per win)
    team1_season_points     INTEGER,
    team2_season_points     INTEGER,

    -- Head-to-head win rate (all historical, before this match)
    h2h_winrate_team1       REAL,
    h2h_matches             INTEGER,

    -- Venue win rate (all historical, before this match)
    team1_venue_winrate     REAL,
    team2_venue_winrate     REAL,

    -- Opponent strength (season win rate — measures how tough opponent is)
    opponent_strength       REAL,       -- team2's season win rate (team1's perspective)

    -- NRR proxy: average run margin of victories this season
    team1_avg_margin        REAL,
    team2_avg_margin        REAL,

    created_at              TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_feat_season ON features(season);
CREATE INDEX IF NOT EXISTS idx_feat_team1  ON features(team1);
"""


# ─── LOAD DATA ───────────────────────────────────────────────────────────────

def load_matches(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql("""
        SELECT match_id, season, date, venue, city,
               team1, team2, toss_winner, toss_decision,
               winner, result, result_margin, stage
        FROM matches
        WHERE result != 'no result' OR result IS NULL
        ORDER BY date ASC, match_id ASC
    """, conn)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Normalise season to a sortable integer (use end year)
    df["season_year"] = df["season"].apply(parse_season_year)

    log.info(f"Loaded {len(df)} matches for feature engineering.")
    return df


def parse_season_year(s: str) -> int:
    """'2007/08' → 2008, '2024' → 2024"""
    s = str(s).strip()
    if "/" in s:
        return int(s.split("/")[0]) + 1
    return int(s)


# ─── FEATURE HELPERS ─────────────────────────────────────────────────────────

def get_team_matches_before(df: pd.DataFrame, team: str, before_idx: int) -> pd.DataFrame:
    """
    All completed matches involving `team` that occurred before row index `before_idx`.
    Returns a df with a `team_won` column (1 if team won, 0 if lost).
    """
    past = df.iloc[:before_idx]
    as_team1 = past[past["team1"] == team].copy()
    as_team2 = past[past["team2"] == team].copy()

    as_team1["team_won"] = (as_team1["winner"] == team).astype(int)
    as_team2["team_won"] = (as_team2["winner"] == team).astype(int)

    combined = pd.concat([as_team1, as_team2], ignore_index=True)
    combined = combined.sort_values("date")
    return combined


def get_team_season_matches_before(df: pd.DataFrame, team: str, season: str, before_idx: int) -> pd.DataFrame:
    """Season-scoped version of get_team_matches_before."""
    tm = get_team_matches_before(df, team, before_idx)
    return tm[tm["season"] == season]


def win_rate(team_matches: pd.DataFrame) -> float:
    if len(team_matches) == 0:
        return 0.5   # no prior info → assume 50/50
    return team_matches["team_won"].mean()


def last_n_win_rate(team_matches: pd.DataFrame, n: int = 5) -> float:
    recent = team_matches.tail(n)
    if len(recent) == 0:
        return 0.5
    return recent["team_won"].mean()


def avg_margin(team_matches: pd.DataFrame) -> float:
    """Average winning margin (in runs) for won matches — NRR proxy."""
    won = team_matches[team_matches["team_won"] == 1]
    margins = won["result_margin"].dropna()
    if len(margins) == 0:
        return 0.0
    return margins.mean()


def h2h_win_rate(df: pd.DataFrame, team1: str, team2: str, before_idx: int) -> tuple:
    """Head-to-head win rate of team1 vs team2 before this match."""
    past = df.iloc[:before_idx]
    h2h = past[
        ((past["team1"] == team1) & (past["team2"] == team2)) |
        ((past["team1"] == team2) & (past["team2"] == team1))
    ].copy()

    if len(h2h) == 0:
        return 0.5, 0

    h2h["team1_won"] = (h2h["winner"] == team1).astype(int)
    return h2h["team1_won"].mean(), len(h2h)


def venue_win_rate(df: pd.DataFrame, team: str, venue: str, before_idx: int) -> float:
    """Team's win rate at this specific venue."""
    past = df.iloc[:before_idx]
    at_venue = past[past["venue"] == venue].copy()

    as_t1 = at_venue[at_venue["team1"] == team].copy()
    as_t2 = at_venue[at_venue["team2"] == team].copy()
    as_t1["team_won"] = (as_t1["winner"] == team).astype(int)
    as_t2["team_won"] = (as_t2["winner"] == team).astype(int)

    combined = pd.concat([as_t1, as_t2])
    if len(combined) == 0:
        return 0.5
    return combined["team_won"].mean()


# ─── MAIN FEATURE BUILDER ────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Iterate over every match in chronological order and compute
    features using only data PRIOR to that match (no leakage).
    """
    records = []

    log.info("Building features for each match (this may take ~30s)...")

    for idx, row in df.iterrows():
        i        = df.index.get_loc(idx)   # positional index for slicing
        team1    = row["team1"]
        team2    = row["team2"]
        season   = row["season"]
        venue    = row["venue"]

        # ── Target variable ──────────────────────────────────────────────
        if row["winner"] == team1:
            target = 1
        elif row["winner"] == team2:
            target = 0
        else:
            target = None   # tie or no result — excluded from training

        # ── Toss features ────────────────────────────────────────────────
        toss_won_team1 = 1 if row["toss_winner"] == team1 else 0
        if row["toss_decision"] in ("bat", "batting"):
            toss_bat_team1 = 1 if row["toss_winner"] == team1 else 0
        else:
            toss_bat_team1 = 0 if row["toss_winner"] == team1 else 1

        # ── Is playoff ───────────────────────────────────────────────────
        is_playoff = 1 if str(row.get("stage", "")) in PLAYOFF_STAGES else 0

        # ── Historical match data for both teams ─────────────────────────
        t1_all    = get_team_matches_before(df, team1, i)
        t2_all    = get_team_matches_before(df, team2, i)
        t1_season = get_team_season_matches_before(df, team1, season, i)
        t2_season = get_team_season_matches_before(df, team2, season, i)

        # ── Form features ────────────────────────────────────────────────
        t1_form5  = last_n_win_rate(t1_season, 5)
        t2_form5  = last_n_win_rate(t2_season, 5)

        # ── Season win rate ──────────────────────────────────────────────
        t1_wr     = win_rate(t1_season)
        t2_wr     = win_rate(t2_season)

        # ── Season matches played ────────────────────────────────────────
        t1_played = len(t1_season)
        t2_played = len(t2_season)

        # ── Season points ────────────────────────────────────────────────
        t1_points = int(t1_season["team_won"].sum()) * 2
        t2_points = int(t2_season["team_won"].sum()) * 2

        # ── Head-to-head ─────────────────────────────────────────────────
        h2h_wr, h2h_n = h2h_win_rate(df, team1, team2, i)

        # ── Venue win rate ───────────────────────────────────────────────
        t1_venue = venue_win_rate(df, team1, venue, i) if pd.notna(venue) else 0.5
        t2_venue = venue_win_rate(df, team2, venue, i) if pd.notna(venue) else 0.5

        # ── NRR proxy ────────────────────────────────────────────────────
        t1_margin = avg_margin(t1_season)
        t2_margin = avg_margin(t2_season)

        records.append({
            "match_id":             row["match_id"],
            "season":               season,
            "date":                 row["date"].strftime("%Y-%m-%d"),
            "team1":                team1,
            "team2":                team2,
            "winner":               row["winner"],
            "target":               target,

            "toss_won_team1":       toss_won_team1,
            "toss_bat_team1":       toss_bat_team1,
            "is_playoff":           is_playoff,

            "team1_form_last5":     round(t1_form5, 4),
            "team2_form_last5":     round(t2_form5, 4),

            "team1_season_winrate": round(t1_wr, 4),
            "team2_season_winrate": round(t2_wr, 4),

            "team1_season_played":  t1_played,
            "team2_season_played":  t2_played,

            "team1_season_points":  t1_points,
            "team2_season_points":  t2_points,

            "h2h_winrate_team1":    round(h2h_wr, 4),
            "h2h_matches":          h2h_n,

            "team1_venue_winrate":  round(t1_venue, 4),
            "team2_venue_winrate":  round(t2_venue, 4),

            "opponent_strength":    round(t2_wr, 4),

            "team1_avg_margin":     round(t1_margin, 2),
            "team2_avg_margin":     round(t2_margin, 2),
        })

        if (i + 1) % 100 == 0:
            log.info(f"  Processed {i+1}/{len(df)} matches...")

    feat_df = pd.DataFrame(records)
    log.info(f"Features built for {len(feat_df)} matches.")
    return feat_df


# ─── SAVE TO DB ──────────────────────────────────────────────────────────────

def save_features(feat_df: pd.DataFrame, conn: sqlite3.Connection, export_csv: bool = False):
    conn.executescript(FEATURES_SCHEMA)
    conn.commit()

    # Drop and rebuild for clean re-runs
    conn.execute("DELETE FROM features")
    conn.commit()

    feat_df.to_sql("features", conn, if_exists="append", index=False)
    log.info(f"Saved {len(feat_df)} rows to `features` table.")

    if export_csv:
        feat_df.to_csv("features.csv", index=False)
        log.info("Exported features.csv for inspection.")


# ─── VALIDATE ────────────────────────────────────────────────────────────────

def validate(conn: sqlite3.Connection):
    total    = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    with_tgt = conn.execute("SELECT COUNT(*) FROM features WHERE target IS NOT NULL").fetchone()[0]
    t1_wins  = conn.execute("SELECT SUM(target) FROM features WHERE target IS NOT NULL").fetchone()[0]

    log.info("── Feature Validation ──────────────────────────────────────────")
    log.info(f"  Total feature rows     : {total}")
    log.info(f"  Rows with valid target : {with_tgt}")
    log.info(f"  Team1 win rate (raw)   : {t1_wins/with_tgt:.3f}  (expect ~0.50 if team assignment is random)")

    log.info("  Feature means (sanity check):")
    for col in [
        "team1_form_last5", "team2_form_last5",
        "team1_season_winrate", "h2h_winrate_team1",
        "team1_venue_winrate", "team1_avg_margin"
    ]:
        val = conn.execute(f"SELECT AVG({col}) FROM features WHERE target IS NOT NULL").fetchone()[0]
        log.info(f"    {col:30s}: {val:.4f}" if val else f"    {col}: None")

    log.info("  Sample 3 rows:")
    cols = "match_id, season, team1, team2, target, team1_form_last5, team1_season_winrate, h2h_winrate_team1"
    for row in conn.execute(f"SELECT {cols} FROM features LIMIT 3").fetchall():
        log.info(f"    {row}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build ML features from IPL matches table")
    parser.add_argument("--db",     default=str(DB_PATH), help="SQLite DB path")
    parser.add_argument("--export", action="store_true",  help="Also export features.csv")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        df       = load_matches(conn)
        feat_df  = build_features(df)
        save_features(feat_df, conn, export_csv=args.export)
        validate(conn)
        log.info(f"\nDone. Features table ready in: {Path(args.db).resolve()}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()