"""
Phase 1 — Step 1 (v2): Kaggle Ball-by-Ball Data Loader
=======================================================
The Kaggle dataset is ball-by-ball (one delivery per row).
This script:
  1. Reads the CSV
  2. Collapses to ONE ROW PER MATCH by taking the first delivery
     of each match_id (all match-level columns are identical across rows)
  3. Parses win margin (runs/wickets) from win_outcome column
  4. Creates and populates the SQLite `matches` table

Usage:
    python load_kaggle_data.py --csv path/to/your_file.csv
    python load_kaggle_data.py --csv path/to/your_file.csv --db ipl.db
"""

import argparse
import sqlite3
import re
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── CONFIG ──────────────────────────────────────────────────────────────────

DB_PATH = Path("ipl.db")

# Map raw CSV column names → internal DB names
# Left = exactly as in your CSV header, Right = our DB column name
COLUMN_MAP = {
    "match_id":         "match_id",
    "date":             "date",
    "season":           "season",
    "venue":            "venue",
    "city":             "city",
    "toss_winner":      "toss_winner",
    "toss_decision":    "toss_decision",
    "match_won_by":     "winner",
    "win_outcome":      "win_outcome",      # e.g. "140 runs", "6 wickets"
    "player_of_match":  "player_of_match",
    "result_type":      "result_type",      # "normal" | "tie" | "no result"
    "method":           "method",           # DLS etc.
    "stage":            "stage",            # "Unknown" | "Final" | "Qualifier 1"
    "event_match_no":   "event_match_no",
    "superover_winner": "superover_winner", # winner when match tied and went to super over
}

# Team columns — used to reconstruct team1 (batting inn1) and team2 (bowling inn1)
BATTING_COL = "batting_team"
BOWLING_COL = "bowling_team"
INNINGS_COL = "innings"

# ─── TEAM NAME NORMALISER ─────────────────────────────────────────────────────

TEAM_ALIASES = {
    "Delhi Daredevils":             "Delhi Capitals",
    "Deccan Chargers":              "Sunrisers Hyderabad",
    "Rising Pune Supergiants":      "Rising Pune Supergiant",
    "Pune Warriors":                "Pune Warriors",
    "Kochi Tuskers Kerala":         "Kochi Tuskers Kerala",
    "Kings XI Punjab":              "Punjab Kings",
    "Royal Challengers Bangalore":  "Royal Challengers Bengaluru",
}

def norm(name) -> str:
    if pd.isna(name):
        return name
    return TEAM_ALIASES.get(str(name).strip(), str(name).strip())


# ─── SCHEMA ──────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    match_id            TEXT    PRIMARY KEY,
    season              TEXT    NOT NULL,
    date                TEXT    NOT NULL,
    venue               TEXT,
    city                TEXT,
    team1               TEXT    NOT NULL,
    team2               TEXT    NOT NULL,
    toss_winner         TEXT,
    toss_decision       TEXT,
    winner              TEXT,
    result              TEXT,
    result_margin       REAL,
    player_of_match     TEXT,
    method              TEXT,
    stage               TEXT,
    event_match_no      TEXT,
    source              TEXT    DEFAULT 'kaggle',
    created_at          TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_season  ON matches(season);
CREATE INDEX IF NOT EXISTS idx_team1   ON matches(team1);
CREATE INDEX IF NOT EXISTS idx_team2   ON matches(team2);
CREATE INDEX IF NOT EXISTS idx_date    ON matches(date);
CREATE INDEX IF NOT EXISTS idx_winner  ON matches(winner);
"""


# ─── WIN OUTCOME PARSER ───────────────────────────────────────────────────────

def parse_win_outcome(win_outcome, result_type) -> tuple:
    """
    Parse win_outcome string into (result, margin).

    "140 runs"  → ("runs",    140.0)
    "6 wickets" → ("wickets",   6.0)
    "tie"       → ("tie",      None)
    NaN         → (None,       None)
    """
    if pd.isna(win_outcome):
        if pd.notna(result_type):
            rt = str(result_type).lower().strip()
            if "tie" in rt:
                return "tie", None
            if "no result" in rt or "abandon" in rt:
                return "no result", None
        return None, None

    s = str(win_outcome).lower().strip()

    m = re.search(r"(\d+)\s+run", s)
    if m:
        return "runs", float(m.group(1))

    m = re.search(r"(\d+)\s+wicket", s)
    if m:
        return "wickets", float(m.group(1))

    if "tie" in s or "super over" in s:
        return "tie", None

    if "no result" in s or "abandon" in s:
        return "no result", None

    return None, None


# ─── COLLAPSE BALL-BY-BALL → MATCH LEVEL ─────────────────────────────────────

def collapse_to_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    One ball per row → one match per row.
    Match-level columns are identical across all balls of the same match,
    so we just take the first row per match_id.
    Team1 = batting team in innings 1, Team2 = bowling team in innings 1.
    """
    log.info("Collapsing ball-by-ball to match-level...")

    # Get team1 and team2 from first ball of innings 1
    inn1       = df[df[INNINGS_COL] == 1]
    first_ball = inn1.groupby("match_id").first().reset_index()[
        ["match_id", BATTING_COL, BOWLING_COL]
    ]

    # All match-level columns — take first row per match (values are the same on every ball)
    match_cols = ["match_id"] + [c for c in COLUMN_MAP if c != "match_id" and c in df.columns]
    match_df   = df.groupby("match_id").first().reset_index()[match_cols]

    # Merge team names in
    merged = match_df.merge(first_ball, on="match_id", how="left")

    log.info(f"  Raw deliveries : {len(df):,}")
    log.info(f"  Unique matches : {len(merged):,}")
    return merged


# ─── CLEAN ───────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Rename to internal names
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

    # Team names (from collapsed innings-1 cols)
    df["team1"]       = df[BATTING_COL].apply(norm)
    df["team2"]       = df[BOWLING_COL].apply(norm)
    df["winner"]      = df["winner"].apply(norm)
    df["toss_winner"] = df["toss_winner"].apply(norm) if "toss_winner" in df.columns else None

    # Parse result type and margin
    outcomes = df.apply(
        lambda r: parse_win_outcome(r.get("win_outcome"), r.get("result_type")),
        axis=1, result_type="expand"
    )
    df["result"]        = outcomes[0]
    df["result_margin"] = outcomes[1]

    # Super over fix: if match tied but superover_winner is known, use that as winner
    if "superover_winner" in df.columns:
        na_vals = {"NA", "nan", "none", ""}
        has_superover = (
            (df["result"] == "tie") &
            df["superover_winner"].notna() &
            (~df["superover_winner"].astype(str).str.strip().str.lower().isin(na_vals))
        )
        fixed = has_superover.sum()
        if fixed:
            df.loc[has_superover, "winner"] = df.loc[has_superover, "superover_winner"].apply(norm)
            df.loc[has_superover, "result"] = "super over"
            log.info(f"  Super over fix: filled winner for {fixed} tied matches.")

    # True ties and no-results (no superover winner) remain with winner = None
    df.loc[df["result"].isin(["tie", "no result"]), "winner"] = None

    # Date → ISO YYYY-MM-DD
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Scrub NA-like strings in text columns
    na_strings = {"NA", "nan", "none", ""}
    for col in ["stage", "method", "player_of_match", "city", "venue", "event_match_no"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: None if (pd.isna(x) or str(x).strip().lower() in na_strings) else x
            )

    df["source"] = "kaggle"

    log.info(f"  After clean: {len(df)} matches")
    return df


# ─── INSERT ──────────────────────────────────────────────────────────────────

FINAL_COLS = [
    "match_id", "season", "date", "venue", "city",
    "team1", "team2", "toss_winner", "toss_decision",
    "winner", "result", "result_margin",
    "player_of_match", "method", "stage", "event_match_no", "source"
]

def insert(df: pd.DataFrame, conn: sqlite3.Connection):
    cols = [c for c in FINAL_COLS if c in df.columns]
    inserted = skipped = errors = 0

    for _, row in df[cols].iterrows():
        values = [None if (pd.isna(v) if not isinstance(v, str) else False) else v for v in row]
        try:
            conn.execute(
                f"INSERT OR IGNORE INTO matches ({', '.join(cols)}) "
                f"VALUES ({', '.join(['?'] * len(cols))})",
                values
            )
            changed = conn.execute("SELECT changes()").fetchone()[0]
            inserted += changed
            skipped  += (1 - changed)
        except Exception as e:
            log.warning(f"Row error ({row.get('match_id', '?')}): {e}")
            errors += 1

    conn.commit()
    log.info(f"  Inserted: {inserted}  |  Skipped (duplicate): {skipped}  |  Errors: {errors}")


# ─── VALIDATE ────────────────────────────────────────────────────────────────

def validate(conn: sqlite3.Connection):
    log.info("── Validation Report ────────────────────────────────────────")
    log.info(f"  Total matches : {conn.execute('SELECT COUNT(*) FROM matches').fetchone()[0]}")
    log.info(f"  Seasons       : {conn.execute('SELECT COUNT(DISTINCT season) FROM matches').fetchone()[0]}")
    log.info(f"  Distinct teams: {conn.execute('SELECT COUNT(DISTINCT t) FROM (SELECT team1 as t FROM matches UNION SELECT team2 FROM matches)').fetchone()[0]}")

    log.info("  Per-season match count:")
    for row in conn.execute("SELECT season, COUNT(*) FROM matches GROUP BY season ORDER BY season"):
        log.info(f"    {row[0]}: {row[1]} matches")

    log.info("  Result type breakdown:")
    for row in conn.execute("SELECT result, COUNT(*) FROM matches GROUP BY result ORDER BY result"):
        log.info(f"    {row[0]}: {row[1]}")

    log.info("  Stage breakdown:")
    for row in conn.execute("SELECT stage, COUNT(*) FROM matches GROUP BY stage ORDER BY COUNT(*) DESC"):
        log.info(f"    {row[0]}: {row[1]}")

    log.info("  Sample 3 rows:")
    for row in conn.execute(
        "SELECT match_id, season, date, team1, team2, winner, result, result_margin FROM matches LIMIT 3"
    ):
        log.info(f"    {row}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Load IPL ball-by-ball CSV into SQLite match table")
    parser.add_argument("--csv", required=True, help="Path to the ball-by-ball CSV")
    parser.add_argument("--db",  default=str(DB_PATH), help="SQLite DB path (default: ipl.db)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    log.info(f"Loading: {csv_path}")
    df_raw = pd.read_csv(csv_path, low_memory=False)
    log.info(f"  Raw shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

    conn = sqlite3.connect(args.db)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
        log.info("Schema initialised.")

        df = collapse_to_matches(df_raw)
        df = clean(df)
        insert(df, conn)
        validate(conn)

        log.info(f"\nDone. Database: {Path(args.db).resolve()}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()