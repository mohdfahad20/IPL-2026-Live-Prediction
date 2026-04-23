"""
predict.py — Ensemble Match Win Probability Predictor
======================================================
Loads all 3 individual models + ensemble, returns both the
ensemble prediction and each model's individual probability.

Used by simulate.py (ensemble only) and dashboard (all 4 for display).
"""

import json
import logging
import pickle
import sqlite3

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from model.train import SoftEnsemble

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DB_PATH   = Path("ipl.db")
MODEL_DIR = Path("models")
META_PATH = MODEL_DIR / "model_meta.json"

_cache = {}   # module-level cache so Streamlit doesn't reload on every call


def _load(name):
    if name not in _cache:
        path = MODEL_DIR / name
        with open(path, "rb") as f:
            _cache[name] = pickle.load(f)
    return _cache[name]


def load_all_models():
    ensemble = _load("model_ensemble.pkl")
    xgb      = _load("model_xgb.pkl")
    rf       = _load("model_rf.pkl")
    lr       = _load("model_lr.pkl")
    with open(META_PATH) as f:
        meta = json.load(f)
    return ensemble, xgb, rf, lr, meta["feature_cols"]


# ─── FEATURE BUILDER ─────────────────────────────────────────────────────────

def build_features(team1, team2, season, venue, toss_winner, toss_decision,
                   is_playoff, conn):
    from features.features import (
        get_team_matches_before, get_team_season_matches_before,
        win_rate, last_n_win_rate, avg_margin, h2h_win_rate, venue_win_rate
    )

    all_m = pd.read_sql("""
        SELECT match_id, season, date, venue, team1, team2,
               toss_winner, toss_decision, winner, result, result_margin, stage
        FROM matches
        WHERE result != 'no result' OR result IS NULL
        ORDER BY date ASC, match_id ASC
    """, conn)
    all_m["date"] = pd.to_datetime(all_m["date"], errors="coerce")
    i = len(all_m)

    t1_all    = get_team_matches_before(all_m, team1, i)
    t2_all    = get_team_matches_before(all_m, team2, i)
    t1_season = get_team_season_matches_before(all_m, team1, season, i)
    t2_season = get_team_season_matches_before(all_m, team2, season, i)

    toss_won_t1 = 1 if toss_winner == team1 else 0
    if toss_decision in ("bat", "batting"):
        toss_bat_t1 = 1 if toss_winner == team1 else 0
    else:
        toss_bat_t1 = 0 if toss_winner == team1 else 1

    t1_wr     = win_rate(t1_season)
    t2_wr     = win_rate(t2_season)
    t1_form5  = last_n_win_rate(t1_season, 5)
    t2_form5  = last_n_win_rate(t2_season, 5)
    t1_played = len(t1_season)
    t2_played = len(t2_season)
    t1_pts    = int(t1_season["team_won"].sum()) * 2 if len(t1_season) else 0
    t2_pts    = int(t2_season["team_won"].sum()) * 2 if len(t2_season) else 0
    h2h_wr, h2h_n = h2h_win_rate(all_m, team1, team2, i)
    t1_venue  = venue_win_rate(all_m, team1, venue, i) if venue else 0.5
    t2_venue  = venue_win_rate(all_m, team2, venue, i) if venue else 0.5
    t1_margin = avg_margin(t1_season)
    t2_margin = avg_margin(t2_season)

    row = {
        "toss_won_team1":       toss_won_t1,
        "toss_bat_team1":       toss_bat_t1,
        "is_playoff":           int(is_playoff),
        "team1_form_last5":     t1_form5,
        "team2_form_last5":     t2_form5,
        "team1_season_winrate": t1_wr,
        "team2_season_winrate": t2_wr,
        "team1_season_played":  t1_played,
        "team2_season_played":  t2_played,
        "team1_season_points":  t1_pts,
        "team2_season_points":  t2_pts,
        "h2h_winrate_team1":    h2h_wr,
        "h2h_matches":          h2h_n,
        "team1_venue_winrate":  t1_venue,
        "team2_venue_winrate":  t2_venue,
        "opponent_strength":    t2_wr,
        "team1_avg_margin":     t1_margin,
        "team2_avg_margin":     t2_margin,
        "form_diff":    t1_form5  - t2_form5,
        "winrate_diff": t1_wr     - t2_wr,
        "points_diff":  t1_pts    - t2_pts,
        "venue_diff":   t1_venue  - t2_venue,
        "margin_diff":  t1_margin - t2_margin,
    }
    return pd.DataFrame([row])


# ─── PREDICT ─────────────────────────────────────────────────────────────────

def predict_match(team1, team2, season="2026", venue=None,
                  toss_winner=None, toss_decision=None,
                  is_playoff=False, conn=None, db_path=DB_PATH):
    """
    Returns dict with ensemble + per-model probabilities:
    {
        team1, team2,
        p_team1_wins,   ← ensemble (used by simulation)
        p_team2_wins,
        model_probs: {
            "XGBoost": 0.61,
            "Random Forest": 0.58,
            "Logistic Regression": 0.55,
            "Ensemble": 0.59,
        }
    }
    """
    ensemble, xgb, rf, lr, feat_cols = load_all_models()

    _conn = conn or sqlite3.connect(db_path)
    try:
        X = build_features(team1, team2, season, venue,
                           toss_winner, toss_decision, is_playoff, _conn)
    finally:
        if conn is None:
            _conn.close()

    X = X[feat_cols].fillna(0.5)

    p_xgb = float(xgb.predict_proba(X)[0, 1])
    p_rf  = float(rf.predict_proba(X)[0, 1])
    p_lr  = float(lr.predict_proba(X)[0, 1])
    p_ens = float(ensemble.predict_proba(X)[0, 1])

    return {
        "team1":        team1,
        "team2":        team2,
        "p_team1_wins": round(p_ens, 4),
        "p_team2_wins": round(1.0 - p_ens, 4),
        "model_probs": {
            "XGBoost":             round(p_xgb, 4),
            "Random Forest":       round(p_rf,  4),
            "Logistic Regression": round(p_lr,  4),
            "Ensemble":            round(p_ens, 4),
        }
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--team1", required=True)
    parser.add_argument("--team2", required=True)
    parser.add_argument("--season", default="2026")
    parser.add_argument("--venue", default=None)
    parser.add_argument("--db", default=str(DB_PATH))
    args = parser.parse_args()

    r = predict_match(team1=args.team1, team2=args.team2,
                      season=args.season, venue=args.venue,
                      db_path=Path(args.db))

    print(f"\n  {r['team1']:35s} → {r['p_team1_wins']:.1%}")
    print(f"  {r['team2']:35s} → {r['p_team2_wins']:.1%}")
    print("\n  Per-model breakdown:")
    for m, p in r["model_probs"].items():
        print(f"    {m:22s}: {p:.1%}")