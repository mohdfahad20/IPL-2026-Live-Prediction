import logging
import sys
import sqlite3
import pandas as pd
from pathlib import Path
from api.core.database import db_connection
from api.core.config import SEASON, ROOT_DIR

sys.path.insert(0, str(ROOT_DIR))

log = logging.getLogger(__name__)


def predict_match(
    team1: str,
    team2: str,
    season: str = SEASON,
    venue: str | None = None,
    toss_winner: str | None = None,
    toss_decision: str | None = None,
    is_playoff: bool = False,
) -> dict:
    # Import here so model_loader has already run _load_all
    # before model.predict tries to unpickle anything
    from api.core.model_loader import get_all_models
    from features.features import (
        get_team_matches_before,
        get_team_season_matches_before,
        win_rate, last_n_win_rate,
        avg_margin, h2h_win_rate, venue_win_rate,
    )

    ensemble, xgb, rf, lr, feat_cols = get_all_models()

    with db_connection() as conn:
        all_m = pd.read_sql("""
            SELECT match_id, season, date, venue, team1, team2,
                   toss_winner, toss_decision, winner, result, result_margin, stage
            FROM matches
            WHERE result != 'no result' OR result IS NULL
            ORDER BY date ASC, match_id ASC
        """, conn)

    all_m["date"] = pd.to_datetime(all_m["date"], errors="coerce")
    i = len(all_m)

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

    X = pd.DataFrame([row])[feat_cols].fillna(0.5)

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