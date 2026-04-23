"""
Phase 3 — Ensemble Model Training
====================================
Trains three models (XGBoost, Random Forest, Logistic Regression),
combines them via soft-voting ensemble, calibrates probabilities,
and saves everything to models/.

Saved files:
  models/model_xgb.pkl       — calibrated XGBoost
  models/model_rf.pkl        — calibrated Random Forest
  models/model_lr.pkl        — calibrated Logistic Regression
  models/model_ensemble.pkl  — soft-voting ensemble (used for predictions)
  models/model_meta.json     — CV scores, feature importances, metadata

Usage:
    python model/train.py
    python model/train.py --db ipl.db
"""

import argparse
import json
import logging
import pickle
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DB_PATH   = Path("ipl.db")
MODEL_DIR = Path("models")

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
TARGET_COL = "target"


# ─── SOFT ENSEMBLE (module-level so pickle can find it) ──────────────────────

class SoftEnsemble:
    """Weighted average of predict_proba from multiple calibrated models."""
    def __init__(self, estimators, weights=None):
        self.estimators = estimators        # list of (name, model)
        self.weights    = weights or [1.0] * len(estimators)
        self.classes_   = np.array([0, 1])

    def predict_proba(self, X):
        total_w = sum(self.weights)
        probs   = np.zeros((len(X), 2))
        for (name, m), w in zip(self.estimators, self.weights):
            probs += m.predict_proba(X) * w
        return probs / total_w

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ─── LOAD ────────────────────────────────────────────────────────────────────

def load_features(conn):
    df = pd.read_sql(
        "SELECT * FROM features WHERE target IS NOT NULL ORDER BY date ASC", conn
    )
    log.info(f"Loaded {len(df)} labelled rows.")
    return df


def add_derived_features(df):
    df = df.copy()
    df["form_diff"]    = df["team1_form_last5"]    - df["team2_form_last5"]
    df["winrate_diff"] = df["team1_season_winrate"] - df["team2_season_winrate"]
    df["points_diff"]  = df["team1_season_points"]  - df["team2_season_points"]
    df["venue_diff"]   = df["team1_venue_winrate"]  - df["team2_venue_winrate"]
    df["margin_diff"]  = df["team1_avg_margin"]     - df["team2_avg_margin"]
    return df


# ─── SEASON HELPERS ──────────────────────────────────────────────────────────

def season_key(s):
    s = str(s)
    return int(s.split("/")[0]) + 1 if "/" in s else int(s)

def get_season_order(df):
    return sorted(df["season"].unique(), key=season_key)


# ─── INDIVIDUAL MODEL BUILDERS ───────────────────────────────────────────────

def build_xgb(early_stopping=True):
    base = dict(
        max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, gamma=1.0,
        reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    if early_stopping:
        base["n_estimators"] = 500
        base["early_stopping_rounds"] = 40
    else:
        base["n_estimators"] = 100
    return XGBClassifier(**base)


def build_rf():
    return RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=10,
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1,
    )


def build_lr():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.5, max_iter=1000, solver="lbfgs", random_state=42)),
    ])


# ─── CROSS VALIDATION ────────────────────────────────────────────────────────

def cv_single_model(name, model_fn, df, n_test_seasons=2, xgb=False):
    seasons      = get_season_order(df)
    test_seasons = [s for s in seasons[-n_test_seasons:] if s != "2026"]
    results      = []

    for test_season in test_seasons:
        idx           = seasons.index(test_season)
        train_seasons = seasons[:idx]
        train_df = df[df["season"].isin(train_seasons)]
        test_df  = df[df["season"] == test_season]
        if len(train_df) < 50 or len(test_df) < 30:
            continue

        X_tr = train_df[FEATURE_COLS].fillna(0.5)
        y_tr = train_df[TARGET_COL].astype(int)
        X_te = test_df[FEATURE_COLS].fillna(0.5)
        y_te = test_df[TARGET_COL].astype(int)

        m = model_fn()
        if xgb:
            m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        else:
            m.fit(X_tr, y_tr)

        y_prob = m.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        results.append({
            "test_season": test_season,
            "accuracy":    round(accuracy_score(y_te, y_pred), 4),
            "auc":         round(roc_auc_score(y_te, y_prob), 4),
            "brier":       round(brier_score_loss(y_te, y_prob), 4),
            "log_loss":    round(log_loss(y_te, y_prob), 4),
        })

    if results:
        log.info(
            f"  {name:20s} CV → "
            f"Acc: {np.mean([r['accuracy'] for r in results]):.3f}  "
            f"AUC: {np.mean([r['auc'] for r in results]):.3f}  "
            f"Brier: {np.mean([r['brier'] for r in results]):.3f}"
        )
    return results


def run_all_cv(df):
    log.info("── Time-Based CV for all models ────────────────────────────────")
    cv = {}
    cv["xgb"] = cv_single_model("XGBoost",            build_xgb, df, xgb=True)
    cv["rf"]  = cv_single_model("Random Forest",       build_rf,  df)
    cv["lr"]  = cv_single_model("Logistic Regression", build_lr,  df)

    # Ensemble CV
    seasons      = get_season_order(df)
    test_seasons = [s for s in seasons[-2:] if s != "2026"]
    ens_results  = []

    for test_season in test_seasons:
        idx           = seasons.index(test_season)
        train_seasons = seasons[:idx]
        train_df = df[df["season"].isin(train_seasons)]
        test_df  = df[df["season"] == test_season]
        if len(train_df) < 50 or len(test_df) < 30:
            continue

        X_tr = train_df[FEATURE_COLS].fillna(0.5)
        y_tr = train_df[TARGET_COL].astype(int)
        X_te = test_df[FEATURE_COLS].fillna(0.5)
        y_te = test_df[TARGET_COL].astype(int)

        probs_all = []
        for fn, is_xgb in [(build_xgb, True), (build_rf, False), (build_lr, False)]:
            m = fn()
            if is_xgb:
                m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            else:
                m.fit(X_tr, y_tr)
            probs_all.append(m.predict_proba(X_te)[:, 1])

        y_prob = np.mean(probs_all, axis=0)
        y_pred = (y_prob >= 0.5).astype(int)
        ens_results.append({
            "test_season": test_season,
            "accuracy":    round(accuracy_score(y_te, y_pred), 4),
            "auc":         round(roc_auc_score(y_te, y_prob), 4),
            "brier":       round(brier_score_loss(y_te, y_prob), 4),
            "log_loss":    round(log_loss(y_te, y_prob), 4),
        })

    if ens_results:
        log.info(
            f"  {'Ensemble':20s} CV → "
            f"Acc: {np.mean([r['accuracy'] for r in ens_results]):.3f}  "
            f"AUC: {np.mean([r['auc'] for r in ens_results]):.3f}  "
            f"Brier: {np.mean([r['brier'] for r in ens_results]):.3f}"
        )
    cv["ensemble"] = ens_results
    return cv


# ─── FINAL TRAINING ──────────────────────────────────────────────────────────

def train_all(df):
    train_df = df[df["season"] != "2026"].copy()
    log.info(f"Training on {len(train_df)} matches (excluding 2026).")

    X = train_df[FEATURE_COLS].fillna(0.5)
    y = train_df[TARGET_COL].astype(int)

    models = {}

    log.info("  Training XGBoost...")
    xgb_base = build_xgb(early_stopping=False)
    xgb_base.fit(X, y, verbose=False)
    xgb_cal = CalibratedClassifierCV(xgb_base, method="isotonic", cv=5)
    xgb_cal.fit(X, y)
    models["xgb"] = xgb_cal
    importances = dict(zip(FEATURE_COLS, xgb_base.feature_importances_))
    sorted_imp  = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    log.info("    Top 8 features:")
    for feat, imp in sorted_imp[:8]:
        log.info(f"      {feat:30s}: {imp:.4f}")

    log.info("  Training Random Forest...")
    rf_base = build_rf()
    rf_base.fit(X, y)
    rf_cal = CalibratedClassifierCV(rf_base, method="isotonic", cv=5)
    rf_cal.fit(X, y)
    models["rf"] = rf_cal
    log.info("    Random Forest done.")

    log.info("  Training Logistic Regression...")
    lr_model = build_lr()
    lr_model.fit(X, y)
    lr_cal = CalibratedClassifierCV(lr_model, method="sigmoid", cv=5)
    lr_cal.fit(X, y)
    models["lr"] = lr_cal
    log.info("    Logistic Regression done.")

    log.info("  Building ensemble...")
    ensemble = SoftEnsemble(
        estimators=[("xgb", xgb_cal), ("rf", rf_cal), ("lr", lr_cal)],
        weights=[0.45, 0.30, 0.25],
    )
    models["ensemble"] = ensemble
    log.info("    Ensemble ready (XGB 45% / RF 30% / LR 25%).")

    return models, sorted_imp


# ─── SAVE ────────────────────────────────────────────────────────────────────

def save_all(models, cv_results, sorted_imp):
    MODEL_DIR.mkdir(exist_ok=True)

    name_map = {
        "xgb":      "model_xgb.pkl",
        "rf":       "model_rf.pkl",
        "lr":       "model_lr.pkl",
        "ensemble": "model_ensemble.pkl",
    }
    for key, filename in name_map.items():
        path = MODEL_DIR / filename
        with open(path, "wb") as f:
            pickle.dump(models[key], f)
        log.info(f"  Saved → {path}")

    with open(MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump(models["ensemble"], f)

    meta = {
        "feature_cols":        FEATURE_COLS,
        "target_col":          TARGET_COL,
        "cv_results":          cv_results,
        "feature_importances": [{"feature": f, "importance": float(i)} for f, i in sorted_imp],
        "model_type":          "SoftEnsemble(XGBoost 45% + RandomForest 30% + LogisticRegression 25%)",
        "trained_on":          "all seasons except 2026",
        "models":              list(name_map.values()),
    }
    with open(MODEL_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Metadata saved → {MODEL_DIR / 'model_meta.json'}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train IPL ensemble model")
    parser.add_argument("--db", default=str(DB_PATH))
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        df = load_features(conn)
    finally:
        conn.close()

    df = add_derived_features(df)

    log.info("\n── Step 1: Cross Validation ─────────────────────────────────────")
    cv_results = run_all_cv(df)

    log.info("\n── Step 2: Final Training ───────────────────────────────────────")
    models, sorted_imp = train_all(df)

    log.info("\n── Step 3: Saving ───────────────────────────────────────────────")
    save_all(models, cv_results, sorted_imp)

    log.info("\nEnsemble training complete. models/model.pkl → SoftEnsemble.")


if __name__ == "__main__":
    main()