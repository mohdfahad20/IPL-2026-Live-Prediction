# 🏏 IPL 2026 Live Prediction System

A production-grade machine learning system that predicts IPL 2026 match outcomes and tournament win probabilities — updated automatically after every match, including playoffs.

[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat&logo=fastapi)](https://ipl-predictor-api.onrender.com/docs)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=flat&logo=streamlit)](https://streamlit.io)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?style=flat&logo=github-actions)](https://github.com/mohdfahad20/IPL-2026-Live-Prediction/actions)

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [ML Pipeline](#ml-pipeline)
- [API Reference](#api-reference)
- [CI/CD Pipeline](#cicd-pipeline)
- [Project Structure](#project-structure)
- [Local Setup](#local-setup)
- [Deployment](#deployment)
- [Data Sources](#data-sources)
- [Edge Cases Handled](#edge-cases-handled)

---

## Features

| Feature | Description |
|---|---|
| **Match Prediction** | Ensemble model (XGBoost + Random Forest + Logistic Regression) gives per-model and combined win probabilities for any matchup |
| **Tournament Simulation** | 10,000 Monte Carlo runs after every match to estimate each team's championship probability — playoff-aware from Q1 through the Final |
| **Playoff Bracket Simulation** | Completed playoff matches (Q1, Eliminator, Q2, Final) use real results; unplayed stages are simulated |
| **Probability Trends** | Track how each team's win probability evolved across the full season |
| **Live Points Table** | Scraped from Cricbuzz with real NRR, falling back to DB computation |
| **Automated Pipeline** | Nightly GitHub Actions job scrapes, retrains, simulates, and deploys without manual intervention |
| **Data Integrity Guards** | Three-layer validation prevents silent corruption from ever reaching the model or frontend |

---

## Architecture

```
GitHub Actions (nightly cron)
│
├── scrape → fix stages → validate → features → train → simulate → log
│
└── artifacts.zip (ipl.db + models/ + standings.json)
          │
          ▼ GitHub Release (latest-data tag)
          │           ← automatic rollback point if pipeline fails
          ▼
    Render cold start downloads artifacts

FastAPI Backend (Render)
├── POST /api/predict
├── POST /api/toss
├── GET  /api/probabilities
├── GET  /api/probabilities/history
├── GET  /api/standings
├── GET  /api/recent-matches
├── GET  /api/venues
└── GET  /health
          │
          │ HTTP JSON
          ▼
Streamlit Frontend
├── Win Probabilities tab
├── Points Table tab
├── Probability Trends tab
├── Match Predictor tab
└── Live Toss Predictor tab
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Models | XGBoost, Random Forest, Logistic Regression (scikit-learn) |
| Simulation | Monte Carlo (10,000 runs, NumPy) — playoff-aware bracket |
| Backend | FastAPI, Uvicorn, Python 3.11 |
| Database | SQLite (ipl.db) |
| Data Scraping | cricdata (CricinfoClient), BeautifulSoup, Requests |
| Feature Engineering | Pandas, NumPy |
| Frontend | Streamlit, Plotly |
| CI/CD | GitHub Actions |
| Deployment | Render (backend), Streamlit Cloud (frontend) |
| Artifact Storage | GitHub Releases (versioned, automatic rollback) |

---

## ML Pipeline

### 1. Data Ingestion

Historical IPL data (2008–2025) is loaded from a Kaggle ball-by-ball dataset into SQLite. 2026 match results are scraped nightly from Cricinfo via the `cricdata` library. Each match is assigned the correct stage label (`League`, `Qualifier 1`, `Eliminator`, `Qualifier 2`, `Final`) based on its match number.

### 2. Feature Engineering (`features/features.py`)

Computes per-match features before the match is played (no data leakage — uses only `df.iloc[:i]`):

| Feature | Description |
|---|---|
| `team1_form_last5` | Win rate in last 5 matches this season |
| `team1_season_winrate` | Season win rate so far |
| `team1_season_points` | Current points in standings |
| `h2h_winrate_team1` | Historical head-to-head win rate vs opponent |
| `team1_venue_winrate` | Win rate at this specific venue |
| `toss_won_team1` | Whether team1 won the toss |
| `toss_bat_team1` | Whether team1 elected to bat |
| `form_diff` | team1_form − team2_form |
| `winrate_diff` | Season win rate difference |
| `points_diff` | Points table difference |
| `venue_diff` | Venue win rate difference |
| `opponent_strength` | Opponent's season win rate |
| `is_playoff` | Playoff match flag |

### 3. Model Training (`model/train.py`)

Trains three base models plus a soft voting ensemble:

```
XGBoost           → strong on form + venue features
Random Forest     → good on head-to-head patterns
Logistic Reg      → calibrated probability baseline
Soft Ensemble     → weighted average (XGB 45% / RF 30% / LR 25%)
```

Models are saved as `.pkl` files. Feature column order is saved in `model_meta.json` for consistent inference. `SoftEnsemble` is defined at module level in `train.py` and imported at the top of `model_loader.py` to ensure pickle deserialization works regardless of entry point.

### 4. Monte Carlo Simulation (`simulate/simulate.py`)

Playoff-aware simulation — handles the full tournament lifecycle:

**Group stage (matches 1–70):**
1. Separate completed matches into `league_completed` and `playoff_completed`
2. Build remaining league fixtures — each team capped at exactly 14 league matches
3. Run `predict_match()` to get win probabilities for each remaining fixture
4. Simulate 10,000 seasons, sampling outcomes per fixture
5. Rank all 10 teams by points then win-rate (NRR proxy)

**Playoff bracket (matches 71–74):**
```
Q1:    #1 vs #2  → winner → Final directly
EL:    #3 vs #4  → loser eliminated
Q2:    Q1-loser vs EL-winner → Final
Final: Q1-winner vs Q2-winner → Champion
```

For each bracket stage, `play_or_lookup()` checks if the match has already been played:
- **Real result exists** → uses actual winner from DB (deterministic across all 10,000 runs)
- **Not yet played** → simulates using model probability

This means once Q1 is played, all 10,000 simulations reflect the real Q1 outcome automatically.

---

## API Reference

### `POST /api/predict`

Match win prediction (pre-toss or post-toss).

**Request**
```json
{
  "team1": "Punjab Kings",
  "team2": "Rajasthan Royals",
  "season": "2026",
  "venue": "Narendra Modi Stadium",
  "toss_winner": "Punjab Kings",
  "toss_decision": "bat"
}
```

**Response**
```json
{
  "team1": "Punjab Kings",
  "team2": "Rajasthan Royals",
  "p_team1_wins": 0.5821,
  "p_team2_wins": 0.4179,
  "model_probs": {
    "XGBoost": 0.61,
    "Random Forest": 0.57,
    "Logistic Regression": 0.54,
    "Ensemble": 0.5821
  }
}
```

### `POST /api/toss`

Pre vs post-toss probability comparison.

### `GET /api/probabilities`

Latest Monte Carlo simulation results. `matches_played` reflects league + completed playoffs; `matches_remaining` reflects remaining league + unplayed playoff stages.

### `GET /api/probabilities/history`

All historical simulation runs — used for probability trend charts.

### `GET /api/standings`

Current IPL 2026 points table with NRR. Reads from `standings.json` (Cricbuzz) as primary, falls back to DB computation.

### `GET /api/recent-matches`

Last 8 completed matches.

### `GET /api/venues`

All historical venue-team combinations for dropdown population.

> Interactive API docs available at `/docs` on the Render backend.

---

## CI/CD Pipeline

### Nightly Job (12 AM IST daily)

```
Download artifacts.zip from GitHub Release
  └─ Save previous 2026 match count for regression check
          ↓
Download Kaggle CSV from Google Drive
          ↓
Initialize DB if missing
          ↓
DELETE 2026 matches (prevents Kaggle duplicates)
          ↓
Scrape fresh 2026 results from Cricinfo
          ↓
Fix match stages
  └─ League matches → 'League'
  └─ Match 71 → 'Qualifier 1', 72 → 'Eliminator', 73 → 'Qualifier 2', 74 → 'Final'
          ↓
GUARD 1: Abort if < 48 matches scraped
GUARD 2: Abort if current count < previous count (regression)
GUARD 3: Abort if bad stages / duplicate IDs / null teams / wrong playoff labels
          ↓
Scrape standings from Cricbuzz
          ↓
Feature engineering → Model training → Simulation → Logging
          ↓
Package ipl.db + models/ + standings.json → Upload to GitHub Release
  └─ Old release preserved until this step succeeds — automatic rollback point
          ↓
Trigger Render redeploy via deploy hook
          ↓
Commit CSV logs to repo
```

### Toss Job

Runs during match windows to log live toss results:

- **Weekdays:** 6:30–8:30 PM IST
- **Weekends:** 2:30–8:30 PM IST

### Data Integrity Guards

Three validation layers run before features/model/simulation:

| Guard | What it checks | Action on failure |
|---|---|---|
| Scrape count floor | `>= 48` matches in DB | Abort — old artifacts preserved |
| Count regression | Today's count `>=` yesterday's count | Abort — old artifacts preserved |
| Data quality | No bad stages, no duplicates, no null teams, no misassigned playoff labels | Abort — old artifacts preserved |

GitHub Release is the automatic rollback point — it is only overwritten at the very end of a fully successful run.

---

## Project Structure

```
IPL-2026-Live-Prediction/
├── api/
│   ├── main.py                    # FastAPI app, registers all routers
│   ├── startup.py                 # Downloads artifacts on cold start (3-retry)
│   ├── core/
│   │   ├── config.py              # Paths + env vars
│   │   ├── database.py            # DB connection context manager
│   │   └── model_loader.py        # Loads + caches pkl models (SoftEnsemble fix)
│   ├── routers/
│   │   ├── predict.py             # POST /api/predict
│   │   ├── toss.py                # POST /api/toss
│   │   ├── simulation.py          # GET  /api/probabilities
│   │   └── standings.py           # GET  /api/standings + /recent-matches + /venues
│   └── services/
│       ├── prediction_service.py
│       ├── toss_service.py
│       ├── simulation_service.py
│       └── standings_service.py
├── data/
│   └── load_kaggle_data.py        # Loads historical Kaggle CSV into DB
├── features/
│   └── features.py                # Computes 23 match features (no leakage)
├── model/
│   ├── train.py                   # Trains ensemble, saves pkl + model_meta.json
│   └── predict.py                 # Prediction logic used by API services
├── simulate/
│   └── simulate.py                # Playoff-aware Monte Carlo simulation
├── scraper/
│   └── scrapper_data.py           # Scrapes live 2026 results; maps match number → stage
├── dashboard/
│   ├── app.py                     # Streamlit frontend (API mode — Streamlit Cloud)
│   └── app_local.py               # Streamlit frontend (local DB mode)
├── .github/
│   └── workflows/
│       └── update_ipl_2026.yml    # Nightly pipeline + toss cron + 3 data guards
├── scrape_standings.py            # Scrapes points table from Cricbuzz
├── toss_scraper.py                # Scrapes live toss results
├── logger.py                      # Logs simulation results to DB
├── render.yaml                    # Render deployment config
├── requirements_pipeline.txt
├── requirements_backend.txt
└── requirements_frontend.txt
```

---

## Local Setup

```bash
# Clone
git clone https://github.com/mohdfahad20/IPL-2026-Live-Prediction
cd IPL-2026-Live-Prediction

# Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements_pipeline.txt
pip install -r requirements_backend.txt
pip install -r requirements_frontend.txt

# Build DB from Kaggle data
python data/load_kaggle_data.py --csv data/IPL.csv --db ipl.db

# Full pipeline
python scraper/scrapper_data.py --db ipl.db
python scrape_standings.py
python features/features.py --db ipl.db
python model/train.py --db ipl.db
python -m simulate.simulate --db ipl.db --n 10000
python logger.py --db ipl.db

# Start API
uvicorn api.main:app --reload --port 8000
# Docs: http://localhost:8000/docs

# Start local dashboard
streamlit run dashboard/app_local.py
```

### DB Diagnostics

```bash
# Check 2026 stage breakdown
python check_db_stage.py

# Check 2026 match state
python -c "
import sqlite3; conn = sqlite3.connect('ipl.db')
rows = conn.execute(\"SELECT match_id,date,team1,team2,winner,stage,event_match_no FROM matches WHERE season='2026' ORDER BY date\").fetchall()
print(f'{len(rows)} matches'); [print(r) for r in rows]
conn.close()
"

# Check latest simulation run
python -c "
import sqlite3; conn = sqlite3.connect('ipl.db')
rows = conn.execute('SELECT run_id, run_at, matches_played, matches_remaining FROM simulation_results ORDER BY run_at DESC LIMIT 5').fetchall()
[print(r) for r in rows]; conn.close()
"
```

---

## Deployment

### Backend — Render

1. Connect the GitHub repo to Render
2. Render auto-detects `render.yaml`
3. Set env var: `ARTIFACTS_URL` → GitHub Release download URL
4. Add `RENDER_DEPLOY_HOOK` as a GitHub Actions secret

### Frontend — Streamlit Cloud

1. Connect the GitHub repo to Streamlit Cloud
2. Set main file: `dashboard/app.py`
3. Add secret: `API_BASE_URL` → Render backend URL

---

## Data Sources

| Source | Used For |
|---|---|
| Kaggle IPL ball-by-ball dataset (2008–2025) | Historical model training data |
| Cricinfo via `cricdata` library | Live 2026 match results (match number → stage mapped automatically) |
| Cricbuzz (HTML scrape + API fallback) | Live points table with accurate NRR |

---

## Edge Cases Handled

| Case | How it's handled |
|---|---|
| Super over matches | Multi-fallback winner extraction (winnerTeamId → superoverWinnerTeamId → statusText) |
| No result / abandoned | 1 point each; excluded from model training target |
| Kaggle + scraper overlap | 2026 data deleted before each scrape to prevent duplicates |
| Wrong stage labels from cricdata | `get_ipl_stage()` maps match number → correct stage on every scrape |
| Playoff teams exceeding 14 matches | Simulation splits league and playoff; 14-match cap applies to league only |
| Already-played playoff matches | `play_or_lookup()` uses real results; unplayed stages are simulated |
| Render cold start | `startup.py` downloads artifacts with 3-retry + exponential backoff |
| Backend sleeping | Streamlit auto-retries with 5s wait and `st.rerun()` |
| Partial or corrupt scrape | Three pipeline guards abort before model sees bad data; last good release preserved |
| Silent count regression | Pipeline aborts if today's match count drops below yesterday's |

---

*Built for learning · IPL 2026 · Ensemble ML · Monte Carlo · FastAPI · Streamlit*