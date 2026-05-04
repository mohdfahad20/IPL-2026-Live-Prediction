# 🏏 IPL 2026 Live Prediction System

A production-grade machine learning system that predicts IPL 2026 match outcomes and tournament win probabilities — updated automatically after every match.

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
| **Toss Impact Analysis** | Compare pre-toss vs post-toss win probabilities showing exactly how much the toss shifted the odds |
| **Tournament Simulation** | 10,000 Monte Carlo runs after every match to estimate each team's championship probability |
| **Probability Trends** | Track how each team's win probability has evolved across the season |
| **Live Points Table** | Scraped from Cricbuzz with real NRR, falling back to DB computation |
| **Automated Pipeline** | Nightly GitHub Actions job scrapes, retrains, simulates, and deploys without manual intervention |

---

## Architecture

```
GitHub Actions (nightly cron)
│
├── scrape → features → train → simulate → log
│
└── artifacts.zip (ipl.db + models/)
          │
          ▼ GitHub Release (latest-data)
          │
          ▼ Render cold start downloads artifacts

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
| Simulation | Monte Carlo (10,000 runs, NumPy) |
| Backend | FastAPI, Uvicorn, Python 3.11 |
| Database | SQLite (ipl.db) |
| Data Scraping | cricdata (CricinfoClient), BeautifulSoup, Requests |
| Feature Engineering | Pandas, NumPy |
| Frontend | Streamlit, Plotly |
| CI/CD | GitHub Actions |
| Deployment | Render (backend), Streamlit Cloud (frontend) |
| Artifact Storage | GitHub Releases |

---

## ML Pipeline

### 1. Data Ingestion

Historical IPL data (2008–2025) is loaded from a Kaggle ball-by-ball dataset into SQLite. 2026 match results are scraped nightly from Cricinfo via the `cricdata` library.

### 2. Feature Engineering (`features/features.py`)

Computes per-match features before the match is played:

| Feature | Description |
|---|---|
| `team1_form_last5` | Win rate in last 5 matches |
| `team1_season_winrate` | Season win rate so far |
| `team1_season_points` | Current points in standings |
| `h2h_winrate_team1` | Head-to-head win rate vs opponent |
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
Soft Ensemble     → weighted average of all three
```

Models are saved as `.pkl` files. Feature column order is saved in `model_meta.json` to ensure consistent inference.

### 4. Monte Carlo Simulation (`simulate/simulate.py`)

For each remaining match in the season:

1. Run `predict_match()` to get win probabilities
2. Simulate outcome using a weighted random draw
3. Update standings and repeat for all remaining matches
4. Count playoff + championship appearances across 10,000 runs
5. Store results in the `simulation_results` table with a timestamp

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

---

### `POST /api/toss`

Pre vs post-toss probability comparison.

**Response**
```json
{
  "pre_toss":  { "p_team1_wins": 0.54, "p_team2_wins": 0.46 },
  "post_toss": { "p_team1_wins": 0.58, "p_team2_wins": 0.42 },
  "toss_shift": 0.04,
  "impact": "Moderate",
  "beneficiary": "Punjab Kings"
}
```

---

### `GET /api/probabilities`

Latest Monte Carlo simulation results.

### `GET /api/probabilities/history`

All historical simulation runs — used for probability trend charts.

### `GET /api/standings`

Current IPL 2026 points table with NRR.

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
          ↓
Download Kaggle CSV from Google Drive
          ↓
Initialize DB if missing
          ↓
DELETE 2026 matches (prevents Kaggle duplicates)
          ↓
Scrape fresh 2026 results from Cricinfo
          ↓
GUARD: abort if < 35 matches scraped
          ↓
Scrape standings from Cricbuzz
          ↓
Feature engineering → Model training → Simulation → Logging
          ↓
Package ipl.db + models/ → Upload to GitHub Release
          ↓
Trigger Render redeploy via deploy hook
          ↓
Commit CSV logs to repo
```

### Toss Job

Runs during match windows to log live toss results:

- **Weekdays:** 6:30–8:30 PM IST
- **Weekends:** 1:30–3:30 PM IST and 6:30–8:30 PM IST

### Data Integrity Guards

- Pipeline aborts if scrape returns fewer than 35 matches
- Old `artifacts.zip` is preserved on GitHub Release if the pipeline fails
- Render never receives a broken deploy

---

## Project Structure

```
IPL-2026-Live-Prediction/
├── api/
│   ├── main.py                    # FastAPI app, registers all routers
│   ├── startup.py                 # Downloads artifacts on cold start
│   ├── core/
│   │   ├── config.py              # Paths + env vars
│   │   ├── database.py            # DB connection context manager
│   │   └── model_loader.py        # Loads + caches pkl models
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
│   └── load_kaggle_data.py        # Loads historical Kaggle ball-by-ball CSV into DB
├── features/
│   └── features.py                # Computes 20+ match features into DB
├── model/
│   ├── train.py                   # Trains ensemble, saves pkl + model_meta.json
│   └── predict.py                 # Prediction logic used by services
├── simulate/
│   └── simulate.py                # Monte Carlo tournament simulation
├── scraper/
│   └── scrapper_data.py           # Scrapes live 2026 match results
├── dashboard/
│   ├── app.py                     # Streamlit frontend (API mode)
│   └── app_local.py               # Streamlit frontend (local DB mode)
├── .github/
│   └── workflows/
│       └── update_ipl_2026.yml    # Nightly pipeline + toss cron
├── scrape_standings.py            # Scrapes points table from Cricbuzz
├── toss_scraper.py                # Scrapes live toss results
├── logger.py                      # Logs simulation results to DB
├── clean_dup.py                   # Deduplication utility
├── render.yaml                    # Render deployment config
├── requirements_pipeline.txt      # Pipeline dependencies
├── requirements_backend.txt       # FastAPI backend dependencies
└── requirements_frontend.txt      # Streamlit frontend dependencies
```

---

## Local Setup

```bash
# Clone the repository
git clone https://github.com/mohdfahad20/IPL-2026-Live-Prediction
cd IPL-2026-Live-Prediction

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements_pipeline.txt
pip install -r requirements_backend.txt
pip install -r requirements_frontend.txt

# Build the database from Kaggle data
python data/load_kaggle_data.py --csv data/IPL.csv --db ipl.db

# Run the full pipeline
python scraper/scrapper_data.py --db ipl.db
python features/features.py --db ipl.db
python model/train.py --db ipl.db
python -m simulate.simulate --db ipl.db --n 10000
python logger.py --db ipl.db

# Start the API backend
uvicorn api.main:app --reload --port 8000
# Interactive docs: http://localhost:8000/docs

# Start the local dashboard
streamlit run dashboard/app_local.py
```

---

## Deployment

### Backend — Render

1. Connect the GitHub repo to Render
2. Render auto-detects `render.yaml`
3. Set env var: `ARTIFACTS_URL` → GitHub Release download URL
4. Add `RENDER_DEPLOY_HOOK` as a secret in GitHub Actions

### Frontend — Streamlit Cloud

1. Connect the GitHub repo to Streamlit Cloud
2. Set main file: `dashboard/app.py`
3. Add secret: `API_BASE_URL` → Render backend URL

---

## Data Sources

| Source | Used For |
|---|---|
| Kaggle IPL ball-by-ball dataset (2008–2025) | Historical model training data |
| Cricinfo via `cricdata` library | Live 2026 match results |
| Cricbuzz (HTML scrape + API fallback) | Live points table with NRR |

---

## Edge Cases Handled

- **Super over matches** — robust winner extraction with multiple fallback strategies
- **No result / abandoned matches** — correctly awards 1 point each; excluded from model training
- **Kaggle + scraper overlap** — 2026 data deleted before each scrape to prevent duplicate rows
- **Render cold start** — `startup.py` downloads artifacts with 3-retry logic and progress logging
- **Backend sleeping** — Streamlit frontend auto-retries with 5s wait and `st.rerun()` on wake
- **Partial scrape** — pipeline aborts before model training if match count looks incomplete

---

*Built for learning · IPL 2026 · Ensemble ML · Monte Carlo · FastAPI · Streamlit*