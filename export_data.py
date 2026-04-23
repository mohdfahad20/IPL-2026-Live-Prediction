import sqlite3
import pandas as pd
import json
from pathlib import Path

conn = sqlite3.connect("ipl.db")

# ──────────────
# 1. Probabilities (from DB)
# ──────────────
try:
    probs = pd.read_sql("SELECT * FROM simulation_results", conn)
    probs.to_json("probabilities.json", orient="records")
    print("✅ probabilities.json created")
except Exception as e:
    print("⚠️ probabilities export failed:", e)

conn.close()

# ──────────────
# 2. Standings (already JSON)
# ──────────────
if Path("standings.json").exists():
    print("✅ standings.json already exists")
else:
    print("⚠️ standings.json missing — run scrape_standings.py")