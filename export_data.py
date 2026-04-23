import sqlite3
import pandas as pd
import json

conn = sqlite3.connect("ipl.db")

df = pd.read_sql("SELECT * FROM simulation_results", conn)

# convert to dict format
probs = dict(zip(df["team"], df["win_prob"]))

with open("probabilities.json", "w") as f:
    json.dump(probs, f, indent=2)

print("✅ probabilities.json fixed")

conn.close()