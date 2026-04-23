import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "test.db"

print("Using DB:", DB_PATH)

conn = sqlite3.connect(DB_PATH)

# Show tables
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
print("\nTables:")
print(tables)

# Show matches schema
schema = pd.read_sql("PRAGMA table_info(matches);", conn)
print("\nMatches Table Schema:")
print(schema)

# Show sample data
df = pd.read_sql("SELECT * FROM matches LIMIT 5;", conn)
print("\nSample Data:")
print(df)

df = pd.read_sql("SELECT * FROM matches WHERE season = '2026';", conn)
print(df.tail(10))

conn.close()