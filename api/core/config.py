import os
from pathlib import Path

# Project root = one level above api/
ROOT_DIR   = Path(__file__).resolve().parent.parent.parent
DB_PATH    = ROOT_DIR / "ipl.db"
MODELS_DIR = ROOT_DIR / "models"

ARTIFACTS_URL = os.environ.get("ARTIFACTS_URL")
SEASON        = "2026"
RUNS_PER_WICKET = 8