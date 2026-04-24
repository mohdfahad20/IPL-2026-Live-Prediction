import pickle
import json
import logging
import sys
from pathlib import Path
from api.core.config import MODELS_DIR, ROOT_DIR

sys.path.insert(0, str(ROOT_DIR))

# CRITICAL — must be imported before any pickle.load() call
# pickle needs SoftEnsemble to be in scope to deserialize the model
from model.train import SoftEnsemble  # noqa: F401
sys.modules['__main__'].SoftEnsemble = SoftEnsemble

log = logging.getLogger(__name__)

_cache: dict = {}


def get_ensemble():
    if "ensemble" not in _cache:
        _load_all()
    return _cache["ensemble"]


def get_all_models():
    if "ensemble" not in _cache:
        _load_all()
    return (
        _cache["ensemble"],
        _cache["xgb"],
        _cache["rf"],
        _cache["lr"],
        _cache["feature_cols"],
    )


def _load_all():
    for key, filename in [
        ("ensemble", "model_ensemble.pkl"),
        ("xgb",      "model_xgb.pkl"),
        ("rf",       "model_rf.pkl"),
        ("lr",       "model_lr.pkl"),
    ]:
        path = MODELS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        with open(path, "rb") as f:
            _cache[key] = pickle.load(f)
        log.info(f"  Loaded {filename}")

    meta_path = MODELS_DIR / "model_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    _cache["feature_cols"] = meta["feature_cols"]
    log.info("All models loaded and cached.")