import sqlite3
import logging
from contextlib import contextmanager
from api.core.config import DB_PATH

log = logging.getLogger(__name__)


def get_db() -> sqlite3.Connection:
    """Return a new SQLite connection."""
    conn = sqlite3.connect(DB_PATH)
    return conn


@contextmanager
def db_connection():
    """
    Context manager for DB connection.
    Auto-closes connection safely.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()