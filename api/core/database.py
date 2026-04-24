import sqlite3
import logging
from contextlib import contextmanager
from api.core.config import DB_PATH

log = logging.getLogger(__name__)


def get_db() -> sqlite3.Connection:
    """Return a new SQLite connection. Caller must close it."""
    return sqlite3.connect(DB_PATH)


@contextmanager
def db_connection():
    """
    Context manager — auto-closes connection.

    Usage:
        with db_connection() as conn:
            conn.execute(...)
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()