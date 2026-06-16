import sqlite3
import threading
from pathlib import Path
from typing import List, Tuple

DB_PATH = Path(__file__).resolve().with_name("context.db")
_db_lock = threading.Lock()


def _conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with _db_lock:
        c = _conn()
        cur = c.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            text TEXT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            session_id TEXT PRIMARY KEY,
            summary TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        c.commit()
        c.close()

def save_message(session_id: str, role: str, text: str):
    with _db_lock:
        c = _conn()
        c.execute("INSERT INTO messages (session_id, role, text) VALUES (?, ?, ?)", (session_id, role, text))
        c.commit()
        c.close()

def get_recent(session_id: str, limit: int = 20) -> List[Tuple[str, str]]:
    c = _conn()
    cur = c.cursor()
    cur.execute("SELECT role, text FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?", (session_id, limit))
    rows = cur.fetchall()
    c.close()
    return list(reversed(rows))

def save_summary(session_id: str, summary: str):
    with _db_lock:
        c = _conn()
        c.execute("INSERT OR REPLACE INTO summaries (session_id, summary) VALUES (?, ?)", (session_id, summary))
        c.commit()
        c.close()

def get_summary(session_id: str) -> str:
    c = _conn()
    cur = c.cursor()
    cur.execute("SELECT summary FROM summaries WHERE session_id=?", (session_id,))
    row = cur.fetchone()
    c.close()
    return row[0] if row else ""


def clear_session(session_id: str):
    with _db_lock:
        c = _conn()
        c.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        c.execute("DELETE FROM summaries WHERE session_id=?", (session_id,))
        c.commit()
        c.close()

# Initialize DB on import
init_db()
