
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from config import WEB_FEEDBACK_DIR


FEEDBACK_DIR = Path(WEB_FEEDBACK_DIR)
PERMANENT_DB = FEEDBACK_DIR / "web_feedback.sqlite3"
TEMP_DB = FEEDBACK_DIR / "temp_feedback.sqlite3"


def connect_db(path):
    """Open a SQLite database with foreign keys enabled."""
    connection = sqlite3.connect(str(path), timeout=30)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_feedback_db():
    """Initialize permanent feedback storage."""
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

    with connect_db(PERMANENT_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                response_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                rating TEXT NOT NULL
                    CHECK (rating IN ('positive', 'negative')),
                reasons TEXT,
                comment TEXT,
                question TEXT,
                answer TEXT,
                references_json TEXT,
                model TEXT
            )
        """)
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS
                idx_feedback_response_id
            ON feedback(response_id)
        """)


def initialize_temp_db():
    """Initialize temporary pilot monitoring storage."""
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

    with connect_db(TEMP_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                response_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                user_email TEXT,
                question TEXT NOT NULL,
                answer TEXT,
                references_json TEXT,
                model TEXT
            )
        """)


def save_feedback(
    response_id,
    rating,
    reasons=None,
    comment=None,
    question=None,
    answer=None,
    references=None,
    model=None,
):
    """Save voluntary feedback, updating an existing rating if needed."""
    if rating not in ("positive", "negative"):
        raise ValueError("Invalid feedback rating")

    if not response_id:
        raise ValueError("Missing response ID")

    timestamp = datetime.now(timezone.utc).isoformat()

    with connect_db(PERMANENT_DB) as conn:
        conn.execute(
            """
            INSERT INTO feedback (
                response_id, timestamp, rating, reasons,
                comment, question, answer, references_json, model
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(response_id) DO UPDATE SET
                timestamp = excluded.timestamp,
                rating = excluded.rating,
                reasons = excluded.reasons,
                comment = excluded.comment,
                question = excluded.question,
                answer = excluded.answer,
                references_json = excluded.references_json,
                model = excluded.model
            """,
            (
                response_id,
                timestamp,
                rating,
                json.dumps(reasons or []),
                comment,
                question,
                answer,
                json.dumps(references or []),
                model,
            ),
        )


def save_temp_interaction(
    response_id,
    question,
    answer=None,
    references=None,
    user_email=None,
    model=None,
):
    """Save a question/answer interaction during the pilot."""
    timestamp = datetime.now(timezone.utc).isoformat()

    with connect_db(TEMP_DB) as conn:
        conn.execute(
            """
            INSERT INTO interactions (
                response_id, timestamp, user_email,
                question, answer, references_json, model
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(response_id) DO UPDATE SET
                answer = excluded.answer,
                references_json = excluded.references_json,
                model = excluded.model
            """,
            (
                response_id,
                timestamp,
                user_email,
                question,
                answer,
                json.dumps(references or []),
                model,
            ),
        )

