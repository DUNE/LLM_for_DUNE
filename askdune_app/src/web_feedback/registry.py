
import secrets
import threading
import time

# Keep response records in memory for 24 hours.
RESPONSE_TTL_SECONDS = 24 * 60 * 60

_responses = {}
_lock = threading.Lock()


def _cleanup_expired():
    """Remove expired records. Caller must hold _lock."""
    now = time.time()

    expired = [
        response_id
        for response_id, record in _responses.items()
        if now - record["created_at"] > RESPONSE_TTL_SECONDS
    ]

    for response_id in expired:
        del _responses[response_id]


def create_response(user_id, question, references=None):
    """Register a new response without saving it to disk."""
    response_id = secrets.token_urlsafe(24)

    with _lock:
        _cleanup_expired()

        _responses[response_id] = {
            "user_id": user_id,
            "question": question,
            "answer": None,
            "references": references or [],
            "created_at": time.time(),
            "complete": False,
        }

    return response_id


def complete_response(response_id, answer):
    """Attach the completed answer to its response record."""
    with _lock:
        if response_id in _responses:
            _responses[response_id]["answer"] = answer
            _responses[response_id]["complete"] = True


def get_response(response_id, user_id):
    """Return a completed response only to its original user."""
    with _lock:
        _cleanup_expired()

        record = _responses.get(response_id)

        if not record:
            return None

        if record["user_id"] != user_id:
            return None

        if not record["complete"]:
            return None

        return record.copy()

def get_response_by_token(response_id):
    """Retrieve a completed response using its unpredictable response ID."""
    with _lock:
        _cleanup_expired()

        record = _responses.get(response_id)

        if not record or not record["complete"]:
            return None

        return record.copy()
