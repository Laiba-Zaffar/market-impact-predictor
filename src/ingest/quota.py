import json
from datetime import datetime, timezone
from pathlib import Path

STATE_PATH = Path(__file__).resolve().parent.parent.parent / ".quota_state.json"
DAILY_LIMIT = 25
SAFETY_MARGIN = 2  # leave a couple of calls unused for ad-hoc debugging


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load() -> dict:
    if not STATE_PATH.exists():
        return {"date": _today(), "calls_used": 0}
    state = json.loads(STATE_PATH.read_text())
    if state.get("date") != _today():
        return {"date": _today(), "calls_used": 0}
    return state


def _save(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state))


def calls_remaining() -> int:
    state = _load()
    return max(0, DAILY_LIMIT - SAFETY_MARGIN - state["calls_used"])


def record_call() -> None:
    state = _load()
    state["calls_used"] += 1
    _save(state)


def is_rate_limit_response(payload: dict) -> bool:
    # Alpha Vantage signals a rate limit via a 200 OK with an "Information"
    # (or sometimes "Note") field, not an HTTP error - so we have to look
    # inside the body rather than trusting the status code.
    note = str(payload.get("Information", "")) + str(payload.get("Note", ""))
    return "rate limit" in note.lower() or "25 requests" in note.lower()
