import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json

import pytest

from src.ingest import quota


@pytest.fixture
def state_path(tmp_path, monkeypatch):
    path = tmp_path / ".quota_state.json"
    monkeypatch.setattr(quota, "STATE_PATH", path)
    return path


def test_fresh_start_has_full_budget_minus_safety_margin(state_path):
    assert quota.calls_remaining() == quota.DAILY_LIMIT - quota.SAFETY_MARGIN


def test_record_call_decreases_remaining_budget(state_path):
    before = quota.calls_remaining()
    quota.record_call()
    assert quota.calls_remaining() == before - 1


def test_budget_never_goes_negative(state_path):
    for _ in range(quota.DAILY_LIMIT + 10):
        quota.record_call()
    assert quota.calls_remaining() == 0


def test_state_resets_on_a_new_utc_day(state_path):
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    state_path.write_text(json.dumps({"date": yesterday, "calls_used": quota.DAILY_LIMIT}))
    # Stale date from a prior day, at the cap - a naive read would report
    # 0 remaining, but the tracker should recognize the day rolled over.
    assert quota.calls_remaining() == quota.DAILY_LIMIT - quota.SAFETY_MARGIN


def test_state_persists_within_the_same_day(state_path):
    quota.record_call()
    quota.record_call()
    # A second, independent read (not just the same in-memory value) should
    # see both calls - this is what actually protects against overspending
    # across separate script runs, not just within one process.
    assert quota.calls_remaining() == quota.DAILY_LIMIT - quota.SAFETY_MARGIN - 2


def test_detects_rate_limit_via_information_field():
    payload = {"Information": "Thank you for using Alpha Vantage! Our standard API rate limit is 25 requests per day."}
    assert quota.is_rate_limit_response(payload) is True


def test_detects_rate_limit_via_note_field():
    payload = {"Note": "You have hit the rate limit."}
    assert quota.is_rate_limit_response(payload) is True


def test_normal_response_is_not_flagged_as_rate_limited():
    payload = {"items": "50", "feed": []}
    assert quota.is_rate_limit_response(payload) is False
