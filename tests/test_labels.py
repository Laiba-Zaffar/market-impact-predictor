import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.features.labels import compute_label_from_series

# A small synthetic trading-day series: Mon-Fri, one gap for a holiday.
DATES = ["2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05", "2026-03-06", "2026-03-09", "2026-03-10"]
CLOSES = [100.0, 102.0, 101.0, 105.0, 103.0, 110.0, 108.0]


def test_published_during_market_hours_uses_same_day_close():
    # 2026-03-03 14:00 UTC is well before the ~20:00 UTC close cutoff.
    label = compute_label_from_series(DATES, CLOSES, "20260303T140000", horizon_days=1)
    assert label["reference_date"] == "2026-03-03"
    assert label["target_date"] == "2026-03-04"


def test_published_after_market_close_rolls_to_next_trading_day():
    # 2026-03-03 22:00 UTC is after the close cutoff - can't react same day.
    label = compute_label_from_series(DATES, CLOSES, "20260303T220000", horizon_days=1)
    assert label["reference_date"] == "2026-03-04"
    assert label["target_date"] == "2026-03-05"


def test_published_on_non_trading_day_rolls_forward():
    # 2026-03-07/08 is a weekend, not in the series at all.
    label = compute_label_from_series(DATES, CLOSES, "20260307T120000", horizon_days=1)
    assert label["reference_date"] == "2026-03-09"


def test_direction_and_magnitude_computed_correctly():
    label = compute_label_from_series(DATES, CLOSES, "20260302T140000", horizon_days=2)
    # ref = 2026-03-02 (100.0), target = 2026-03-04 (101.0)
    assert label["reference_close"] == 100.0
    assert label["target_close"] == 101.0
    assert label["direction"] == 1
    assert abs(label["return_pct"] - 0.01) < 1e-9


def test_direction_is_zero_when_price_drops():
    # ref = 2026-03-05 (105.0), target = 2026-03-06 (103.0) - a drop
    label = compute_label_from_series(DATES, CLOSES, "20260305T140000", horizon_days=1)
    assert label["direction"] == 0
    assert label["return_pct"] < 0


def test_returns_none_when_not_enough_future_trading_days():
    label = compute_label_from_series(DATES, CLOSES, "20260310T140000", horizon_days=1)
    assert label is None


def test_article_before_price_history_rolls_to_first_available_day():
    # An article dated before the series starts isn't a failure case -
    # "first trading day strictly after article_date" still resolves,
    # landing on the very first date in the series.
    label = compute_label_from_series(DATES, CLOSES, "20260101T140000", horizon_days=1)
    assert label is not None
    assert label["reference_date"] == "2026-03-02"
