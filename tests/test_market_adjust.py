import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.features.market_adjust import (
    MIN_BETA_OBS,
    compute_abnormal_fields,
    ols_slope,
    paired_daily_returns,
)


def make_series(n: int, daily_return: float, start: float = 100.0):
    """A synthetic trading-day series compounding at a fixed daily rate.
    Dates are sequential integers formatted as dates - not real trading
    days, but the label arithmetic only cares about ordering and shared
    keys, not about which weekday a date actually was."""
    dates = [f"2026-01-{i + 1:02d}" if i < 31 else f"2026-02-{i - 30:02d}" for i in range(n)]
    closes = [start * (1 + daily_return) ** i for i in range(n)]
    return dates, closes


def test_ols_slope_recovers_a_known_beta():
    market = [0.01, -0.02, 0.015, -0.005, 0.03]
    stock = [2.0 * m for m in market]  # beta of exactly 2, no noise
    assert ols_slope(market, stock) == 2.0


def test_ols_slope_survives_a_flat_market():
    # A market series that never moves can't explain anything. Falling
    # back to beta=1 beats dividing by zero.
    assert ols_slope([0.0] * 5, [0.01, 0.02, -0.01, 0.0, 0.005]) == 1.0


def test_paired_returns_skip_dates_the_market_did_not_trade():
    dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
    closes = [100.0, 110.0, 121.0]
    # The market is missing 01-02 entirely - pairing the stock's move
    # across that gap against an unrelated market day would corrupt beta.
    market = {"2026-01-01": 50.0, "2026-01-03": 55.0}
    stock_returns, market_returns = paired_daily_returns(dates, closes, market)
    assert stock_returns == []
    assert market_returns == []


def test_refuses_when_there_is_too_little_history_for_a_beta():
    n = MIN_BETA_OBS - 5
    dates, closes = make_series(n, 0.001)
    market = {d: 50.0 * (1.0005**i) for i, d in enumerate(dates)}
    # Reference near the end, so the trailing window is shorter than the
    # minimum - the honest answer is None, not a beta fitted to noise.
    result = compute_abnormal_fields(dates, closes, market, dates[-3], dates[-1])
    assert result is None


def test_a_pure_market_move_has_near_zero_abnormal_return():
    # A stock that is exactly the market (beta 1, no idiosyncratic move)
    # should have essentially no abnormal return - that's the whole point
    # of the adjustment.
    n = MIN_BETA_OBS + 40
    dates, closes = make_series(n, 0.002)
    market = {d: 50.0 * (1.002**i) for i, d in enumerate(dates)}

    result = compute_abnormal_fields(dates, closes, market, dates[-4], dates[-1])
    assert result is not None
    assert abs(result["beta"] - 1.0) < 1e-6
    assert abs(result["abnormal_return"]) < 1e-9


def test_beta_ignores_data_after_the_reference_date():
    """The point-in-time guarantee, tested directly.

    Two series identical up to the reference date and wildly different
    after it must produce the same beta - otherwise the label is built
    from information that didn't exist when the news broke, which is the
    exact lookahead error the time-based split exists to prevent.
    """
    n = MIN_BETA_OBS + 60
    dates, closes = make_series(n, 0.002)
    market = {d: 50.0 * (1.002**i) for i, d in enumerate(dates)}
    reference_index = MIN_BETA_OBS + 20

    # Same history, then a violent divergence strictly after the reference.
    diverged = list(closes)
    for i in range(reference_index + 1, n):
        diverged[i] = closes[i] * 5.0

    baseline = compute_abnormal_fields(dates, closes, market, dates[reference_index], dates[reference_index + 3])
    shocked = compute_abnormal_fields(dates, diverged, market, dates[reference_index], dates[reference_index + 3])

    assert baseline is not None and shocked is not None
    assert abs(baseline["beta"] - shocked["beta"]) < 1e-9
    assert abs(baseline["trailing_volatility"] - shocked["trailing_volatility"]) < 1e-9
    # The future move itself must still show up in the label - the test
    # is that it didn't leak into the *estimator*, not that it vanished.
    assert shocked["abnormal_return"] > baseline["abnormal_return"]


def test_standardization_scales_with_the_horizon():
    # Same abnormal move judged over a longer horizon is a smaller
    # surprise, because volatility accumulates with sqrt(time).
    n = MIN_BETA_OBS + 60
    dates, closes = make_series(n, 0.002)
    market = {d: 50.0 * (1.002**i) for i, d in enumerate(dates)}
    reference_index = MIN_BETA_OBS + 20
    closes[reference_index + 1] *= 1.05
    closes[reference_index + 5] *= 1.05

    short = compute_abnormal_fields(dates, closes, market, dates[reference_index], dates[reference_index + 1])
    long = compute_abnormal_fields(dates, closes, market, dates[reference_index], dates[reference_index + 5])
    assert short is not None and long is not None
    assert short["standardized_abnormal_return"] > long["standardized_abnormal_return"]
