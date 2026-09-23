"""Market adjustment for forward-return labels.

A raw close-to-close return is mostly not about the news. Measured on
this project's own dataset, ~29% of raw forward-return variance is
market-wide, and on 43.7% of days more than 80% of the 21 tickers moved
the same direction - that's the index moving, not 21 separate news
reactions. Two consequences, both of which showed up in the baseline:

  1. A raw-return model gets a free lunch. P(up) on raw returns is
     0.5515 here purely because the sample period drifted upward, so
     "always predict up" scores 55.15% without predicting anything.
     That is exactly the majority-class collapse the baseline hit. On
     market-adjusted returns the base rate is 0.4891 - no free lunch,
     the model has to actually discriminate.
  2. Real signal gets buried. Correlation between Alpha Vantage's
     ticker sentiment and the label roughly doubles (+0.0123 -> +0.0292)
     once the market component is removed.

The adjustment is the standard event-study form: subtract the stock's
beta-weighted share of the market's move over the same window, then
divide by trailing volatility so a 1% move in a quiet utility and a 1%
move in a volatile growth name aren't treated as the same size event.

Everything here is point-in-time: beta and volatility are estimated
from a trailing window that ENDS at the reference close. Using the
full-sample beta instead would leak future information into the label -
a subtle lookahead error that inflates backtests exactly the way the
time-based split was built to prevent.
"""
import math
from statistics import fmean, stdev

# Trading days of history used to estimate beta and volatility. ~6 months:
# long enough that the estimate isn't noise, short enough to track a
# company whose risk profile actually changed.
BETA_WINDOW = 120

# Below this many paired observations, a beta estimate is noise dressed
# up as a number. Refuse to emit a label rather than emit a bad one -
# same principle as the pipeline's other minimum-data guards.
MIN_BETA_OBS = 60


def ols_slope(x: list[float], y: list[float]) -> float:
    """Least-squares slope of y on x - here, beta of the stock's daily
    returns against the market's. Written out rather than pulled from
    numpy/sklearn because it's four lines and keeps this module free of
    heavy imports, so the label logic stays fast to test."""
    mean_x, mean_y = fmean(x), fmean(y)
    covariance = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    variance = sum((xi - mean_x) ** 2 for xi in x)
    if variance == 0:
        # A market series with zero variance over the window can't
        # explain anything; fall back to beta=1 (assume it moves with
        # the market) rather than dividing by zero.
        return 1.0
    return covariance / variance


def paired_daily_returns(
    dates: list[str],
    closes: list[float],
    market_by_date: dict[str, float],
) -> tuple[list[float], list[float]]:
    """Daily returns for the stock and the market over the dates both
    actually traded. Aligning on shared dates rather than assuming the
    two series line up matters: a ticker halted for a day, or a series
    fetched over a slightly different window, would otherwise pair a
    stock's Tuesday against the market's Wednesday and quietly corrupt
    every beta downstream."""
    stock_returns: list[float] = []
    market_returns: list[float] = []
    for i in range(1, len(dates)):
        previous_date, current_date = dates[i - 1], dates[i]
        if previous_date not in market_by_date or current_date not in market_by_date:
            continue
        previous_market, current_market = market_by_date[previous_date], market_by_date[current_date]
        if closes[i - 1] == 0 or previous_market == 0:
            continue
        stock_returns.append((closes[i] - closes[i - 1]) / closes[i - 1])
        market_returns.append((current_market - previous_market) / previous_market)
    return stock_returns, market_returns


def compute_abnormal_fields(
    dates: list[str],
    closes: list[float],
    market_by_date: dict[str, float],
    reference_date: str,
    target_date: str,
) -> dict | None:
    """Turn a raw forward return into a market-adjusted, volatility-
    normalized one.

    Takes the reference/target dates already resolved by
    `compute_label_from_series` rather than re-deriving them, so the
    market-hours cutoff logic stays in exactly one place.

    Returns None when the label can't be computed honestly - missing
    benchmark data on either boundary date, too little history for a
    trustworthy beta, or a zero-volatility window. All real, expected
    cases, not errors.
    """
    if reference_date not in market_by_date or target_date not in market_by_date:
        return None

    reference_index = dates.index(reference_date)
    target_index = dates.index(target_date)

    market_reference = market_by_date[reference_date]
    market_target = market_by_date[target_date]
    if market_reference == 0 or closes[reference_index] == 0:
        return None

    stock_return = (closes[target_index] - closes[reference_index]) / closes[reference_index]
    market_return = (market_target - market_reference) / market_reference

    # Trailing window ending AT the reference close - never past it.
    window_start = max(0, reference_index - BETA_WINDOW)
    window_dates = dates[window_start : reference_index + 1]
    window_closes = closes[window_start : reference_index + 1]

    stock_returns, market_returns = paired_daily_returns(window_dates, window_closes, market_by_date)
    if len(stock_returns) < MIN_BETA_OBS:
        return None

    beta = ols_slope(market_returns, stock_returns)
    daily_volatility = stdev(stock_returns)
    if daily_volatility == 0:
        return None

    abnormal_return = stock_return - beta * market_return

    # Scale the trailing DAILY volatility to the label's horizon before
    # dividing, so the standardized value means "how many standard
    # deviations was this move, for this stock, over this many days" -
    # comparable across both tickers and horizons.
    horizon_days = target_index - reference_index
    horizon_volatility = daily_volatility * math.sqrt(horizon_days)

    return {
        "beta": beta,
        "market_return": market_return,
        "abnormal_return": abnormal_return,
        "abnormal_direction": 1 if abnormal_return > 0 else 0,
        "trailing_volatility": daily_volatility,
        "standardized_abnormal_return": abnormal_return / horizon_volatility,
    }
