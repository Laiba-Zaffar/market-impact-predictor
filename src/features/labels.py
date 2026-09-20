import sys
from datetime import datetime, time as dtime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.storage.db import get_prices

# Approximate US market close in UTC (~4pm ET). Doesn't precisely handle
# the EST/EDT transition (off by up to an hour part of the year) - a
# known simplification, not a rigorous market-calendar implementation.
MARKET_CLOSE_UTC_HOUR = 20


def parse_time_published(time_published: str) -> datetime:
    """Alpha Vantage format: '20260919T141725', always UTC."""
    return datetime.strptime(time_published, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def compute_label_from_series(dates: list[str], closes: list[float], time_published: str, horizon_days: int) -> dict | None:
    """Pure function: given a ticker's trading-day series (ascending, no
    gaps other than non-trading days) and an article's publish time,
    find the reference close (day of/after publication) and the target
    close `horizon_days` trading days later, and compute the label.

    Returns None if there isn't enough price history on either side -
    e.g. the article predates our data, or there aren't horizon_days of
    future trading days yet (both real, expected cases, not errors).
    """
    published_at = parse_time_published(time_published)
    article_date = published_at.date().isoformat()
    cutoff = datetime.combine(published_at.date(), dtime(MARKET_CLOSE_UTC_HOUR, 0), tzinfo=timezone.utc)

    if article_date in dates and published_at <= cutoff:
        # Published during (or before) that day's market session - that
        # day's close already reflects the market's initial reaction.
        ref_idx = dates.index(article_date)
    else:
        # Published after close, or on a non-trading day (weekend/holiday) -
        # the market can't react until the next session opens.
        ref_idx = next((i for i, d in enumerate(dates) if d > article_date), None)
        if ref_idx is None:
            return None

    target_idx = ref_idx + horizon_days
    if target_idx >= len(dates):
        return None

    reference_close = closes[ref_idx]
    target_close = closes[target_idx]
    return {
        "reference_date": dates[ref_idx],
        "target_date": dates[target_idx],
        "reference_close": reference_close,
        "target_close": target_close,
        "direction": 1 if target_close > reference_close else 0,
        "return_pct": (target_close - reference_close) / reference_close,
    }


def compute_label(conn, ticker: str, time_published: str, horizon_days: int) -> dict | None:
    rows = get_prices(conn, ticker)
    dates = [r["date"] for r in rows]
    closes = [r["close"] for r in rows]
    return compute_label_from_series(dates, closes, time_published, horizon_days)
