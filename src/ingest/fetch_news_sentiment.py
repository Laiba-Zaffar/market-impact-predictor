import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.ingest.tickers import TICKER_UNIVERSE
from src.ingest.quota import calls_remaining, is_rate_limit_response, record_call
from src.storage.db import (
    get_connection,
    insert_article,
    insert_ticker_sentiment,
    is_window_fetched,
    mark_window_fetched,
)

load_dotenv()
API_KEY = os.environ["ALPHA_VANTAGE_API_KEY"]
API_URL = "https://www.alphavantage.co/query"
QUARTERS_BACK = 4  # one year of history


def quarter_windows(quarters_back: int) -> list[tuple[str, str, str]]:
    """Returns (time_from, time_to, quarter_key) tuples, oldest first."""
    now = datetime.now(timezone.utc)
    current_quarter_start_month = ((now.month - 1) // 3) * 3 + 1
    windows = []
    year, month = now.year, current_quarter_start_month
    for _ in range(quarters_back):
        start = datetime(year, month, 1, tzinfo=timezone.utc)
        end_month = month + 3
        end_year = year
        if end_month > 12:
            end_month -= 12
            end_year += 1
        end = min(datetime(end_year, end_month, 1, tzinfo=timezone.utc), now)
        quarter_key = f"{year}-Q{(month - 1) // 3 + 1}"
        windows.append((start.strftime("%Y%m%dT0000"), end.strftime("%Y%m%dT%H%M"), quarter_key))
        month -= 3
        if month <= 0:
            month += 12
            year -= 1
    return list(reversed(windows))


def fetch_window(ticker: str, time_from: str, time_to: str) -> dict:
    response = requests.get(
        API_URL,
        params={
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "time_from": time_from,
            "time_to": time_to,
            "limit": 1000,
            "sort": "EARLIEST",
            "apikey": API_KEY,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def store_feed(conn, feed: list[dict]) -> int:
    stored = 0
    for item in feed:
        article_id = insert_article(
            conn,
            {
                "url": item["url"],
                "title": item["title"],
                "source": item.get("source"),
                "time_published": item["time_published"],
                "overall_sentiment_score": item.get("overall_sentiment_score"),
                "overall_sentiment_label": item.get("overall_sentiment_label"),
                "summary": item.get("summary"),
            },
        )
        if article_id is None:
            continue  # already stored from an earlier overlapping window
        stored += 1
        for ts in item.get("ticker_sentiment", []):
            insert_ticker_sentiment(
                conn,
                {
                    "article_id": article_id,
                    "ticker": ts["ticker"],
                    "relevance_score": float(ts["relevance_score"]),
                    "ticker_sentiment_score": float(ts["ticker_sentiment_score"]),
                    "ticker_sentiment_label": ts["ticker_sentiment_label"],
                },
            )
    return stored


def run_once() -> None:
    conn = get_connection()
    windows = quarter_windows(QUARTERS_BACK)

    work_items = [
        (ticker, time_from, time_to, quarter_key)
        for ticker in TICKER_UNIVERSE
        for time_from, time_to, quarter_key in windows
        if not is_window_fetched(conn, ticker, quarter_key)
    ]

    if not work_items:
        print("Backfill complete - nothing left to fetch.")
        conn.close()
        return

    budget = calls_remaining()
    print(f"{len(work_items)} window(s) left to backfill, {budget} call(s) available today.")

    calls_made = 0
    for ticker, time_from, time_to, quarter_key in work_items:
        if calls_made >= budget:
            print("Daily quota budget reached for this run - stopping here, resume tomorrow.")
            break

        payload = fetch_window(ticker, time_from, time_to)
        record_call()
        calls_made += 1

        if is_rate_limit_response(payload):
            print(f"Alpha Vantage reports the rate limit is hit - stopping. Response: {payload}")
            break

        feed = payload.get("feed", [])
        new_count = store_feed(conn, feed)
        conn.commit()
        mark_window_fetched(conn, ticker, quarter_key, len(feed), datetime.now(timezone.utc).isoformat())
        conn.commit()

        print(f"  [{quarter_key}] {ticker}: {len(feed)} articles returned, {new_count} new")
        time.sleep(1)  # be polite between calls, not just quota-compliant

    conn.close()
    print(f"Done this run. {calls_made} call(s) used.")


if __name__ == "__main__":
    run_once()
