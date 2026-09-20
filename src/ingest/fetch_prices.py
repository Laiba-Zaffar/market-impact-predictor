import sys
from pathlib import Path

import yfinance as yf

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.ingest.tickers import TICKER_UNIVERSE
from src.storage.db import get_connection, insert_price

# One extra month of buffer past the news backfill window, so even an
# article from the oldest covered quarter has enough future trading days
# to compute a multi-day forward return without running off the end of
# the price history.
PERIOD = "14mo"


def fetch_ticker_prices(ticker: str):
    history = yf.Ticker(ticker).history(period=PERIOD, interval="1d")
    return history


def run_once() -> None:
    conn = get_connection()
    total_rows = 0

    for ticker in TICKER_UNIVERSE:
        history = fetch_ticker_prices(ticker)
        if history.empty:
            print(f"  {ticker}: no data returned")
            continue

        new_rows = 0
        for date, row in history.iterrows():
            insert_price(
                conn,
                {
                    "ticker": ticker,
                    "date": date.strftime("%Y-%m-%d"),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                },
            )
            new_rows += 1
        conn.commit()
        total_rows += new_rows
        print(f"  {ticker}: {new_rows} trading days stored")

    conn.close()
    print(f"Done. {total_rows} price rows stored across {len(TICKER_UNIVERSE)} tickers.")


if __name__ == "__main__":
    run_once()
