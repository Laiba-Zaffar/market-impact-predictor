import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.features.labels import compute_label_from_series
from src.storage.db import get_all_article_topics, get_connection, get_prices

HORIZON_DAYS = 3
OUTPUT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "dataset.csv"

# Alpha Vantage's fixed topic vocabulary. Hardcoded rather than discovered
# from whatever's in the DB at build time, so the feature schema stays
# stable across runs - a saved model needs to know exactly which columns
# to expect, not "whichever topics happened to show up so far."
TOPIC_CATEGORIES = [
    "blockchain", "earnings", "ipo", "mergers_and_acquisitions",
    "financial_markets", "economy_fiscal", "economy_monetary", "economy_macro",
    "energy_transportation", "finance", "life_sciences", "manufacturing",
    "real_estate", "retail_wholesale", "technology",
]
TOPIC_FEATURES = [f"topic_{t}" for t in TOPIC_CATEGORIES]


def topic_features_for(article_topics: dict[str, float]) -> dict[str, float]:
    """Pivot a {topic: relevance_score} dict (only the topics an article
    actually had) into a fixed-width feature dict covering every known
    category, 0.0 where absent. A pure function so the pivoting logic is
    testable without a database."""
    return {f"topic_{t}": article_topics.get(t, 0.0) for t in TOPIC_CATEGORIES}


def build_dataset(horizon_days: int = HORIZON_DAYS) -> pd.DataFrame:
    conn = get_connection()
    rows = conn.execute(
        """SELECT a.id, a.time_published, a.overall_sentiment_score,
                  ts.ticker, ts.relevance_score, ts.ticker_sentiment_score
           FROM articles a
           JOIN ticker_sentiment ts ON ts.article_id = a.id
           ORDER BY a.time_published"""
    ).fetchall()

    # Cache each ticker's price series once instead of re-querying the DB
    # per row - with tens of thousands of articles, a fresh query per row
    # turned this into the dominant cost. There are only as many distinct
    # series to cache as there are tickers, not as many as there are rows.
    price_series_cache: dict[str, tuple[list[str], list[float]]] = {}

    def get_series(ticker: str) -> tuple[list[str], list[float]]:
        if ticker not in price_series_cache:
            price_rows = get_prices(conn, ticker)
            price_series_cache[ticker] = (
                [r["date"] for r in price_rows],
                [r["close"] for r in price_rows],
            )
        return price_series_cache[ticker]

    # One query for every article's topics, not one per row - same fix as
    # the price-series cache above, applied before it became a problem
    # this time instead of after.
    topics_by_article = get_all_article_topics(conn)

    records = []
    skipped = 0
    articles_with_topics = 0
    for article_id, time_published, overall_sentiment, ticker, relevance, ticker_sentiment in rows:
        dates, closes = get_series(ticker)
        label = compute_label_from_series(dates, closes, time_published, horizon_days)
        if label is None:
            # Article too recent (not enough future trading days yet) or
            # predates our price history - not an error, just not usable
            # as a training example yet.
            skipped += 1
            continue

        article_topics = topics_by_article.get(article_id, {})
        if article_topics:
            articles_with_topics += 1
        topic_features = topic_features_for(article_topics)

        records.append(
            {
                "article_id": article_id,
                "ticker": ticker,
                "time_published": time_published,
                "overall_sentiment_score": overall_sentiment,
                "relevance_score": relevance,
                "ticker_sentiment_score": ticker_sentiment,
                **topic_features,
                **label,
            }
        )
    conn.close()

    df = pd.DataFrame(records)
    print(f"Built {len(df)} labeled examples ({skipped} skipped - not enough price history around them).")
    print(f"{articles_with_topics}/{len(df)} examples have real topic data (older articles predate this feature).")
    return df


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time, not randomly. A random split would let the model
    train on articles published *after* some test examples - meaning it
    could implicitly learn from information that wouldn't have existed
    yet at prediction time. That's lookahead leakage, and it's exactly
    the kind of mistake that makes a backtest look great and a live
    model fail."""
    df = df.sort_values("time_published").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


if __name__ == "__main__":
    dataset = build_dataset()
    train_df, test_df = time_based_split(dataset)
    print(f"train={len(train_df)}  test={len(test_df)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved full dataset to {OUTPUT_PATH}")
