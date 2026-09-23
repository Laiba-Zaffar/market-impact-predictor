import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.features.events import is_event_worthy
from src.features.labels import compute_label_from_series
from src.features.market_adjust import compute_abnormal_fields
from src.ingest.tickers import BENCHMARK_TICKER
from src.storage.db import get_all_article_topics, get_connection, get_prices

HORIZON_DAYS = 3
OUTPUT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "dataset.csv"
EVENT_OUTPUT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "events.csv"

# Alpha Vantage floors relevance at 0.30 and assigns a row for every
# ticker an article mentions at all, so a passing mention of AAPL inside
# an article about someone else becomes an "AAPL training example."
# A modest floor removes the weakest of those links. Deliberately not
# aggressive - only ~9% of rows sit below 0.5, so this is a cheap tidy-up
# rather than one of the load-bearing fixes.
MIN_RELEVANCE = 0.5

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

# Columns that are a property of the (ticker, reference_date) cell rather
# than of any one article - identical for every article in the group, so
# aggregation takes the first rather than averaging identical values.
LABEL_COLUMNS = [
    "reference_date", "target_date", "reference_close", "target_close",
    "direction", "return_pct", "beta", "market_return", "abnormal_return",
    "abnormal_direction", "trailing_volatility", "standardized_abnormal_return",
]


def topic_features_for(article_topics: dict[str, float]) -> dict[str, float]:
    """Pivot a {topic: relevance_score} dict (only the topics an article
    actually had) into a fixed-width feature dict covering every known
    category, 0.0 where absent. A pure function so the pivoting logic is
    testable without a database."""
    return {f"topic_{t}": article_topics.get(t, 0.0) for t in TOPIC_CATEGORIES}


def aggregate_to_events(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse article-level rows to one row per (ticker, reference_date).

    Every article about AAPL that resolves to the same reference close
    carries the *identical* forward label. Measured on this corpus, each
    label cell was reused ~21 times - so 55,532 "examples" were really
    2,630 independent observations, and near-duplicates of most test
    labels sat in the training set. That inflates apparent sample size,
    leaks across the train/test boundary, and makes every significance
    estimate downstream ~4.6x (sqrt(21)) too confident.

    Collapsing to the event level is the honest unit: one label, one row.
    The article count survives as `n_articles`, which is a real feature
    in its own right - a burst of coverage on one name in one day is
    itself information, and it was previously being expressed only as
    duplicated rows.
    """
    sentiment_aggregations = {
        "overall_sentiment_score": ["mean"],
        "ticker_sentiment_score": ["mean", "max", "min"],
        "relevance_score": ["mean", "max"],
    }
    # Topics: max across the day's articles, i.e. "did ANY article today
    # carry this topic, and how strongly" - a mean would dilute a single
    # genuine earnings story among a day's routine coverage.
    topic_aggregations = {column: ["max"] for column in TOPIC_FEATURES}

    grouped = df.groupby(["ticker", "reference_date"], as_index=False).agg(
        {
            **sentiment_aggregations,
            **topic_aggregations,
            **{column: ["first"] for column in LABEL_COLUMNS if column != "reference_date"},
            "time_published": ["min"],
            "article_id": ["count"],
        }
    )
    grouped.columns = [
        column if not aggregation or aggregation in ("first", "min")
        else f"{column}_{aggregation}"
        for column, aggregation in grouped.columns
    ]
    grouped = grouped.rename(columns={"article_id_count": "n_articles"})
    return grouped


def build_dataset(horizon_days: int = HORIZON_DAYS, apply_event_filter: bool = True) -> pd.DataFrame:
    """Article-level labeled rows, market-adjusted.

    `apply_event_filter=False` is kept as an escape hatch so the cost of
    the filter can be measured directly (build both, compare) rather
    than assumed.
    """
    conn = get_connection()
    rows = conn.execute(
        """SELECT a.id, a.time_published, a.overall_sentiment_score, a.title, a.source,
                  ts.ticker, ts.relevance_score, ts.ticker_sentiment_score
           FROM articles a
           JOIN ticker_sentiment ts ON ts.article_id = a.id
           ORDER BY a.time_published"""
    ).fetchall()

    # The benchmark series, once - every row needs it, and it's the same
    # series for all of them.
    benchmark_rows = get_prices(conn, BENCHMARK_TICKER)
    market_by_date = {r["date"]: r["close"] for r in benchmark_rows}
    if not market_by_date:
        conn.close()
        raise SystemExit(
            f"No {BENCHMARK_TICKER} price data in the database. Abnormal-return labels "
            f"can't be computed without a market benchmark - run "
            f"`python -m src.ingest.fetch_prices` first."
        )

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

    # Third instance of the same lesson: an abnormal-return label depends
    # only on (ticker, reference_date, target_date), but it was being
    # recomputed per article row - and each computation runs an OLS beta
    # over a 120-day window. With ~13 articles per event cell that's an
    # order of magnitude of wasted regressions, and it showed up as the
    # build taking minutes instead of seconds.
    abnormal_cache: dict[tuple[str, str, str], dict | None] = {}

    def get_abnormal(ticker: str, dates, closes, reference_date: str, target_date: str) -> dict | None:
        key = (ticker, reference_date, target_date)
        if key not in abnormal_cache:
            abnormal_cache[key] = compute_abnormal_fields(
                dates, closes, market_by_date, reference_date, target_date
            )
        return abnormal_cache[key]

    records = []
    skipped_no_price = 0
    skipped_not_event = 0
    skipped_low_relevance = 0
    skipped_no_benchmark = 0
    for article_id, time_published, overall_sentiment, title, source, ticker, relevance, ticker_sentiment in rows:
        if apply_event_filter and not is_event_worthy(title, source):
            skipped_not_event += 1
            continue
        if relevance is not None and relevance < MIN_RELEVANCE:
            skipped_low_relevance += 1
            continue

        dates, closes = get_series(ticker)
        label = compute_label_from_series(dates, closes, time_published, horizon_days)
        if label is None:
            # Article too recent (not enough future trading days yet) or
            # predates our price history - not an error, just not usable
            # as a training example yet.
            skipped_no_price += 1
            continue

        abnormal = get_abnormal(
            ticker, dates, closes, label["reference_date"], label["target_date"]
        )
        if abnormal is None:
            # No benchmark data on a boundary date, or too little history
            # for a trustworthy beta. Refusing beats emitting a label
            # whose market component is a guess.
            skipped_no_benchmark += 1
            continue

        records.append(
            {
                "article_id": article_id,
                "ticker": ticker,
                "time_published": time_published,
                # Carried through for M10 - the headline is the only
                # feature in this project that isn't a number someone
                # else computed.
                "title": title,
                "overall_sentiment_score": overall_sentiment,
                "relevance_score": relevance,
                "ticker_sentiment_score": ticker_sentiment,
                **topic_features_for(topics_by_article.get(article_id, {})),
                **label,
                **abnormal,
            }
        )
    conn.close()

    df = pd.DataFrame(records)
    print(f"Built {len(df)} labeled article-level rows from {len(rows)} article-ticker pairs.")
    print(f"  skipped {skipped_not_event} (no event in the article)")
    print(f"  skipped {skipped_low_relevance} (relevance < {MIN_RELEVANCE})")
    print(f"  skipped {skipped_no_price} (not enough price history around them)")
    print(f"  skipped {skipped_no_benchmark} (no benchmark / too little history for beta)")
    return df


def build_event_dataset(horizon_days: int = HORIZON_DAYS, apply_event_filter: bool = True) -> pd.DataFrame:
    """The dataset the models should actually train on: one row per
    (ticker, reference_date), market-adjusted, event-filtered."""
    article_level = build_dataset(horizon_days, apply_event_filter)
    if article_level.empty:
        return article_level
    events = aggregate_to_events(article_level)
    print(f"Aggregated to {len(events)} event-level rows (from {len(article_level)} article rows).")
    return events


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
    events = build_event_dataset()
    train_df, test_df = time_based_split(events)
    print(f"train={len(train_df)}  test={len(test_df)}")

    EVENT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(EVENT_OUTPUT_PATH, index=False)
    print(f"Saved event-level dataset to {EVENT_OUTPUT_PATH}")
