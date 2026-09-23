import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.features.build_dataset import LABEL_COLUMNS, TOPIC_FEATURES, aggregate_to_events
from src.features.events import filter_stats, is_event_worthy

# Every title below is copied verbatim from the project's own corpus, not
# invented - a filter tuned against imagined examples tends to pass its
# own tests and fail on real data.
REAL_NO_EVENT_TITLES = [
    ("GRIMES & Co WEALTH MANAGEMENT LLC Sells 3,223 Shares of Alphabet Inc. $GOOGL", "MarketBeat"),
    ("Canoe Financial LP Sells 36,525 Shares of Microsoft Corporation $MSFT", "MarketBeat"),
    ("UNIVEST FINANCIAL Corp Trims Stock Position in Alphabet Inc. $GOOGL", "MarketBeat"),
    ("UBS Group Cuts PPG Industries (NYSE:PPG) Price Target to $110.00", "Some Wire"),
    ("Is Johnson & Johnson Still the Ultimate Safe Dividend Stock to Buy?", "The Globe and Mail"),
    ("Where Is Plug Power Stock Headed?", "Some Wire"),
    ("Why KLA Corporation (KLAC) Stock Is Down Today", "Some Wire"),
]

REAL_EVENT_TITLES = [
    ("UnitedHealth Group (UNH) Announces 5% Dividend Increase", "GuruFocus"),
    ("Johnson & Johnson Invests $300,000 to Expand Nursing Education at Rutgers-Camden", "TAPinto"),
    ("IAMGOLD Corp. (IAG) Tops Q1 EPS by 11c", "StreetInsider"),
    ("Enovix Shares Climb After Company Names Former Apple AirPods Manufacturing Leader as COO", "Some Wire"),
]


def test_real_no_event_titles_are_all_filtered_out():
    for title, source in REAL_NO_EVENT_TITLES:
        assert not is_event_worthy(title, source), f"should have been dropped: {title}"


def test_real_event_titles_survive_the_filter():
    for title, source in REAL_EVENT_TITLES:
        assert is_event_worthy(title, source), f"should have been kept: {title}"


def test_content_mill_source_is_dropped_regardless_of_title():
    # Even a title with no bad pattern in it - the source alone is enough.
    assert not is_event_worthy("Apple announces new product", "MarketBeat")


def test_empty_and_whitespace_titles_are_not_events():
    assert not is_event_worthy("", "Reuters")
    assert not is_event_worthy("   ", "Reuters")


def test_source_is_optional():
    # Callers that only have a title shouldn't have to fabricate a source.
    assert is_event_worthy("Pfizer Announces FDA Approval of New Treatment")
    assert not is_event_worthy("Should You Buy Pfizer Stock Right Now?")


def test_filter_stats_separates_source_drops_from_pattern_drops():
    articles = [
        ("Apple announces new product", "MarketBeat"),        # dropped: source
        ("Why Apple Stock Is Down Today", "Reuters"),          # dropped: pattern
        ("Apple Announces $50 Billion Buyback", "Reuters"),    # kept
    ]
    stats = filter_stats(articles)
    assert stats["total"] == 3
    assert stats["kept"] == 1
    assert stats["dropped_by_source"] == 1
    assert stats["dropped_by_pattern"] == 1


def make_article_rows(n_articles: int) -> pd.DataFrame:
    """n articles about the same ticker resolving to the same reference
    close - i.e. n rows that all carry one identical label."""
    base = {column: 0.0 for column in TOPIC_FEATURES}
    base.update({column: 1.0 for column in LABEL_COLUMNS if column not in ("reference_date", "target_date")})
    return pd.DataFrame(
        [
            {
                **base,
                "article_id": i,
                "ticker": "AAPL",
                "reference_date": "2026-03-02",
                "target_date": "2026-03-05",
                "time_published": f"20260302T1{i:02d}000",
                "overall_sentiment_score": 0.1 * i,
                "ticker_sentiment_score": 0.1 * i,
                "relevance_score": 0.9,
                "topic_earnings": 1.0 if i == 0 else 0.0,
            }
            for i in range(n_articles)
        ]
    )


def test_aggregation_collapses_duplicate_labels_to_one_row():
    # The core fix: 20 articles sharing one label are one observation,
    # not 20. Counting them as 20 overstates the sample ~4.5x in any
    # significance estimate (sqrt(20)).
    events = aggregate_to_events(make_article_rows(20))
    assert len(events) == 1
    assert events.iloc[0]["n_articles"] == 20


def test_aggregation_keeps_article_count_as_a_feature():
    # Coverage volume is real information - it was previously expressed
    # only as duplicated rows, where the model couldn't use it.
    quiet = aggregate_to_events(make_article_rows(2)).iloc[0]["n_articles"]
    busy = aggregate_to_events(make_article_rows(30)).iloc[0]["n_articles"]
    assert quiet == 2 and busy == 30


def test_aggregation_takes_the_max_not_the_mean_of_topic_flags():
    # One genuine earnings story among 19 routine articles should still
    # read as "earnings happened today", not as 1/20th of an earnings.
    events = aggregate_to_events(make_article_rows(20))
    assert events.iloc[0]["topic_earnings_max"] == 1.0


def test_aggregation_preserves_the_shared_label_exactly():
    events = aggregate_to_events(make_article_rows(12)).iloc[0]
    assert events["reference_date"] == "2026-03-02"
    assert events["target_date"] == "2026-03-05"
    assert events["abnormal_direction"] == 1.0


def test_aggregation_separates_different_reference_dates():
    first = make_article_rows(5)
    second = make_article_rows(5)
    second["reference_date"] = "2026-03-03"
    second["article_id"] = second["article_id"] + 100
    events = aggregate_to_events(pd.concat([first, second], ignore_index=True))
    assert len(events) == 2
    assert set(events["n_articles"]) == {5}
