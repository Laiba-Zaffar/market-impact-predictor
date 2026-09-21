import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.features.build_dataset import TOPIC_CATEGORIES, time_based_split, topic_features_for


def make_df(n: int) -> pd.DataFrame:
    # Deliberately out of order, to verify the split sorts by time first.
    times = [f"2026010{i}T120000" for i in range(n, 0, -1)]
    return pd.DataFrame({"time_published": times, "value": range(n)})


def test_split_is_time_ordered_not_random():
    df = make_df(10)
    train, test = time_based_split(df, test_frac=0.3)
    # Every train timestamp must be <= every test timestamp - the whole
    # point of a time-based split.
    assert train["time_published"].max() <= test["time_published"].min()


def test_split_sizes_respect_test_fraction():
    df = make_df(10)
    train, test = time_based_split(df, test_frac=0.2)
    assert len(train) == 8
    assert len(test) == 2


def test_split_covers_the_whole_dataset():
    df = make_df(7)
    train, test = time_based_split(df, test_frac=0.3)
    assert len(train) + len(test) == len(df)


def test_topic_features_covers_every_known_category():
    features = topic_features_for({"earnings": 0.8})
    assert set(features.keys()) == {f"topic_{t}" for t in TOPIC_CATEGORIES}


def test_topic_features_defaults_absent_topics_to_zero():
    features = topic_features_for({"earnings": 0.8})
    assert features["topic_earnings"] == 0.8
    assert features["topic_technology"] == 0.0


def test_topic_features_handles_no_topics_at_all():
    # An older article with no topic data captured - every feature is 0.0,
    # not missing/NaN, so the model sees "no signal" rather than a hole.
    features = topic_features_for({})
    assert all(v == 0.0 for v in features.values())
