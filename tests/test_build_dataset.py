import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.features.build_dataset import time_based_split


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
