import sys
from unittest.mock import MagicMock
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.eval.backtest import run_backtest


def make_test_df():
    return pd.DataFrame(
        {
            "overall_sentiment_score": [0.5, -0.3, 0.1, 0.2],
            "relevance_score": [0.9, 0.8, 0.5, 0.6],
            "ticker_sentiment_score": [0.4, -0.2, 0.05, 0.1],
            "return_pct": [0.02, -0.01, 0.005, -0.03],
        }
    )


def make_fake_model(up_confidences):
    # Real sklearn models return a numpy array from predict_proba, and
    # run_backtest slices it numpy-style ([:, 1]) - a plain list here
    # would silently pass a naive mock but fail against the real thing.
    model = MagicMock()
    model.predict_proba.return_value = np.array([[1 - p, p] for p in up_confidences])
    return model


def test_confident_up_prediction_goes_long():
    df = make_test_df()
    # Confident "up" (0.9) on the first row only - should count as a long
    # trade capturing that row's real return.
    model = make_fake_model([0.9, 0.5, 0.5, 0.5])
    result = run_backtest(df, model, confidence_threshold=0.55)
    assert result["n_trades"] == 1
    assert abs(result["strategy_cumulative_return"] - 0.02) < 1e-9


def test_confident_down_prediction_goes_short_and_flips_sign():
    df = make_test_df()
    # Confident "down" (p_up=0.1, so 1-p_up=0.9) on row 1 (return -0.01) -
    # a short position profits from a negative return, so the strategy's
    # captured return should be +0.01, not -0.01.
    model = make_fake_model([0.5, 0.1, 0.5, 0.5])
    result = run_backtest(df, model, confidence_threshold=0.55)
    assert result["n_trades"] == 1
    assert abs(result["strategy_cumulative_return"] - 0.01) < 1e-9


def test_low_confidence_stays_flat():
    df = make_test_df()
    model = make_fake_model([0.52, 0.51, 0.5, 0.53])  # all below threshold
    result = run_backtest(df, model, confidence_threshold=0.55)
    assert result["n_trades"] == 0
    assert result["strategy_cumulative_return"] == 0.0


def test_buy_hold_return_ignores_the_model_entirely():
    df = make_test_df()
    model = make_fake_model([0.9, 0.9, 0.9, 0.9])  # model's calls don't matter here
    result = run_backtest(df, model, confidence_threshold=0.55)
    assert abs(result["buy_hold_cumulative_return"] - df["return_pct"].sum()) < 1e-9
