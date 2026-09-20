import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import joblib
import pandas as pd

from src.features.build_dataset import build_dataset, time_based_split
from src.model.config import FEATURES, MIN_EXAMPLES_FOR_SPLIT, MODELS_DIR

CONFIDENCE_THRESHOLD = 0.55  # only take a position when the model is at least this sure


def run_backtest(test_df: pd.DataFrame, direction_model, confidence_threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    up_confidence = direction_model.predict_proba(test_df[FEATURES])[:, 1]

    strategy_returns = []
    n_trades = 0
    for p_up, actual_return in zip(up_confidence, test_df["return_pct"]):
        if p_up >= confidence_threshold:
            strategy_returns.append(actual_return)  # long
            n_trades += 1
        elif (1 - p_up) >= confidence_threshold:
            strategy_returns.append(-actual_return)  # short
            n_trades += 1
        else:
            strategy_returns.append(0.0)  # flat - not confident enough to trade

    return {
        "n_examples": len(test_df),
        "n_trades": n_trades,
        "strategy_cumulative_return": sum(strategy_returns),
        "buy_hold_cumulative_return": test_df["return_pct"].sum(),
    }


def run() -> None:
    dataset = build_dataset()
    if len(dataset) < MIN_EXAMPLES_FOR_SPLIT:
        print(f"Only {len(dataset)} examples - too few for a backtest that means anything. Skipping.")
        return

    _, test_df = time_based_split(dataset)
    direction_model = joblib.load(MODELS_DIR / "direction_classifier.joblib")

    result = run_backtest(test_df, direction_model)

    print("\n--- NAIVE BACKTEST (see caveats below - not investment advice) ---")
    print(f"Test examples: {result['n_examples']}, trades taken: {result['n_trades']}")
    print(f"Strategy cumulative return: {result['strategy_cumulative_return']:.4f}")
    print(f"Buy-and-hold cumulative return (same examples, no model): {result['buy_hold_cumulative_return']:.4f}")
    print(
        """
CAVEATS - read before drawing any conclusion from the numbers above:
  - No transaction costs, slippage, or spread modeled - real trading
    would erode returns, possibly past whatever edge shows up here.
  - No position sizing or risk management - assumes unlimited capital
    and ignores concentration risk across overlapping trades in the
    same ticker.
  - The 21-ticker universe was hand-picked (large, liquid, currently
    healthy companies) - not an unbiased trading universe, and that
    selection itself is a form of hindsight bias.
  - Covers one historical period only - results are specific to
    whatever market regime that was, not a general claim about the
    strategy.
  - Small sample size until the backfill finishes - treat every number
    here as illustrating the *mechanism*, not a real performance claim,
    until there's enough data to say otherwise.
"""
    )


if __name__ == "__main__":
    run()
