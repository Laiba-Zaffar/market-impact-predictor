"""Runs M3 through M6 end-to-end: build the dataset, train the baseline
models, check calibration, and backtest. Re-run this any time the news
backfill has added more data - nothing here needs to change, just the
data underneath it growing."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.eval import backtest
from src.model import calibrate, train_baseline

if __name__ == "__main__":
    print("=== M4: training baseline models ===")
    train_baseline.run()

    print("\n=== M5: checking calibration ===")
    calibrate.run()

    print("\n=== M6: backtest ===")
    backtest.run()
