from pathlib import Path

# Shared constants with no heavy dependencies, so modules that only need
# these (like backtest.py) don't have to import sklearn transitively just
# to read a threshold or a feature list.
FEATURES = ["overall_sentiment_score", "relevance_score", "ticker_sentiment_score"]
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MIN_EXAMPLES_FOR_SPLIT = 30
