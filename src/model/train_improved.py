"""Second iteration, built on evidence from the baseline, not a bigger
model for its own sake. train_baseline.py (plain LogisticRegression /
LinearRegression on 3 raw features) collapsed to predicting the majority
class - 56.5% accuracy, exactly the base rate. That's a real, measured
reason to invest further, not an assumption that more complexity helps.

Three changes, all usable with data already collected (no new API calls):
  1. Feature scaling (StandardScaler) - logistic regression is sensitive
     to feature scale; the baseline had none.
  2. Class balancing (class_weight="balanced") - with weak signal, an
     unbalanced classifier gravitates to the majority class rather than
     using what little signal exists. This directly targets that failure
     mode from the baseline run.
  3. Ticker as a feature (one-hot) - different companies plausibly have
     different baseline drift the 3 sentiment scores alone can't capture.

Then, only after those are in place: a non-linear model
(HistGradientBoostingClassifier/Regressor) to check whether the 3
features contain interactions a linear model can't use, now that there's
enough data (31k+ examples) for a more expressive model to be justified
rather than just overfitting.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build_dataset import build_dataset, time_based_split
from src.model.config import MIN_EXAMPLES_FOR_SPLIT, MODELS_DIR

NUMERIC_FEATURES = ["overall_sentiment_score", "relevance_score", "ticker_sentiment_score"]
CATEGORICAL_FEATURES = ["ticker"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def evaluate_direction(name: str, model, train_df, test_df):
    model.fit(train_df[ALL_FEATURES], train_df["direction"])
    predictions = model.predict(test_df[ALL_FEATURES])
    actual = test_df["direction"]

    print(f"\n{name}:")
    print(f"  accuracy:  {accuracy_score(actual, predictions):.3f}")
    print(f"  precision: {precision_score(actual, predictions, zero_division=0):.3f}")
    print(f"  recall:    {recall_score(actual, predictions, zero_division=0):.3f}")
    print(f"  f1:        {f1_score(actual, predictions, zero_division=0):.3f}")
    print(f"  confusion matrix [[TN FP][FN TP]]:\n{confusion_matrix(actual, predictions)}")
    return model, f1_score(actual, predictions, zero_division=0)


def evaluate_magnitude(name: str, model, train_df, test_df):
    model.fit(train_df[ALL_FEATURES], train_df["return_pct"])
    predictions = model.predict(test_df[ALL_FEATURES])
    actual = test_df["return_pct"]

    mae = mean_absolute_error(actual, predictions)
    print(f"\n{name}:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {root_mean_squared_error(actual, predictions):.4f}")
    return model, mae


def run() -> None:
    dataset = build_dataset()
    if len(dataset) < MIN_EXAMPLES_FOR_SPLIT:
        print(f"Only {len(dataset)} examples - below the {MIN_EXAMPLES_FOR_SPLIT} minimum. Skipping.")
        return

    train_df, test_df = time_based_split(dataset)
    print(f"train={len(train_df)}  test={len(test_df)}")

    print("\n" + "=" * 60)
    print("DIRECTION CLASSIFIER - comparing configs")
    print("=" * 60)

    baseline = Pipeline([("prep", make_preprocessor()), ("clf", LogisticRegression())])
    balanced = Pipeline(
        [("prep", make_preprocessor()), ("clf", LogisticRegression(class_weight="balanced"))]
    )
    boosted = Pipeline([("prep", make_preprocessor()), ("clf", HistGradientBoostingClassifier())])

    results = []
    for name, model in [
        ("Logistic + scaling + ticker (no balancing)", baseline),
        ("Logistic + scaling + ticker + class_weight=balanced", balanced),
        ("HistGradientBoosting + scaling + ticker", boosted),
    ]:
        fitted, f1 = evaluate_direction(name, model, train_df, test_df)
        results.append((name, fitted, f1))

    best_name, best_direction_model, best_f1 = max(results, key=lambda r: r[2])
    print(f"\nBest by F1: {best_name} (f1={best_f1:.3f})")

    print("\n" + "=" * 60)
    print("MAGNITUDE REGRESSOR - comparing configs")
    print("=" * 60)

    linear = Pipeline([("prep", make_preprocessor()), ("reg", LinearRegression())])
    boosted_reg = Pipeline([("prep", make_preprocessor()), ("reg", HistGradientBoostingRegressor())])

    mag_results = []
    for name, model in [
        ("Linear + scaling + ticker", linear),
        ("HistGradientBoosting + scaling + ticker", boosted_reg),
    ]:
        fitted, mae = evaluate_magnitude(name, model, train_df, test_df)
        mag_results.append((name, fitted, mae))

    best_mag_name, best_magnitude_model, best_mae = min(mag_results, key=lambda r: r[2])
    print(f"\nBest by MAE: {best_mag_name} (mae={best_mae:.4f})")

    # Saved under different filenames than train_baseline.py's output on
    # purpose - this pipeline expects a "ticker" column too, not just the
    # 3 numeric features, so it isn't a drop-in replacement for whatever
    # calibrate.py/backtest.py currently load. Wiring those up to the new
    # feature set is the next step, not done silently as a side effect
    # of running this comparison.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_direction_model, MODELS_DIR / "direction_classifier_v2.joblib")
    joblib.dump(best_magnitude_model, MODELS_DIR / "magnitude_regressor_v2.joblib")
    print(f"\nSaved best models ({best_name} / {best_mag_name}) to {MODELS_DIR} as *_v2.joblib")


if __name__ == "__main__":
    run()
