import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import joblib
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

from src.features.build_dataset import build_dataset, time_based_split
from src.model.config import FEATURES, MIN_EXAMPLES_FOR_SPLIT, MODELS_DIR

# Deliberately simple linear baselines, not gradient boosting or a neural
# net - same "baseline before deep learning" call as everywhere else in
# this project. Bonus: logistic/linear regression coefficients are
# directly interpretable ("which feature pushes the prediction which
# way"), which a black-box ensemble wouldn't give for free.


def train_direction_classifier(train_df, test_df):
    model = LogisticRegression()
    model.fit(train_df[FEATURES], train_df["direction"])

    predictions = model.predict(test_df[FEATURES])
    actual = test_df["direction"]

    print("\nDirection classifier:")
    print(f"  accuracy:  {accuracy_score(actual, predictions):.3f}")
    print(f"  precision: {precision_score(actual, predictions, zero_division=0):.3f}")
    print(f"  recall:    {recall_score(actual, predictions, zero_division=0):.3f}")
    print(f"  f1:        {f1_score(actual, predictions, zero_division=0):.3f}")
    print(f"  confusion matrix [[TN FP][FN TP]]:\n{confusion_matrix(actual, predictions)}")
    print(f"  coefficients: {dict(zip(FEATURES, model.coef_[0]))}")

    return model


def train_magnitude_regressor(train_df, test_df):
    model = LinearRegression()
    model.fit(train_df[FEATURES], train_df["return_pct"])

    predictions = model.predict(test_df[FEATURES])
    actual = test_df["return_pct"]

    print("\nMagnitude regressor:")
    print(f"  MAE:  {mean_absolute_error(actual, predictions):.4f}")
    print(f"  RMSE: {root_mean_squared_error(actual, predictions):.4f}")
    print(f"  coefficients: {dict(zip(FEATURES, model.coef_))}")

    return model


def run() -> None:
    dataset = build_dataset()

    if len(dataset) < MIN_EXAMPLES_FOR_SPLIT:
        print(
            f"\nOnly {len(dataset)} labeled examples available - below the "
            f"{MIN_EXAMPLES_FOR_SPLIT} minimum for a meaningful train/test "
            f"split. Training on everything anyway so the code path is "
            f"verified end-to-end, but these metrics are not a real "
            f"evaluation - re-run once the news backfill has more data."
        )
        train_df, test_df = dataset, dataset
    else:
        train_df, test_df = time_based_split(dataset)
        print(f"\ntrain={len(train_df)}  test={len(test_df)}")

    direction_model = train_direction_classifier(train_df, test_df)
    magnitude_model = train_magnitude_regressor(train_df, test_df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(direction_model, MODELS_DIR / "direction_classifier.joblib")
    joblib.dump(magnitude_model, MODELS_DIR / "magnitude_regressor.joblib")
    print(f"\nSaved models to {MODELS_DIR}")


if __name__ == "__main__":
    run()
