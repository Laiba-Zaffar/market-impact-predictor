import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from src.features.build_dataset import build_dataset, time_based_split
from src.model.config import FEATURES, MIN_EXAMPLES_FOR_SPLIT

# A raw LogisticRegression's predict_proba is *a* probability, but not
# necessarily a *calibrated* one - "70% confident" should mean right
# about 70% of the time, and there's no guarantee of that out of the box.
# This is exactly the gap between "the model outputs a number that looks
# like a probability" and "the number is actually trustworthy," which
# matters a lot more in finance than in a toy classification demo.


def brier_and_curve(model, test_df) -> tuple[float, tuple]:
    probs = model.predict_proba(test_df[FEATURES])[:, 1]
    actual = test_df["direction"]
    brier = brier_score_loss(actual, probs)
    curve = calibration_curve(actual, probs, n_bins=5, strategy="quantile")
    return brier, curve


def run() -> None:
    dataset = build_dataset()

    if len(dataset) < MIN_EXAMPLES_FOR_SPLIT:
        print(
            f"Only {len(dataset)} labeled examples - too few to meaningfully "
            f"calibrate or evaluate calibration (CalibratedClassifierCV's "
            f"cross-validation folds would be near-empty). Skipping until "
            f"the backfill has more data. This is expected right now, not "
            f"a bug."
        )
        return

    train_df, test_df = time_based_split(dataset)

    uncalibrated = LogisticRegression()
    uncalibrated.fit(train_df[FEATURES], train_df["direction"])
    raw_brier, raw_curve = brier_and_curve(uncalibrated, test_df)

    calibrated = CalibratedClassifierCV(LogisticRegression(), method="sigmoid", cv=3)
    calibrated.fit(train_df[FEATURES], train_df["direction"])
    cal_brier, cal_curve = brier_and_curve(calibrated, test_df)

    print(f"Brier score (lower is better) - raw: {raw_brier:.4f}, calibrated: {cal_brier:.4f}")
    print(f"Raw reliability curve (predicted, observed): {list(zip(*raw_curve[::-1]))}")
    print(f"Calibrated reliability curve (predicted, observed): {list(zip(*cal_curve[::-1]))}")

    if cal_brier > raw_brier:
        print(
            "Calibration made the Brier score worse, not better - plausible "
            "with this little data (CalibratedClassifierCV needs enough "
            "examples per fold to learn a real correction, not noise). "
            "Worth re-checking once there's more data, not something to "
            "paper over."
        )


if __name__ == "__main__":
    run()
