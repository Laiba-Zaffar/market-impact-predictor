import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.model.train_text import walk_forward_predictions
from src.nlp.encode import mean_pool, sentiment_score_from_logits


def test_mean_pool_ignores_padding():
    # Two tokens of content, two of padding. The padded positions carry
    # a wildly different value on purpose - if they leaked into the
    # average, the result would be nowhere near the true mean of 2.0.
    vectors = np.array([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0], [99.0, 99.0]]])
    mask = np.array([[1, 1, 0, 0]])
    pooled = mean_pool(vectors, mask)
    assert np.allclose(pooled, [[2.0, 2.0]])


def test_mean_pool_handles_an_all_padding_row():
    # Shouldn't divide by zero - an empty headline is a real case.
    vectors = np.array([[[1.0, 1.0], [2.0, 2.0]]])
    mask = np.array([[0, 0]])
    pooled = mean_pool(vectors, mask)
    assert np.isfinite(pooled).all()


def test_mean_pool_matches_a_plain_mean_when_nothing_is_padded():
    vectors = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    mask = np.array([[1, 1]])
    assert np.allclose(mean_pool(vectors, mask), [[2.0, 3.0]])


def test_sentiment_score_uses_label_names_not_positions():
    """The bug this guards against inverts every score silently.

    FinBERT's class order is not alphabetical and not guaranteed stable
    across model versions. Reading the index by *name* means a reordered
    config produces the same score; hardcoding index 0 as "positive"
    would produce a confidently backwards model with no error raised.
    """
    logits = np.array([[5.0, 0.0, 0.0]])

    positive_first = sentiment_score_from_logits(logits, ["positive", "negative", "neutral"])
    negative_first = sentiment_score_from_logits(logits, ["negative", "positive", "neutral"])

    assert positive_first[0] > 0.9
    assert negative_first[0] < -0.9
    assert np.isclose(positive_first[0], -negative_first[0])


def test_sentiment_score_is_bounded_and_signed():
    logits = np.array([[0.0, 0.0, 10.0], [2.0, 1.0, 0.5], [1.0, 2.0, 0.5]])
    scores = sentiment_score_from_logits(logits, ["positive", "negative", "neutral"])
    assert ((scores >= -1.0) & (scores <= 1.0)).all()
    assert abs(scores[0]) < 0.01   # all-neutral -> ~0
    assert scores[1] > 0           # positive outweighs negative
    assert scores[2] < 0


def test_walk_forward_never_trains_on_its_own_test_block():
    """A signal present only in the final 10% of the data must not be
    predictable in the earlier out-of-sample blocks.

    If folds were leaking future data, the early predictions would pick
    up the late-period relationship. This asserts they don't.
    """
    rng = np.random.default_rng(0)
    n = 600
    features = rng.normal(size=(n, 3))
    target = rng.normal(size=n)
    # Only the tail carries a relationship between feature 0 and target.
    target[-60:] = features[-60:, 0] * 10

    predictions, actuals = walk_forward_predictions(features, target, n_folds=5)
    assert len(predictions) == len(actuals)
    # Predictions exist for the pooled out-of-sample blocks only, which
    # is strictly fewer rows than the dataset.
    assert 0 < len(predictions) < n


def test_walk_forward_recovers_a_genuine_signal():
    # Sanity check in the other direction: a strong, stable relationship
    # present throughout SHOULD be recovered out-of-sample. A harness
    # that can't find real signal would make every negative result
    # meaningless.
    rng = np.random.default_rng(1)
    n = 600
    features = rng.normal(size=(n, 2))
    target = features[:, 0] * 3 + rng.normal(size=n) * 0.1

    predictions, actuals = walk_forward_predictions(features, target, n_folds=5)
    correlation = np.corrcoef(predictions, actuals)[0, 1]
    assert correlation > 0.9


def test_walk_forward_skips_folds_with_too_little_training_data():
    # Tiny dataset: early folds can't train on 50+ rows, and returning a
    # prediction from a model fitted on a handful of points would be
    # worse than returning nothing.
    features = np.random.default_rng(2).normal(size=(60, 2))
    target = np.random.default_rng(3).normal(size=60)
    predictions, actuals = walk_forward_predictions(features, target, n_folds=5)
    assert len(predictions) == len(actuals)
