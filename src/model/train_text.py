"""M10 - does owning the NLP help?

Three rounds of feature engineering (scaling, ticker identity, topic
categories) all failed to beat a trivial baseline, and the conclusion
was that the bottleneck is the information content of Alpha Vantage's
three sentiment numbers rather than the model's capacity to use them.
That conclusion was never actually tested, because every feature tried
was another transformation of those same three numbers. The headline
text - the thing that actually contains the news - was stored from the
start and never read.

This is the test. A ladder of text representations, each evaluated with
an identical protocol, so the only thing that differs between rungs is
how much of the headline survives into the features:

  rung 0  Alpha Vantage's sentiment score         (someone else's model)
  rung 1  TF-IDF over the headlines + linear probe (our own text
          processing, no pretrained model at all)
  rung 2  FinBERT sentiment score computed here    (own model, same
          shape as rung 0 - a controlled comparison)
  rung 3  384-d sentence embedding + linear probe  (own model, full text)

Rung 1 is not a placeholder. With ~2,000 independent labels, a sparse
bag-of-words model is a genuinely appropriate choice - often a stronger
baseline on short financial headlines than a dense embedding, because
it can key on exact tokens ("beats", "downgrade", "recall") without
needing to learn them from scarce labels. If the pretrained rungs can't
beat it, that is a real finding rather than a missing experiment.

Rungs 2-3 require torch/transformers and are skipped automatically when
those aren't installed, so the ablation always produces whatever it can
rather than failing entirely.

**Evaluation is walk-forward, not a single split.** A single 80/20 split
leaves ~400 test rows, and at n=400 you need an IC around 0.10 before a
t-statistic clears 2 - so a single split literally cannot distinguish
these rungs at the effect sizes in play. Walk-forward refits at each
step and pools the out-of-sample predictions, yielding ~1,600 honest
test predictions instead of 400. Every fold still trains only on data
preceding its test window, so the no-lookahead property holds.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build_dataset import aggregate_to_events, build_dataset

TARGET = "standardized_abnormal_return"

# Expanding-window folds. Five is a compromise: more folds means more
# pooled out-of-sample predictions, but each early fold trains on very
# little data and contributes noisier predictions.
N_FOLDS = 5

# Ridge penalties searched inside each training fold only. Hundreds of
# text features against ~1,600 rows needs real regularization - an
# unpenalized fit would interpolate the training data and generalize to
# nothing. The grid reaches very high alpha on purpose: if the honest
# answer is "shrink this to almost zero," the search should say so.
ALPHA_GRID = np.logspace(0, 6, 25)

TFIDF_COMPONENTS = 100
EMBEDDING_COMPONENTS = 32


def transformers_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def walk_forward_predictions(
    features: np.ndarray, target: np.ndarray, n_folds: int = N_FOLDS
) -> tuple[np.ndarray, np.ndarray]:
    """Expanding-window out-of-sample predictions.

    Fold k trains on everything before block k and predicts block k. No
    fold ever sees its own test window or anything after it, so pooled
    predictions are all genuinely out-of-sample - the walk-forward
    analogue of the time-based split, giving many more test points
    without weakening the no-lookahead guarantee.

    Returns (predictions, actuals) over the pooled test blocks.
    """
    n = len(features)
    block = n // (n_folds + 1)
    predictions, actuals = [], []

    for fold in range(1, n_folds + 1):
        train_end = block * fold
        test_end = block * (fold + 1) if fold < n_folds else n

        train_x, train_y = features[:train_end], target[:train_end]
        test_x, test_y = features[train_end:test_end], target[train_end:test_end]
        if len(test_x) == 0 or len(train_x) < 50:
            continue

        # Alpha is selected by RidgeCV *inside* the training block, so
        # the penalty is never chosen using data the fold is about to be
        # scored on. Selecting it once on the full dataset would be a
        # quiet form of leakage.
        model = Pipeline(
            [("scale", StandardScaler()), ("ridge", RidgeCV(alphas=ALPHA_GRID))]
        )
        model.fit(train_x, train_y)
        predictions.append(model.predict(test_x))
        actuals.append(test_y)

    return np.concatenate(predictions), np.concatenate(actuals)


def evaluate(name: str, features: np.ndarray, target: np.ndarray) -> dict:
    """Identical protocol for every rung - same folds, same metric."""
    if features.ndim == 1:
        features = features.reshape(-1, 1)

    predictions, actuals = walk_forward_predictions(features, target)
    ic, _ = spearmanr(predictions, actuals)
    if not np.isfinite(ic):
        # A rung whose predictions are constant (everything shrunk to
        # the mean) has no rank correlation to report. That's a real
        # outcome - say so rather than printing nan.
        ic = 0.0
    n = len(predictions)
    t_stat = ic * np.sqrt(max(n - 2, 1) / max(1 - ic**2, 1e-12))

    # Direction accuracy alongside IC, against the actual base rate
    # rather than against 50% - the base rate is the only bar that means
    # anything, as M4-M7 demonstrated the hard way.
    actually_up = actuals > 0
    accuracy = ((predictions > 0) == actually_up).mean()
    base_rate = max(actually_up.mean(), 1 - actually_up.mean())

    return {
        "name": name,
        "n_features": features.shape[1],
        "n_oos": n,
        "ic": ic,
        "t": t_stat,
        "accuracy": accuracy,
        "base_rate": base_rate,
    }


def build_event_text() -> pd.DataFrame:
    """Event-level rows, each carrying the day's headlines as one string.

    Labels are per (ticker, date) but headlines are per article, so an
    event's headlines are concatenated into a single document. For
    bag-of-words that is the natural pooling: the union of the day's
    vocabulary, with repeated terms weighted up - which is what you
    want, since a word appearing across several of the day's stories is
    more likely to describe the actual event.
    """
    articles = build_dataset()
    if articles.empty:
        return articles

    events = aggregate_to_events(articles)
    documents = (
        articles.groupby(["ticker", "reference_date"])["title"]
        .apply(lambda titles: " . ".join(t for t in titles if isinstance(t, str)))
        .reset_index()
        .rename(columns={"title": "headlines"})
    )
    merged = events.merge(documents, on=["ticker", "reference_date"], how="inner")
    print(f"Event-level rows with headline text: {len(merged)}")
    print(f"  mean articles per event: {merged['n_articles'].mean():.1f}")
    return merged


def tfidf_features(documents: list[str]) -> np.ndarray:
    """TF-IDF over the headlines, reduced to a dense matrix.

    Both steps are unsupervised - they never see the target - so fitting
    them on the full corpus is the same trade-off already accepted for
    PCA below. A stricter protocol would refit the vectorizer inside
    every fold; that would be more correct, and the cost is that the
    vocabulary changes between folds, which makes the rungs harder to
    compare. Flagged here rather than glossed over.
    """
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=3,
        stop_words="english",
        sublinear_tf=True,
    )
    sparse = vectorizer.fit_transform(documents)
    print(f"  TF-IDF: {sparse.shape[1]} terms over {sparse.shape[0]} documents")
    reduced = TruncatedSVD(n_components=TFIDF_COMPONENTS, random_state=0).fit_transform(sparse)
    return reduced


def dimensionality_sweep(documents: list[str], target: np.ndarray) -> None:
    """Score the TF-IDF rung across many SVD widths, and print all of it.

    This exists to make a specific mistake impossible: running one
    dimensionality, getting a result, and reporting it. The sweep's
    shape is the actual finding. If the text carried signal, IC would
    move smoothly with the number of components and hold a stable sign.
    If it were overfitting, IC would degrade monotonically as components
    grew. Noise looks like neither - it wanders and flips sign, and
    somewhere in a long enough sweep it crosses |t| > 2 by luck.

    Printing every row means the one that crosses can't be quietly
    reported as the result. Same discipline as the topic-feature hit in
    M8: seven tests, one marginal winner, no finding.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), min_df=3,
        stop_words="english", sublinear_tf=True,
    )
    sparse = vectorizer.fit_transform(documents)

    print("\n" + "=" * 80)
    print("TF-IDF DIMENSIONALITY SWEEP - read the shape, not the best row")
    print("=" * 80)
    print(f"{'SVD components':<18}{'IC':>9}{'t':>8}{'acc':>8}{'base':>8}")
    print("-" * 80)

    ics = []
    crossings = 0
    for k in [2, 5, 10, 20, 50, 100, 200]:
        reduced = TruncatedSVD(n_components=k, random_state=0).fit_transform(sparse)
        result = evaluate(f"svd-{k}", reduced, target)
        ics.append(result["ic"])
        if abs(result["t"]) > 2:
            crossings += 1
        print(
            f"{k:<18}{result['ic']:>+9.4f}{result['t']:>+8.2f}"
            f"{result['accuracy']:>8.3f}{result['base_rate']:>8.3f}"
        )

    sign_flips = sum(1 for a, b in zip(ics, ics[1:]) if a * b < 0)
    print("-" * 80)
    print(f"sign changes across the sweep: {sign_flips} of {len(ics) - 1} steps")
    print(f"rows crossing |t| > 2: {crossings} of {len(ics)}")
    if sign_flips >= 2:
        print(
            "\nIC flips sign repeatedly and doesn't trend with dimensionality.\n"
            "That is the signature of noise - not of signal (which would move\n"
            "smoothly) and not of overfitting (which would degrade monotonically).\n"
            "No single row here is selectable as 'the' result: choosing the best\n"
            "one after seeing the target is exactly the multiple-comparisons\n"
            "error, and with 7 rows tested, one crossing |t|>2 is expected by chance."
        )


def transformer_features(events: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    """FinBERT sentiment and sentence embeddings, if torch is installed."""
    from src.nlp.encode import EMBEDDING_MODEL, SENTIMENT_MODEL, load_or_encode

    articles = build_dataset()
    article_ids = articles["article_id"].tolist()
    headlines = articles["title"].fillna("").tolist()

    print("\nFinBERT sentiment:")
    finbert = load_or_encode(article_ids, headlines, SENTIMENT_MODEL)
    print("\nSentence embeddings:")
    embeddings = load_or_encode(article_ids, headlines, EMBEDDING_MODEL)

    articles = articles[articles["article_id"].isin(embeddings)].copy()
    articles["finbert_score"] = [float(finbert[i]) for i in articles["article_id"]]

    dim = len(next(iter(embeddings.values())))
    columns = [f"emb_{i}" for i in range(dim)]
    frame = pd.DataFrame(
        np.stack([embeddings[i] for i in articles["article_id"]]),
        columns=columns,
        index=articles.index,
    )
    text = pd.concat([articles[["ticker", "reference_date", "finbert_score"]], frame], axis=1)
    pooled = text.groupby(["ticker", "reference_date"], as_index=False).mean()

    # Pooling is not obviously right. An event averages ~13 headlines,
    # and the mean of 13 vectors drifts toward the corpus centroid, so a
    # null result on pooled embeddings can't distinguish "the
    # representation is uninformative" from "the averaging threw the
    # information away." The single highest-relevance headline is
    # carried alongside to separate those.
    top = articles.loc[articles.groupby(["ticker", "reference_date"])["relevance_score"].idxmax()]
    top_frame = pd.concat(
        [
            top[["ticker", "reference_date"]].reset_index(drop=True),
            pd.DataFrame(
                np.stack([embeddings[i] for i in top["article_id"]]),
                columns=[f"top_{c}" for c in columns],
            ),
        ],
        axis=1,
    )

    joined = events.merge(pooled, on=["ticker", "reference_date"], how="inner")
    joined = joined.merge(top_frame, on=["ticker", "reference_date"], how="inner")

    pooled_embeddings = joined[columns].values
    top_embeddings = joined[[f"top_{c}" for c in columns]].values
    return [
        ("rung 2  FinBERT sentiment (ours)", joined[["finbert_score"]].values),
        ("rung 2b FinBERT + AV sentiment", joined[["finbert_score", "ticker_sentiment_score_mean"]].values),
        ("rung 3  embeddings, mean-pooled", pooled_embeddings),
        ("rung 3b embeddings, PCA-32", PCA(n_components=EMBEDDING_COMPONENTS, random_state=0).fit_transform(pooled_embeddings)),
        ("rung 3c embedding, top article", top_embeddings),
        ("rung 3d top article, PCA-32", PCA(n_components=EMBEDDING_COMPONENTS, random_state=0).fit_transform(top_embeddings)),
    ]


def run() -> None:
    events = build_event_text()
    if events.empty:
        print("No data.")
        return

    events = events.sort_values("time_published").reset_index(drop=True)
    target = events[TARGET].values

    print("\nBuilding TF-IDF features:")
    tfidf = tfidf_features(events["headlines"].tolist())

    rungs = [
        ("rung 0  AV ticker sentiment (theirs)", events[["ticker_sentiment_score_mean"]].values),
        ("rung 0b AV both sentiment scores", events[["ticker_sentiment_score_mean", "overall_sentiment_score_mean"]].values),
        ("rung 1  TF-IDF headlines, SVD-100", tfidf),
    ]

    if transformers_available():
        rungs.extend(transformer_features(events))
    else:
        print("\n[torch/transformers not installed - rungs 2-3 skipped]")

    print("\n" + "=" * 80)
    print(f"M10 LADDER - walk-forward out-of-sample, target = {TARGET}")
    print("=" * 80)
    print(f"{'representation':<38}{'dims':>5}{'IC':>9}{'t':>7}{'acc':>8}{'base':>8}")
    print("-" * 80)

    results = []
    for name, features in rungs:
        result = evaluate(name, features, target)
        results.append(result)
        print(
            f"{result['name']:<38}{result['n_features']:>5}{result['ic']:>+9.4f}"
            f"{result['t']:>+7.2f}{result['accuracy']:>8.3f}{result['base_rate']:>8.3f}"
        )

    print("-" * 80)
    print(f"pooled out-of-sample predictions per rung: {results[0]['n_oos']}")

    best = max(results, key=lambda r: abs(r["t"]))
    print(f"\nStrongest by |t|: {best['name']} (IC={best['ic']:+.4f}, t={best['t']:+.2f})")
    if abs(best["t"]) < 2:
        print(
            "\nWARNING: no representation reaches |t| > 2. None of these are\n"
            "distinguishable from noise at this sample size - do not report any\n"
            "of them as a working signal."
        )
    if best["accuracy"] <= best["base_rate"]:
        print(
            f"WARNING: the strongest rung's direction accuracy ({best['accuracy']:.3f}) "
            f"does not beat the base rate ({best['base_rate']:.3f})."
        )

    dimensionality_sweep(events["headlines"].fillna("").tolist(), target)


if __name__ == "__main__":
    run()
