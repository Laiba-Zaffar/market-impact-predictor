"""Turning headlines into numbers this project actually owns.

Until now every feature here was computed by Alpha Vantage - their
sentiment score, their topic tags. The headline text itself was stored
and never read. That caps the project at "how good is someone else's
sentiment model," and it's the reason three rounds of feature
engineering had nothing new to work with: they were all transformations
of the same three numbers.

Two representations are produced, deliberately at different levels of
compression:

  - `finbert_score` - one number per headline, P(positive) - P(negative)
    from a BERT fine-tuned on financial text. Same *shape* as Alpha
    Vantage's score, computed here instead. That makes it a controlled
    comparison: if own-sentiment beats theirs, the sentiment model was
    the bottleneck; if it doesn't, the bottleneck is elsewhere.

  - `embedding` - a 384-dimensional vector per headline from a sentence
    encoder, with nothing thrown away. This is the representation a
    single sentiment number is a lossy summary of.

Why frozen encoders and a linear probe rather than fine-tuning the
encoder end-to-end: after event-level deduplication this dataset has
~2,000 independent labels. Fine-tuning 110M parameters against 2,000
labels overfits - the model memorizes rather than learns, and the test
score measures nothing. Probing frozen features is the standard
sample-efficient choice at this scale, and it keeps the comparison
above honest, since all three rungs then differ only in the
representation and not in how hard the model was allowed to fit.

Encoding is the slow step (CPU-only here), so results are cached to
disk keyed by article id. Re-running is cheap; only new articles are
encoded.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "encodings"

SENTIMENT_MODEL = "ProsusAI/finbert"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Headlines are short. 64 tokens covers essentially all of them, and
# padding everything to 512 would waste most of the compute budget on
# padding on a 4-core CPU.
MAX_TOKENS = 64
BATCH_SIZE = 32


def mean_pool(token_vectors: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Average a headline's token vectors into one vector, ignoring
    padding.

    Pure function over arrays so the pooling arithmetic is testable
    without loading a model. Weighting by the attention mask matters:
    averaging over padded positions would drag every short headline's
    vector toward whatever the pad token encodes, making short and long
    headlines artificially different.
    """
    mask = attention_mask[..., None].astype(np.float32)
    summed = (token_vectors * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    return summed / counts


def sentiment_score_from_logits(logits: np.ndarray, label_order: list[str]) -> np.ndarray:
    """P(positive) - P(negative), collapsing a 3-class financial
    sentiment head into one signed number.

    Deliberately the same shape as Alpha Vantage's score so the two are
    directly comparable. `label_order` is read from the model's own
    config rather than assumed - FinBERT's class order is not
    alphabetical, and hardcoding the wrong index would silently invert
    the sign of every score, which is the kind of bug that produces a
    confidently backwards model rather than an error.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)

    lowered = [label.lower() for label in label_order]
    positive_index = lowered.index("positive")
    negative_index = lowered.index("negative")
    return probabilities[:, positive_index] - probabilities[:, negative_index]


def _load(model_name: str):
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_name == SENTIMENT_MODEL:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def encode_headlines(
    headlines: list[str],
    model_name: str,
    progress_every: int = 2000,
) -> np.ndarray:
    """Run one encoder over a list of headlines.

    Returns a (n,) array of sentiment scores for the sentiment model, or
    an (n, dim) array of embeddings for the embedding model.
    """
    import torch

    tokenizer, model = _load(model_name)
    is_sentiment = model_name == SENTIMENT_MODEL
    label_order = None
    if is_sentiment:
        label_order = [model.config.id2label[i] for i in range(model.config.num_labels)]
        print(f"  {model_name} label order: {label_order}")

    outputs = []
    for start in range(0, len(headlines), BATCH_SIZE):
        batch = headlines[start : start + BATCH_SIZE]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt"
        )
        with torch.no_grad():
            result = model(**encoded)

        if is_sentiment:
            outputs.append(
                sentiment_score_from_logits(result.logits.numpy(), label_order)
            )
        else:
            outputs.append(
                mean_pool(
                    result.last_hidden_state.numpy(),
                    encoded["attention_mask"].numpy(),
                )
            )

        if progress_every and start and start % progress_every < BATCH_SIZE:
            print(f"  {start}/{len(headlines)} headlines encoded")

    return np.concatenate(outputs, axis=0)


def cache_path(model_name: str) -> Path:
    return CACHE_DIR / f"{model_name.replace('/', '__')}.npz"


def load_or_encode(article_ids: list[int], headlines: list[str], model_name: str) -> dict[int, np.ndarray]:
    """Encode only what isn't cached yet, then return every requested id.

    CPU encoding of tens of thousands of headlines takes minutes, and
    this gets called from more than one place - recomputing it each time
    would make the whole ablation painful to iterate on, which in
    practice means it stops getting re-run.
    """
    path = cache_path(model_name)
    cached: dict[int, np.ndarray] = {}
    if path.exists():
        with np.load(path) as stored:
            ids = stored["article_ids"]
            values = stored["values"]
        cached = {int(i): v for i, v in zip(ids, values)}
        print(f"  loaded {len(cached)} cached encodings from {path.name}")

    missing = [(i, h) for i, h in zip(article_ids, headlines) if i not in cached]
    if missing:
        print(f"  encoding {len(missing)} new headlines with {model_name} (CPU)...")
        new_values = encode_headlines([h for _, h in missing], model_name)
        for (article_id, _), value in zip(missing, new_values):
            cached[article_id] = value

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        all_ids = np.array(sorted(cached))
        np.savez_compressed(
            path,
            article_ids=all_ids,
            values=np.stack([cached[int(i)] for i in all_ids]),
        )
        print(f"  cached {len(cached)} encodings to {path.name}")

    return {i: cached[i] for i in article_ids if i in cached}
