"""Filtering news down to articles that plausibly contain an event.

A model that maps "news about ticker X" to "X's subsequent move" assumes
the article describes something that happened. Much of a retail news
feed doesn't. Three distinct failure modes, all present in this
project's corpus:

  1. **Ownership boilerplate.** 13F filings are public 45 days after the
     quarter ends, and content mills auto-generate one article per
     filer per holding: "GRIMES & Co WEALTH MANAGEMENT LLC Sells 3,223
     Shares of Alphabet Inc." Stale by construction, and there are
     thousands of them - MarketBeat alone is 18.8% of the corpus.

  2. **Opinion and listicles.** "Is Johnson & Johnson Still the Ultimate
     Safe Dividend Stock to Buy?" contains no event at all, just a
     writer's view. The publish timestamp is arbitrary with respect to
     the price.

  3. **Post-hoc explainers.** "Why KLA Corporation (KLAC) Stock Is Down
     Today" is published *because* the stock already moved. Training on
     these teaches the model to explain the past, not predict the
     future - and because the move is already in the reference close,
     they add noise to the forward label rather than signal.

Filtering on title patterns and known content-mill sources is a blunt
instrument, and deliberately so: the point is to remove the clearly
event-free bulk cheaply, not to build a classifier for article quality.
`filter_stats()` reports what was dropped so the cost is visible rather
than assumed - if this ever starts removing real events, that shows up
as a drop rate that doesn't match the spot check.
"""
import re

# Publishers whose output is predominantly auto-generated from filings
# and screeners. Judged by reading actual titles from each, not by
# reputation - several are legitimate outlets whose *syndicated feed*
# here is overwhelmingly boilerplate.
CONTENT_MILL_SOURCES = {
    "MarketBeat",
    "Defense World",
    "ETF Daily News",
    "Insider Monkey",
    "Simply Wall Street",
    "AD HOC NEWS",
}

# Case-insensitive title patterns. Each is anchored to phrasing that is
# characteristic of a whole genre, not to any one publisher's template.
NO_EVENT_PATTERNS = [
    # 13F / institutional ownership boilerplate
    r"\b(?:buys|sells|acquires|trims|boosts|lowers|raises|reduces|grows|takes)\b.{0,40}\b(?:shares|stake|position|holdings)\b",
    r"\b(?:shares|stake|position|holdings)\s+(?:in|of)\b.{0,40}\b(?:bought|sold|acquired|purchased)\b",
    r"\bhas\s+\$[\d.,]+\s+(?:million|billion)?\s*(?:stake|position|holdings)\b",
    r"\b(?:stake|position|holdings)\s+(?:in|of)\b.{0,30}\b(?:increased|decreased|raised|lowered)\b",
    # Analyst note churn - auto-generated per rating change, and the
    # genuinely market-moving ones are a small minority of the volume.
    r"\bprice\s+target\b",
    r"\b(?:reiterates?|maintains?|reaffirms?)\b.{0,30}\brating\b",
    r"\bcoverage\s+(?:initiated|started)\b",
    # Opinion, listicles, and speculation - no event, arbitrary timestamp
    r"^\s*(?:is|are|should|why|what|how|where|when|can|will|do|does)\b.*\?\s*$",
    r"\b(?:should\s+you|is\s+it\s+time\s+to|worth\s+buying|a\s+buy\s+now|better\s+buy)\b",
    r"\b(?:\d+|three|four|five|seven|ten)\s+(?:top|best|great|cheap|undervalued)?\s*stocks?\b",
    r"\bstocks?\s+to\s+(?:buy|watch|avoid|consider)\b",
    r"\b(?:here'?s\s+why|what\s+to\s+know|things\s+to\s+know|what\s+to\s+expect)\b",
    # Perpetual quote/screener pages - a "publish time" that has nothing
    # to do with any event, since the page is regenerated continuously.
    r"\b(?:live\s+price|stock\s+price,?\s+chart|quote,?\s+chart)\b",
    # Recurring aggregate roundups - many tickers, no single event
    r"\b(?:market|stock|sector|pharma|tech)\s+(?:roundup|wrap|recap|movers|highlights)\b",
    r"\b(?:midday|premarket|after-hours|morning|closing)\s+(?:movers|report|update|wrap|bell)\b",
]

# Headlines about a price move. These need more care than the patterns
# above, because two very different articles share the same phrasing:
#
#   "Why KLA Corporation (KLAC) Stock Is Down Today"       <- no event
#   "Enovix Shares Climb After Company Names ... as COO"   <- real event
#
# The first is pure post-hoc narration; the move is already in the
# reference close and there's nothing else in the article. The second
# reports an actual corporate action and happens to lead with the
# reaction. Dropping both costs real events - this was caught by a
# filter test built from verbatim corpus titles rather than invented
# ones, which is the whole reason to write them that way.
PRICE_MOVE_PATTERNS = [
    r"\b(?:why|how)\b.{0,60}\bstock\b.{0,20}\b(?:is|was|are|were)\b.{0,20}\b(?:up|down|falling|rising|sinking|soaring|plunging|jumping|climbing|moving|slipping|tumbling)\b",
    r"\bstock\s+(?:is|was)\s+(?:up|down|falling|rising|soaring|plunging|sinking|tumbling)\b.{0,20}\btoday\b",
    r"\b(?:shares?|stock)\s+(?:jumps?|jumped|plunges?|plunged|soars?|soared|sinks?|sank|tumbles?|tumbled|slides?|slid|surges?|surged|climbs?|climbed|slips?|slipped|rallies|rallied)\b",
]

# A stated cause, with enough text after it to be an actual cause rather
# than a trailing fragment. "Shares climb after earnings beat" keeps its
# event; "Shares climb today" doesn't have one.
CAUSAL_CONNECTOR = r"\b(?:after|following|as|on)\b\s+\S+(?:\s+\S+){2,}"

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in NO_EVENT_PATTERNS]
_COMPILED_MOVE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in PRICE_MOVE_PATTERNS]
_COMPILED_CAUSAL = re.compile(CAUSAL_CONNECTOR, re.IGNORECASE)


def is_event_worthy(title: str, source: str | None = None) -> bool:
    """True when an article plausibly reports something that happened.

    Pure function over (title, source) so the whole filter is testable
    without a database or a network call - the same reason the label
    arithmetic was split out into its own pure function.
    """
    if not title or not title.strip():
        return False
    if source is not None and source in CONTENT_MILL_SOURCES:
        return False
    if any(pattern.search(title) for pattern in _COMPILED_PATTERNS):
        return False
    if any(pattern.search(title) for pattern in _COMPILED_MOVE_PATTERNS):
        # A price-move headline is only event-free if it never says what
        # caused the move.
        return bool(_COMPILED_CAUSAL.search(title))
    return True


def filter_stats(articles: list[tuple[str, str | None]]) -> dict:
    """Drop-rate breakdown over (title, source) pairs.

    Reported rather than assumed: a filter that silently removed 90% of
    the corpus, or 2%, would both be quietly wrong, and neither shows up
    in a model metric until much later.
    """
    total = len(articles)
    by_source = sum(1 for title, source in articles if source in CONTENT_MILL_SOURCES)
    kept = sum(1 for title, source in articles if is_event_worthy(title, source))
    return {
        "total": total,
        "kept": kept,
        "dropped": total - kept,
        "dropped_by_source": by_source,
        "dropped_by_pattern": (total - kept) - by_source,
        "keep_rate": kept / total if total else 0.0,
    }
