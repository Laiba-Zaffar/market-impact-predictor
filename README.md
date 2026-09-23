# Market Impact Predictor

Given a financial news event, predict the direction, magnitude, and
calibrated confidence of the stock price move that follows it. Builds on
[fin-event-intel](../fin-event-intel)'s event/entity extraction work, but
needs its own historical dataset — a live feed only teaches a model what
happens *after* you start running it, never what happened before.

## Highlights

- **Found that three "no signal" results were partly measuring defects
  in the label, not the features.** The target was a raw close-to-close
  return, so ~29% of its variance was the index moving rather than news,
  which handed the model a free lunch: 55.15% accuracy available from
  upward drift alone. That *was* the majority-class collapse. The
  label is now a beta-adjusted, volatility-normalized abnormal return,
  estimated point-in-time; the base rate is 0.5075 and the free lunch is
  gone. ([details](#m8-the-label-was-the-problem))
- **Discovered the real sample size was ~21x smaller than the row
  count.** Every article about one ticker resolving to one reference
  close carries an identical label - 55,532 "examples" covered only
  2,630 distinct labels (2,002 once no-event articles are also
  dropped), with near-duplicates of most test labels sitting in the
  training set. Every significance estimate before this was ~4.6x
  (sqrt(21)) too confident.
- **Caught a costly wrong assumption about a third-party API before it
  compounded.** Assumed Alpha Vantage's multi-ticker query was an OR
  filter; it's actually an AND ("articles mentioning ALL listed tickers
  simultaneously"), which silently returned near-empty results and burned
  20 of a 25/day quota before the pattern was obvious enough to question.
  Redesigned around single-ticker calls once caught. ([details](#m1-lesson-verify-api-semantics-before-designing-around-them))
- **Every stage refuses to produce a misleading number on too little
  data**, rather than reporting a number that looks like a result but
  isn't one - verified on 15 examples (correctly refused to evaluate)
  and again at full volume, where it produced a real negative result
  instead of a flattering one. M8 later showed that result was itself
  measuring a defective label, which is the more interesting finding.
- **Time-based train/test split, not random** — a random split on
  time-series financial data lets the model implicitly train on
  information from the future relative to a test example, which makes
  a backtest look better than any live version of the model ever could.
- **The backtest ships with its own caveats printed alongside the
  numbers** — no transaction costs, hand-picked ticker universe,
  single historical period — rather than a clean-looking result someone
  has to know to distrust.
- **43 tests, all deterministic, none needing a trained model or live
  API** — including one that caught a real bug in its own mock (a test
  double returning a Python list where the real dependency returns a
  numpy array, which broke the code's `[:, 1]` slicing), and one that
  asserts lookahead leakage *cannot* happen: two price series identical
  up to the reference date and violently divergent after it must
  produce an identical beta.

## Architecture

```mermaid
flowchart LR
    AV[Alpha Vantage News\nSentiment API] -- daily cron,\nquota-aware --> Ingest[fetch_news_sentiment.py]
    YF[yfinance\n21 tickers + SPY] --> Prices[fetch_prices.py]
    Ingest --> DB[(SQLite)]
    Prices --> DB
    DB --> Filter[features/events.py\ndrop no-event articles]
    Filter --> Labels[features/labels.py\nprice alignment]
    DB --> Adjust[features/market_adjust.py\npoint-in-time beta + vol]
    Labels --> Dataset
    Adjust --> Dataset[features/build_dataset.py\nabnormal label, event-level dedup,\ntime-based split]
    Dataset --> Train[model/train_baseline.py]
    Train --> Models[(models/*.joblib)]
    Models --> Calibrate[model/calibrate.py]
    Models --> Backtest[eval/backtest.py]
```

Two independent data sources (news+sentiment, and raw prices) feed the
same SQLite database; `features/labels.py` is the join point that turns
"an article about ticker X at time T" into "what actually happened to
X's price afterward."

Three things sit between that join and the model, and each exists
because its absence produced a misleading result (see
[M8](#m8-the-label-was-the-problem)):

- `features/events.py` drops articles containing no event - 13F
  boilerplate, opinion listicles, post-hoc price explainers.
- `features/market_adjust.py` strips the market's share of the move,
  using a beta estimated only from data available at the reference
  close, and scales by trailing volatility.
- `build_dataset.aggregate_to_events()` collapses the many articles
  sharing one label into the single observation they actually are.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in ALPHA_VANTAGE_API_KEY
```

**Collect data** (news backfill is quota-limited to ~23 calls/day free
tier - runs incrementally, resumable, via a daily cron job):
```bash
python -m src.ingest.fetch_news_sentiment   # news + sentiment, quota-aware
python -m src.ingest.fetch_prices           # prices, no rate limit
```

**Run the modeling pipeline** (safe to re-run any time - picks up
whatever data exists):
```bash
python -m src.run_pipeline
```

## Testing

```bash
python -m pytest tests/ -v
```

43 tests, all fast (~1s total), all deterministic - no live API calls,
no trained model required. Same philosophy as Project 1: test the logic
you actually own directly (date/price arithmetic, the time-based split,
the quota tracker's persistence and reset behavior, the backtest's
position-sizing logic against a mocked model), and don't try to unit-test
a live model's output quality - that's what the manual verification
against real data (see below) and the eventual calibration/backtest
metrics are for.

## Roadmap

- [x] M0 — repo scaffold
- [~] M1 — historical news+sentiment ingestion (Alpha Vantage, quota-aware) — 11 of 21 tickers done (38,507 articles), rest still backfilling
- [x] M2a — historical price data (yfinance, 14mo × 21 tickers)
- [x] M2b — price alignment → forward-return labels
- [x] M3 — dataset construction (time-based split, no lookahead leakage)
- [x] M4 — baseline model — real result on 31,230 examples: doesn't beat the base rate yet (see below)
- [x] M5 — confidence calibration — real result: minimal improvement, consistent with M4's weak-signal finding
- [x] M6 — backtest evaluation — real result: underperforms buy-and-hold (35.07 vs 38.06)
- [x] M7 — portfolio polish (tests, README, this list)
- [x] M8 — **label correctness**: abnormal returns, event-level dedup, event filter — the three negative results above were partly measuring label defects, not features ([below](#m8-the-label-was-the-problem))
- [ ] M9 — re-point `train_improved` / `calibrate` / `backtest` at the event-level dataset (they still read the old raw-return columns)
- [ ] M10 — own the NLP: fine-tune an encoder on headlines instead of consuming a third party's sentiment float

## Engineering notes

The rest of this README is the build log - what was tried, what broke,
what got measured, and why. Kept chronological rather than rewritten as
if everything worked first try.

### M1 lesson: verify API semantics before designing around them

Alpha Vantage's `tickers` parameter looked like it should batch multiple
tickers per call (`tickers=AAPL,MSFT,...`) to conserve the 25-calls/day
quota. It doesn't work that way - per AV's own docs, it's an AND filter
("articles that simultaneously mention" every listed ticker), not an OR.
Batching 5 unrelated tickers together asks for articles mentioning all
5 at once, which almost never happens - the first backfill run burned
20 of that day's 25 calls returning near-zero results before this was
caught. Redesigned around one ticker per call, quarterly time windows
instead of monthly to keep the total call count reasonable (~84 calls
for a year across 21 tickers). Lesson kept for the interview, not just
here: verify an unfamiliar third-party API's actual semantics with one
cheap call before designing a whole strategy around an assumption.

Quota protection is two layers, not one: a local JSON counter tracks
calls used per UTC day, but every response is also checked for Alpha
Vantage's own rate-limit signal (a 200 OK with an "Information" or
"Note" field, not an HTTP error code) as a hard backstop - trusting only
local bookkeeping felt too fragile after already miscounting once.

The backfill runs via a real OS cron job (`crontab`, not a cloud-scheduled
task), since the script needs the project's local venv, `.env`, and
SQLite file - none of which a cloud sandbox would have access to.
Setting it up correctly took several tries: cron doesn't start in the
project directory, so a first attempt with relative paths would have
silently failed every day. Caught by testing the *exact* installed
crontab line from a cold `$HOME` shell before trusting it unattended,
after (embarrassingly) writing the same broken relative-path version
five times in a row via inline shell variables - switching to writing
the line to a file and reading it back is what actually broke that loop.

### M2b: market-hours cutoff for price alignment

An article published during market hours has its reaction reflected in
that same day's close; one published after close (or on a weekend/
holiday) can't be reacted to until the next session opens. Getting this
backwards would mean using a closing price that occurred *before* the
news broke as if it already reflected the market's response - a subtle
lookahead error, not just an off-by-one. The cutoff is a fixed ~20:00
UTC approximation of US market close, which doesn't precisely handle
the EST/EDT transition (off by up to an hour part of the year) - a
documented simplification, not a rigorous market-calendar implementation
(that would mean pulling in something like `pandas_market_calendars`).

### M4-M6, first run: too little data, and the pipeline said so

`scikit-learn` stalled four times installing on this machine (connection
established, then zero data movement for minutes) before finally
installing cleanly on a later attempt - real network flakiness that day,
not a code problem. Once it installed, `python -m src.run_pipeline` ran
M4 through M6 for the first time, on the 15 labeled examples that
existed at that point:

- **Direction classifier**: accuracy 0.667, but precision/recall/F1 all
  0.000 - the confusion matrix shows it predicted "down" for every
  single example. Not a bug: the model correctly found no real signal
  in 15 points and fell back to the majority class (10 of 15 were
  actually down).
- **Calibration (M5)** and **backtest (M6)** both correctly refused to
  run at all - 15 examples is below the 30-example minimum guard built
  into both, and each prints an explicit message saying so.

This confirmed the pipeline works exactly as designed - honest about
having too little data to say anything, rather than producing a
plausible-looking number regardless of sample size.

### M4-M6, the real run: 31,230 examples, and an honest negative result

The cron job set up to run the backfill daily never actually fired - the
laptop's boot log showed it powered on at 12:55pm, well after the 6am
scheduled time. Plain `cron` doesn't catch up on missed runs; it just
waits for the next scheduled time, which meant the "daily" backfill had
only run twice, both by hand during setup, not automatically overnight.

Running it manually once the day's quota had genuinely reset produced a
real surprise: every single API call returned the full 1,000-article cap.
Real per-ticker news volume is far higher than the ~150/quarter estimated
from one early test - one day's 23 calls (6 of 21 tickers) pulled in
19,192 articles and 62,511 ticker-sentiment pairs, versus the handful
that existed before.

That volume immediately exposed a real performance bug: `build_dataset()`
was calling `get_prices()` fresh for every row instead of once per
ticker, so 62k rows meant 62k separate database queries. It had been
fine at 15 rows and invisible until there was enough data to hurt -
caching each ticker's price series once and reusing it across its rows
took the build from "still running after several minutes" to 6.3 seconds.

With that fixed, `python -m src.run_pipeline` produced the first
real evaluation - 24,984 training examples, 6,246 held out for testing:

- **Direction classifier**: accuracy 0.565, recall 1.000, precision
  0.565 - the confusion matrix shows it predicted "up" for every single
  test example. Same majority-class collapse as the 15-example run, just
  the opposite direction (56.5% of this test period was actually "up") -
  with real volume behind it now, this is a genuine, not-yet-encouraging
  finding: three sentiment-only features don't carry enough linear
  signal to beat the base rate.
- **Calibration (M5)**: Brier score barely moved, 0.2460 → 0.2459.
  Consistent with M4's finding - calibration corrects a *miscalibrated*
  signal, and there's not much real signal here to correct.
- **Backtest (M6)**: strategy cumulative return 35.07 vs. buy-and-hold's
  38.06 on the same test examples. The strategy underperformed simply
  holding - trading on a weak, mostly-one-directional signal added noise
  without adding edge.

This is not the result I'd have picked to headline, and that's exactly
why it's the one worth keeping in the README rather than a cherry-picked
one. It's also a completely legitimate, expected place for a first
baseline to land: three off-the-shelf sentiment scores were never
guaranteed to contain a real trading signal, and now there's honest
evidence they mostly don't - not a guess, an actual measurement on 31k
examples with correct methodology behind it. That's precisely the
evidence needed to justify the next real investment (richer features,
starting with Project 1's own NER/event-classification output) instead
of shipping a baseline no one checked.

`reports/reliability_diagram.png` (from M5) plots predicted probability
against observed frequency for both the raw and calibrated model against
the perfect-calibration diagonal - added during M7 polish after noticing
`matplotlib` was installed but never actually used anywhere.

### First improvement pass: fixed a real failure mode, didn't fix accuracy

`src/model/train_improved.py` tests three changes against the baseline,
all using data already collected - feature scaling, `ticker` added as a
one-hot feature, class balancing, and a non-linear model
(`HistGradientBoosting`) - to see whether any of them turn the baseline's
weak result into a real one. Honest outcome, not the one I'd have picked
to headline:

| Config | Accuracy | F1 | What actually happened |
|---|---|---|---|
| Baseline (3 features, no scaling) | 0.565 | 0.722 | Predicted "up" for every single example - the accuracy number is just the base rate |
| + scaling + ticker | 0.554 | 0.664 | Real variation for the first time - both false negatives *and* true negatives are nonzero |
| + `class_weight="balanced"` | 0.457 | 0.495 | Worse than the base rate - forcing balance on a weak signal just adds noise |
| HistGradientBoosting instead | 0.545 | 0.657 | Statistically indistinguishable from plain logistic regression |

Raw accuracy technically *dropped* (0.565 → 0.554) - but the baseline's
0.565 came from a model that had stopped trying, predicting one class
regardless of input. The scaled+ticker version is the first one that's
actually discriminating between examples, which matters more than the
accuracy digit even though it's a less impressive-looking number.

Two things worth being able to say plainly, not just the numbers:
- **Class balancing is not automatically a good idea.** It's a common
  reflex fix for imbalanced data, and here it made results worse - with
  genuinely weak signal, forcing the classifier to split evenly just
  means it's guessing more symmetrically, not guessing better.
- **A stronger model did not fix a weak feature set.** Gradient boosting
  performed the same as logistic regression on the same 3 features. That's
  real evidence the bottleneck is the *information content* of the
  features, not the model's capacity to use them - which means the next
  real improvement is better features (Alpha Vantage's `topics` field is
  fetched and currently discarded; Project 1's own NER/event
  classification is unused here too), not a bigger model on the same
  three numbers.

Saved as `models/*_v2.joblib`, deliberately not overwriting the
production `direction_classifier.joblib` / `magnitude_regressor.joblib`
that `calibrate.py` and `backtest.py` load - this pipeline expects a
`ticker` column those don't pass yet, so wiring it in is a real next
step, not something to do silently as a side effect of running a
comparison script.

### Second improvement pass: real topic data, and a second honest negative result

The first pass showed the bottleneck was feature information, not model
choice - so the next lever tried was Alpha Vantage's `topics` field
(earnings, M&A, macro, etc. per article), which was being fetched in
every API response and thrown away. Added:

- An `article_topics` table and the fetch code to actually store it.
- `topic_features_for()` in `build_dataset.py` - a pure function pivoting
  each article's topics into a fixed 15-column feature vector (one per
  Alpha Vantage topic category, 0.0 where absent), tested in isolation.
- Those 15 columns wired into `train_improved.py`'s feature set.

That shipped before there was real (non-constant) topic data to test it
against. A further backfill (11 of 21 tickers now covered, 38,507
articles, 55,532 labeled examples - up from 31,230) produced 24,302
examples with real topic tags, enough to actually answer the question.

**The answer is no - and it's worse than "no improvement."** None of
the three trained configs beat a trivial "always predict the majority
class" baseline (0.564 accuracy / 0.721 F1) on this larger dataset:

| Config | Accuracy | F1 |
|---|---|---|
| Trivial majority-class baseline | 0.564 | 0.721 |
| Logistic + scaling + ticker + topics | 0.485 | 0.584 |
| + `class_weight="balanced"` | 0.459 | 0.510 |
| HistGradientBoosting + topics | 0.500 | 0.600 |

Splitting the test set by whether an example actually has real topic
data (vs. the old constant-zero rows) makes it clearer that topics
specifically aren't helping, not just diluted by old rows: the
HistGradientBoosting model scores 0.467 accuracy against a 0.558
trivial baseline on the topic-tagged subset, versus 0.528 against 0.571
on the subset without topic data - the model does relatively *worse*
exactly where the new feature is populated.

`train_improved.py` now prints the trivial-baseline comparison and
warns explicitly when the "best" trained config doesn't beat it, so
this can't get silently reported as progress again by picking the top
row of a table without a reference point.

**Reading this straight:** three feature-engineering attempts now
(ticker+scaling, class balancing, topic categories) on top of Alpha
Vantage's overall/relevance/ticker sentiment scores, and none of them
produce a model that beats guessing the majority class. That's
consistent evidence the sentiment scores AV provides just don't carry
enough signal for next-few-days direction on their own - the credible
next lever is different information entirely (Project 1's own
event/entity extraction, or price/volume-based features), not another
transformation of the same three numbers.

### M8: the label was the problem

Three feature-engineering attempts had all failed to beat a
majority-class baseline, and the conclusion drawn above was that the
sentiment scores carry no signal. Before spending more on features, the
next thing worth checking was whether the *target* was measuring what it
claimed to. It wasn't, in three separate ways - all of which inflate or
distort exactly the numbers reported in M4-M7.

**1. The label was a raw return, so it was mostly the stock market.**

`direction = 1 if target_close > reference_close` is a raw close-to-close
move. Measured on this corpus, ~29% of that variance is market-wide, and
on 43.7% of days more than 80% of the 21 tickers moved the same
direction - one index, not 21 independent news reactions. The
consequence is not subtle:

| | P(up) |
|---|---|
| raw return | 0.5515 |
| market-adjusted return | 0.5075 |

The baseline's "56.5% accuracy, predicts up for everything" was the model
correctly exploiting upward drift that had been baked into the target.
**The majority-class collapse was a property of the label, not evidence
about the features.** The fix is the standard event-study form: subtract
the stock's beta-weighted share of the market's move over the same
window, then divide by trailing volatility so a 1% move in a quiet name
and a 1% move in a volatile one aren't scored as the same size surprise.

Beta and volatility are estimated from a 120-day window that **ends at
the reference close** - never past it. Using a full-sample beta would
leak the future into the label, which is the same class of error the
time-based split already guards against, just hidden one level deeper.
`test_beta_ignores_data_after_the_reference_date` asserts this directly:
two series identical up to the reference and violently divergent after
it must produce an identical beta.

**2. The sample was ~27x smaller than the row count.**

Every article about AAPL resolving to the same reference close carries
the *identical* label. Each label cell was reused ~21 times: the 55,532
rows reported in M4-M7 covered only 2,630 distinct labels. (The current
event-level dataset has 2,002 rows - the further drop is the event
filter below, not deduplication.) Because duplicates straddle the
train/test boundary, near-twins of most test labels sat in the training
set - not classic lookahead leakage, but leakage, and every significance
estimate before this was ~4.6x (sqrt(21)) too confident.

Collapsing to one row per (ticker, reference_date) is the honest unit.
The article count survives as `n_articles`, which is real information -
a burst of coverage on one name in one day - that was previously
expressed only as duplicated rows, where no model could use it.

**3. A large share of the corpus contains no event.**

A model mapping "news about X" to "X's subsequent move" assumes the
article describes something that happened. Much of a retail feed
doesn't:

- **Ownership boilerplate** - 13F filings are public 45 days after
  quarter end, and content mills auto-generate one article per filer per
  holding ("GRIMES & Co WEALTH MANAGEMENT LLC Sells 3,223 Shares of
  Alphabet Inc."). MarketBeat alone is 18.8% of the corpus.
- **Opinion and listicles** - "Is Johnson & Johnson Still the Ultimate
  Safe Dividend Stock to Buy?" has no event and an arbitrary timestamp.
- **Post-hoc explainers** - "Why KLA Corporation (KLAC) Stock Is Down
  Today" is published *because* the stock already moved. That move is
  already inside the reference close, so these add noise to a forward
  label rather than signal.

63.6% of articles survive the filter. Writing the filter tests from
verbatim corpus titles rather than invented ones immediately earned its
keep: it caught a false positive where *"Enovix Shares Climb After
Company Names Former Apple AirPods Manufacturing Leader as COO"* was
being dropped as a post-hoc explainer despite reporting a genuine
corporate action. Price-move headlines are now dropped only when they
never state a cause.

**What this changed, and what it didn't.**

The honest answer on signal is unchanged - and now trustworthy:

| feature (event-level, n=2,002) | rank IC | t |
|---|---|---|
| `ticker_sentiment_score_mean` | +0.0117 | 0.52 |
| `overall_sentiment_score_mean` | +0.0357 | 1.60 |
| `n_articles` | +0.0157 | 0.70 |
| `topic_earnings_max` | +0.0420 | 1.88 |
| `topic_economy_macro_max` | +0.0471 | 2.11 |

That last row crosses the usual |t|>2 threshold, and is reported here
*because* it shouldn't be believed: seven features were tested, so ~0.35
false positives at p<0.05 are expected by chance, and one marginal hit
out of seven is exactly what noise looks like. It would need to hold
out-of-sample, on data it wasn't discovered in, before it counted as
anything.

The more useful finding is a power calculation. Detecting IC=0.03 at
t=2 needs n≈4,444; this dataset has 2,002. **"No signal" and "not enough
data to see a signal" are currently indistinguishable here** - which
means the honest next step is more history and better features, not
another model on the same 2,002 rows.

Accuracy was also the wrong metric throughout. An IC of 0.03 corresponds
to roughly 51.5% direction accuracy - a yardstick too coarse to resolve
the effect being looked for, which is part of why M4-M7 could only ever
return "no."

One performance note, since it's the third instance of the same lesson:
the abnormal-return computation depends only on
(ticker, reference_date, target_date), but ran once per *article* row,
each time fitting a 120-day OLS beta. Memoizing on that key took the
build back to 14 seconds. The price-series cache, the topics query, and
now this - same shape every time.

### Known limitations, stated plainly

- **Underpowered for the effect size being looked for.** 2,002
  event-level observations; detecting IC=0.03 at t=2 needs ~4,400. No
  conclusion drawn here separates "no signal" from "not enough data."
- **There is no NLP in this project yet.** Every feature is a float
  Alpha Vantage computed - `overall_sentiment_score`,
  `ticker_sentiment_score`, topic relevances. The modeling is
  scikit-learn on someone else's sentiment model's output, which caps
  how good it can get and makes the headline text itself unused. M10.
- **Small, hand-picked ticker universe** (21 large, liquid, currently
  healthy companies across sectors) - not an unbiased trading universe;
  selecting it at all is a mild form of hindsight bias.
- **The event filter is blunt on purpose** - title regexes plus a
  content-mill source blocklist, not a classifier. It drops real events
  from blocklisted publishers (an Insider Monkey article about a genuine
  product launch goes with the rest), and `filter_stats()` reports the
  drop rate so that cost stays visible rather than assumed.
- **Beta is estimated against SPY alone**, not a market + sector model.
  A sector shock still lands in the "abnormal" return for every name in
  that sector on the same day.
- **Daily price granularity only** - no intraday reaction captured,
  since free historical intraday data isn't available a year back.
  Meaningful for "does this news shift the multi-day trajectory," not
  for "how did the price move in the first five minutes."
- **Backtest has no transaction costs, slippage, or position sizing** -
  printed explicitly alongside every backtest run so the numbers are
  never read in isolation from that context.
- **Every quarterly news query hit Alpha Vantage's 1,000-article cap** -
  real volume per ticker is higher than that, and the fetch uses
  `sort=EARLIEST`, so each quarter's *later* articles for high-volume
  tickers are systematically missing, not randomly. A truer fetch would
  detect the cap being hit and split that window into smaller ones.
- **Confidence looks roughly reasonable but hasn't been rigorously
  checked** - the Brier score barely changed with calibration (0.2460 →
  0.2459), consistent with a model that doesn't have much real signal to
  calibrate in the first place. That measurement also predates M8, so it
  was taken against a 55/45 base rate that left little to correct;
  it needs re-running against the abnormal-return label before it means
  anything either way.
- **M4-M7's reported numbers are superseded, not deleted.** They are
  kept above as the chronological record of how the label defects were
  found, but every accuracy/Brier/backtest figure in those sections was
  computed on raw returns over duplicated rows. `train_improved.py`,
  `calibrate.py` and `backtest.py` still read those old columns and have
  not yet been re-pointed at the event-level dataset (M9), so running
  the pipeline today reproduces the old numbers, not the M8 ones.
- **The daily cron backfill silently never fired** - scheduled for 6am,
  but the laptop was off/asleep then, and plain `cron` doesn't catch up
  missed runs. Found via the boot log, not a smoking gun - worth moving
  to a trigger that survives a personal laptop's actual usage pattern
  (e.g. a systemd timer with `Persistent=true`, which runs a missed job
  as soon as the machine wakes) rather than assuming a laptop behaves
  like an always-on server.
