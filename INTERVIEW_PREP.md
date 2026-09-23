# Market Impact Predictor — Interview Preparation

A working document for explaining this project out loud. Every answer is
grounded in something the repo actually does or measures — nothing here
claims a result the code can't reproduce.

**How to use it:** read the Glossary first if the finance vocabulary is
new. Then work through Part 2 (the story) until you can tell it without
notes. Parts 3–6 are the follow-up questions. Part 7 is the ones that
could sink you — rehearse those hardest.

---

## Part 1 — Glossary, in plain English

Read this once. The rest of the document assumes these words.

### The basics

**Ticker** — a company's short code on the stock market. `AAPL` is Apple.

**Closing price** — the price of one share at the moment the market shuts
for the day (4pm New York time). Markets aren't open 24/7, which turns
out to matter a lot in this project.

**Return** — how much the price changed, as a percentage. If a stock goes
from $100 to $103, the return is +3%.

**Horizon** — how far ahead you're predicting. This project uses **3
trading days**. "Trading days" means market days: Friday + 3 trading days
is the following Wednesday, because weekends don't count.

### The market itself

**The market / the index** — all stocks taken together. When people say
"the market was up today," they mean stocks in general rose.

**SPY** — a fund that tracks the 500 largest US companies. It's the
standard stand-in for "the market as a whole." This project uses it as
the benchmark.

**Raw return** — how much a stock moved, full stop. Includes everything:
company news, market mood, interest rates, panic, all of it.

**Abnormal return** — how much a stock moved *beyond what the market
explains*. This is the single most important concept in the project.

> **The analogy to use in an interview:** Suppose a student scores 80%
> on an exam. Is that good? You can't tell until you know the class
> average. If the class averaged 78%, the student did roughly normally.
> If the class averaged 50%, that 80% is genuinely impressive.
>
> Raw return is the raw score. Abnormal return is the score *relative to
> the class average*. A stock rising 2% on a day the whole market rose
> 2% tells you nothing about that company.

**Beta** — how strongly one stock tends to follow the market. Beta 1.0
means it moves roughly in step. Beta 2.0 means it swings twice as hard.
Beta 0.5 means it's calmer than the market. In the exam analogy, beta
says how sensitive *this particular student* is to an easy or hard exam.

The average beta across this project's 21 stocks is **0.78** — they're
slightly calmer than the market overall, which makes sense for large,
established companies.

**Volatility** — how jumpy a stock normally is. A utility company might
drift 0.5% on a typical day; a fast-growing tech company might swing 4%.

**Volatility normalization** — dividing the move by that stock's normal
jumpiness, so you're measuring *surprise*, not raw size. A 1% move in a
sleepy stock is a bigger surprise than a 1% move in a wild one. Same
reason you'd judge "ran 100m in 13 seconds" differently for a sprinter
than for someone who never runs.

**Market-neutral** — a strategy whose profit doesn't depend on the market
going up. Once you predict abnormal returns instead of raw ones, your
model is automatically market-neutral.

### Measuring whether a model works

**Base rate** — what you'd get by always guessing the most common answer,
with no skill at all. If stocks rose on 55% of days, always predicting
"up" scores 55%. **This is the number any real model must beat.**

**Majority-class collapse** — when a model gives up on the input and just
predicts the most common answer every single time. It looks like it's
scoring well; it's actually doing nothing. This project hit exactly that.

**Lookahead bias (or leakage)** — accidentally letting the model use
information that wouldn't have existed yet at the moment of prediction.

> **The analogy:** studying for an exam using an answer key that was
> written *after* the exam. You'll ace practice and fail the real thing.
>
> This is the single most serious methodological error in financial ML,
> and it's the thing interviewers probe hardest, because it makes broken
> models look brilliant right up until they lose money.

**Point-in-time** — the discipline of computing everything using *only*
what was knowable at that moment. The cure for lookahead bias.

**Time-based split** — training on older data and testing on newer data.
The opposite is a random split, which would let the model learn from
next month while being tested on last month — leakage.

**Information Coefficient (IC)** — the standard finance measure of how
well predictions line up with what actually happened. Technically a rank
correlation, ranging −1 to +1.

Scale, because the numbers look tiny to outsiders:
- **IC = 0** — no predictive power at all
- **IC = 0.02–0.05** — weak, but real professional signals live here
- **IC = 0.10** — very strong
- **IC = 0.50** — you have made an error somewhere; go find it

**t-statistic** — how confident you can be that a result isn't luck. Rule
of thumb: **|t| > 2** means it's probably real. Below that, you can't
distinguish it from coincidence.

**Statistical power** — whether you have *enough data* to detect an
effect at all. Critical and widely ignored: with too little data, you'll
report "no effect" even when there is one. Like trying to hear a whisper
in a loud room — silence doesn't prove nobody spoke.

**Multiple comparisons** — the trap where testing many things guarantees
some look significant by pure chance. Test 20 features at the usual
threshold and about 1 will "pass" on luck alone.

**Calibration** — whether stated confidence matches reality. A
well-calibrated model that says "70% confident" is right about 70% of the
time. An uncalibrated one might say 90% and be right half the time.

**Brier score** — measures calibration quality. Lower is better.

**Backtest** — simulating a strategy on historical data to see what it
would have earned.

**Buy-and-hold** — just buying and sitting there. The lazy benchmark a
strategy has to beat to justify existing.

### Data sources

**13F filing** — a legally required disclosure of what big investment
firms own, published **45 days after** the quarter ends. Public and
stale by design. Automated news sites generate thousands of articles
from these ("Firm X sold 3,223 shares of Y"). They carry essentially no
predictive information, and they're **18.8% of this project's corpus**.

**Weak supervision** — generating training labels automatically instead
of paying humans to label. Here, labels come from price data: the market
itself says whether the news was good or bad.

---

## Part 2 — The core story

### Q: Walk me through this project.

> I built a pipeline that takes a financial news article and predicts the
> direction, size, and confidence of the affected stock's move over the
> next three trading days.
>
> It collects news for 21 large companies across sectors, pulls matching
> price history, and automatically generates training labels from what
> the price actually did — so no hand-labeling.
>
> The interesting part isn't the model. It's that my first three attempts
> all found no signal, and when I dug into *why*, the problem turned out
> to be in how I'd defined the thing I was predicting — not in the
> features I was feeding the model. Fixing that changed what the project
> is actually measuring.

**Why this works:** it leads with a real finding rather than a tech
stack, and it signals you debug your own methodology. Let them ask the
follow-up.

---

### Q: What was wrong with your target?

> Three things, and each one independently made the results misleading.
>
> **First, I was predicting the wrong thing.** My label was the raw price
> move. But on 43.7% of days, more than 80% of my 21 stocks moved the
> same direction — that's the whole market moving, not 21 separate news
> reactions. About 29% of my label's variance was market-wide.
>
> That's why my model collapsed to always predicting "up." Stocks rose
> 55% of the time in my sample period, so "always say up" scored 55%
> without predicting anything. I thought I had a weak-signal problem. I
> actually had a free lunch baked into my target.
>
> The fix is standard event-study methodology: subtract the market's
> share of the move, then divide by how volatile that stock normally is.
> After that, the base rate is 50.75% — no free lunch left. The model has
> to actually discriminate.

**Follow-up you should expect — "how do you compute the market's share?"**

> Beta — how strongly that stock tracks the market — estimated by
> regressing the stock's daily moves against the market's over the prior
> 120 trading days. Then abnormal return = the stock's move minus beta
> times the market's move.
>
> The critical detail is that the window **ends at the prediction date**.
> If I'd used the full sample to estimate beta, I'd be using future data
> to build a label about the past — lookahead bias, one level deeper than
> the train/test split, and much easier to miss.
>
> I have a test for exactly that: two price series identical up to the
> reference date and wildly different after it must produce an identical
> beta. If someone accidentally widens that window, the test fails.

---

### Q: What was the second problem?

> My sample was about 21 times smaller than I thought.
>
> Every article about Apple published on the same day maps to the same
> price move — so they all carry an **identical** label. I had 55,532
> rows, but only 2,630 distinct labels. I was counting the same
> observation twenty-one times.
>
> Two consequences. My confidence intervals were roughly 4.6 times too
> narrow — that's the square root of 21. And because the duplicates sat
> on both sides of my train/test boundary, near-copies of most test
> labels were also in training. Not classic leakage, but leakage.
>
> I collapsed it to one row per company per day. That's the honest unit:
> one label, one row. I kept the article count as a feature called
> `n_articles`, because a burst of coverage is genuinely informative —
> it just shouldn't be expressed as duplicated rows the model can't see.

**Why interviewers love this one:** most candidates never question their
own row count. Noticing that rows ≠ independent observations is a
statistics instinct, not a coding one.

---

### Q: And the third?

> A large share of my news wasn't news.
>
> Three distinct kinds of junk. **13F boilerplate** — auto-generated
> articles about investment firms' holdings, published 45 days after the
> fact. One site, MarketBeat, was 18.8% of my entire corpus.
> **Opinion pieces** — "Is Johnson & Johnson Still the Ultimate Safe
> Dividend Stock?" has no event in it and an arbitrary timestamp.
>
> And the sneakiest: **post-hoc explainers**. "Why KLA Stock Is Down
> Today" is published *because* the stock already moved. That move is
> already in my starting price, so those articles add noise to a forward
> prediction — they describe the past, not the future.
>
> I filter on title patterns plus a source blocklist. 63.6% of articles
> survive.

**Follow-up — "isn't a regex filter crude?"**

> Yes, deliberately. The goal is removing the obviously event-free bulk
> cheaply, not building an article-quality classifier. And I report the
> drop rate rather than assuming it, so if it ever started removing real
> events that would show up.
>
> It does have a real cost I'd state upfront: blocking by source drops
> genuine events from those publishers too.
>
> One thing I'd mention — I wrote the filter tests using headlines copied
> verbatim from my own data rather than invented examples. That caught a
> false positive immediately: "Enovix Shares Climb After Company Names
> Former Apple Manufacturing Leader as COO" was being dropped as a
> post-hoc explainer, but it reports a real event — an executive hire.
> So now a price-move headline is only dropped if it never states a
> cause. A filter tested against imagined examples would have passed its
> own tests and quietly failed on real data.

---

## Part 3 — Results, and how to present a negative one

### Q: So does it work?

Answer this one straight. Do not dress it up.

> No — not yet, and I can tell you precisely what "no" means here.
>
> After all three fixes, the information coefficient on the sentiment
> scores is +0.012 with a t-statistic of 0.52. That's indistinguishable
> from zero.
>
> But here's the part I think matters more. To detect an IC of 0.03 —
> which would be a genuinely useful signal — you need about 4,400
> observations. I have 2,002. **I'm underpowered by more than half.**
>
> So "there's no signal" and "I don't have enough data to see the signal"
> are currently the same answer, and I can't separate them. That's an
> honest limit, and it tells me the next investment is more data and
> better features — not another model on the same 2,002 rows.

**Why this is strong:** you turned a null result into a quantified,
actionable statement. That's the difference between "it didn't work" and
"here is exactly what it would take to find out."

---

### Q: Did anything look promising?

This is a trap, and the trap is that the honest answer is more impressive
than the exciting one.

> One feature crossed the usual significance threshold — a macro-economic
> topic flag, IC +0.047, t = 2.11.
>
> I don't believe it, and I'd say so in a report. I tested seven
> features. At the standard 5% threshold you expect about 0.35 false
> positives by chance across seven tests, so one marginal hit is exactly
> what noise looks like. That's the multiple-comparisons problem.
>
> Before I'd call it a finding, it would have to hold on data it wasn't
> discovered in.

**If you say only one thing in an interview, make it this.** Anyone can
report a number above a threshold. Knowing when *not* to is what
separates someone who can be trusted with a research result.

---

### Q: Your accuracy was 56.5%. Isn't that better than chance?

> No, and that's exactly the trap. The base rate was 56.5% — stocks rose
> that often in my sample. The model was predicting "up" for every single
> example. It scored the base rate because it *was* the base rate.
>
> That's also why accuracy was the wrong metric throughout. An IC of
> 0.03 corresponds to roughly 51.5% accuracy. Accuracy is too coarse a
> ruler to see the effect I'm looking for. IC and rank correlation are
> the right tools.

---

## Part 4 — Engineering questions

### Q: How do you know you don't have lookahead bias?

The single most important question in the interview. Have three layers
ready.

> Three places, and I guard each one.
>
> **The train/test split** is by time, never random. Random would let the
> model learn from next month while being tested on last month.
>
> **The label alignment** respects market hours. An article published
> during trading hours can affect that day's close. One published at 9pm
> can't — the market is shut, so the earliest reaction is the next
> session's close. Getting this backwards would mean using a price from
> *before* the news as if it already reflected the news.
>
> **The beta estimation** uses only the 120 days ending at the prediction
> date. This is the one people miss, because it's hidden inside label
> construction rather than in the model. I test it directly.

---

### Q: Tell me about a bug you found.

Pick the performance one — it has a pattern, and a pattern is a better
story than an incident.

> The same bug three times, which is what makes it worth telling.
>
> Building the dataset was recomputing per-row things that only vary per
> group. First it re-queried a ticker's entire price history for every
> article. That was invisible at 15 rows and crippling at 62,000.
>
> Then the abnormal-return calculation — which fits a 120-day regression
> — ran once per article, when it only depends on the company and the
> date. With around thirteen articles per company-day, that's an order of
> magnitude of identical regressions. Build went from minutes to 14
> seconds.
>
> The lesson I actually took: I only noticed because the data grew. The
> honest version is that I should be asking "what does this actually
> depend on?" at write time, not waiting for it to hurt.

---

### Q: How do you test a machine learning pipeline?

> I test the logic I own, not the model's output quality. 43 tests, all
> deterministic, none needing a trained model or a network call.
>
> That means date and price arithmetic, the market-hours cutoff, the
> time-based split, the beta estimator, the aggregation, the filter.
>
> I don't write tests asserting "accuracy > 0.6" — that's not a test,
> it's a measurement, and it makes the suite fail for reasons that aren't
> bugs. Model quality is what the evaluation harness is for.
>
> The tests that earned their keep: the filter test built from real
> headlines that caught a false positive, and the point-in-time beta test
> that makes a whole class of leakage impossible to reintroduce.

---

## Part 5 — Judgment questions

### Q: Why keep the failed attempts in your README?

> Because how I found the defects is more useful than the final number.
> If I deleted the three failed attempts, the interesting part — that I
> didn't accept my own negative result and went looking for why — would
> be gone.
>
> I marked them as superseded rather than removing them, and I stated
> explicitly that every figure in those sections predates the fix.

---

### Q: You did three rounds of feature engineering that all failed. What would you do differently?

> I'd have questioned the target before the third round of features.
>
> After the first negative result I added feature scaling and company
> identity. After the second I added topic categories. Both failed. I
> was iterating on the model side without ever asking whether the thing
> I was predicting was well-defined.
>
> The tell was there: the model collapsed to one class, and the base rate
> was suspiciously close to the accuracy. I read that as weak signal. It
> was actually a sign the target had drift baked into it.
>
> My rule now: **when a model collapses to the majority class, suspect
> the label before the features.**

---

### Q: What's the biggest weakness of this project?

Say this before they find it.

> There's no NLP in it yet.
>
> Every feature is a number Alpha Vantage computed for me — their
> sentiment score, their topic tags. I never touch the headline text. So
> it's really scikit-learn on someone else's model's output, and it's
> capped by how good their sentiment model is.
>
> That's my next step and the one I'd prioritize: fine-tune an encoder on
> the headlines against my own labels, so the text-to-signal step is
> something I own and can improve. It's also the step where the remaining
> upside is, since I've now shown three transformations of their three
> numbers don't help.

**Why lead with this:** naming your own biggest gap is disarming, and it
converts a weakness into a roadmap. It also pre-empts the worst possible
version of the question.

---

## Part 6 — Numbers to have ready

| Fact | Number |
|---|---|
| Companies tracked | 21, plus SPY as benchmark |
| Articles collected | 38,507 |
| Article–company pairs | 120,106 |
| Prediction horizon | 3 trading days |
| **Duplication factor** | **~21x** (55,532 rows, 2,630 distinct labels) |
| **Event-level observations** | **2,002** |
| Label variance that was market-wide | ~29% |
| Days where >80% of stocks moved together | 43.7% |
| **Base rate, raw return** | **53.1%** |
| **Base rate, abnormal return** | **50.75%** |
| Articles surviving the event filter | 63.6% |
| MarketBeat share of corpus | 18.8% |
| Average beta | 0.78 |
| Beta window / minimum | 120 days / 60 observations |
| IC, sentiment (event-level) | +0.012, t = 0.52 |
| Observations needed for IC=0.03 | ~4,400 |
| Tests | 43, ~1 second |
| Dataset build time | 14 seconds |

---

## Part 7 — Questions that could sink you

Rehearse these. The failure mode is bluffing.

**"What's your Sharpe ratio?"**
> I don't report one. My signal isn't statistically distinguishable from
> zero, so a Sharpe ratio would be a number without a result behind it.
> I also don't model transaction costs, so any figure would be optimistic
> by an unknown amount.

*(Sharpe ratio = return relative to how much it bounced around. Higher is
better; roughly, "was the profit worth the stress." Above 1 is good.)*

**"Why three days? Why not one, or intraday?"**
> Mostly a data constraint — free intraday history doesn't go back a
> year. It's a real limitation, because news impact is concentrated in
> the first minutes to hours, and a three-day window lets unrelated
> noise accumulate. I'd expect a shorter horizon to have a better
> signal-to-noise ratio, and that's a change I'd make with better data.

**"Your universe is 21 healthy large companies. Isn't that biased?"**
> Yes, and it's in my limitations. I picked companies that are currently
> large and liquid, which means I picked survivors — that's hindsight
> bias. A fair universe would be chosen as of the *start* of the period
> and include ones that later did badly.

**"Isn't SPY alone too crude a benchmark?"**
> It is. A sector shock still lands in my "abnormal" return for every
> company in that sector. The standard improvement is adding sector
> factors, so a pharma-wide selloff doesn't read as company-specific
> news.

**"Where did the signal go? You said market adjustment doubled it."**
> That was an early measurement on duplicated data using a crude market
> proxy, and it didn't survive doing it properly. Once I used a real
> benchmark and counted each observation once, the apparent signal in raw
> returns turned out to be the market component — which is exactly what
> the adjustment is supposed to remove. The higher number was measuring
> the thing I was trying to get rid of.

**If you genuinely don't know something:**
> I haven't worked with that — here's how I'd approach finding out.

Never bluff a finance concept. The person across the table has years on
you there, and they already know you're early-career. What they're
actually testing is whether your claims can be trusted.

---

## Closing line, if you get one

> The headline result is negative, and the useful part is that it's a
> *trustworthy* negative now. Before, I couldn't tell whether my
> sentiment features were uninformative or whether my label was mostly
> measuring the S&P 500. Now I can — and I know it would take about
> twice the data to answer the next question properly.
