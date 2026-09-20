# Market Impact Predictor

Given a financial news event, predict the direction, magnitude, and
calibrated confidence of the stock price move that follows it. Builds on
[fin-event-intel](../fin-event-intel)'s event/entity extraction work, but
needs its own historical dataset — a live feed only teaches a model what
happens *after* you start running it, never what happened before.

## Roadmap

- [x] M0 — repo scaffold
- [~] M1 — historical news+sentiment ingestion (Alpha Vantage, quota-aware) — **in progress**, running via a daily cron job (25 calls/day free tier), a few more days to finish the full 1-year backfill
- [x] M2a — historical price data (yfinance, 14mo × 21 tickers)
- [x] M2b — price alignment → forward-return labels
- [x] M3 — dataset construction (time-based split, no lookahead leakage) — code complete, verified against real data
- [x] M4 — baseline model: direction classifier + magnitude regressor — code runs end-to-end; results not meaningful yet (see below)
- [x] M5 — confidence calibration (Brier score + reliability curve) — correctly no-ops until there's enough data (see below)
- [x] M6 — backtest evaluation (with honest limitations, not a hype number) — correctly no-ops until there's enough data
- [ ] M7 — portfolio polish

## M4-M6 ran, but on far too little data to mean anything yet

`scikit-learn` stalled four times installing on this machine (established
connection, zero data movement) before finally installing cleanly on a
later attempt - real network flakiness that day, not a code problem.
Once it installed, `python -m src.run_pipeline` ran M4 through M6
end-to-end for the first time, on the 15 labeled examples that exist so
far:

- **Direction classifier**: accuracy 0.667, but precision/recall/F1 are
  all 0.000 - the confusion matrix shows it predicted "down" for every
  single example. That's not a bug, it's the model correctly finding
  no real signal in 15 points and falling back to the majority class
  (10 of the 15 were actually down). Expect this to keep happening
  until there's real volume.
- **Calibration (M5)** and **backtest (M6)** both correctly refused to
  run at all - 15 examples is below the 30-minimum guard, and both
  print an explicit message saying so rather than producing a number
  that would look like a result but isn't one.

This is the pipeline working exactly as designed: it's built to be
honest about when it doesn't have enough to say anything, not to
produce a plausible-looking number regardless. Re-run
`python -m src.run_pipeline` any time - once the Alpha Vantage backfill
(M1) has enough volume, the exact same code will produce a real
evaluation with no changes needed.

## Design note: baseline before deep learning

M4 starts with plain logistic/linear regression, not gradient boosting
or a transformer. Same philosophy as Project 1's M3: ship a simple,
fast, CPU-friendly baseline first, measure where it's weak, then decide
if a heavier model is actually justified - rather than assuming bigger
is better before there's evidence either way. Bonus: linear model
coefficients are directly interpretable, which a black-box ensemble
wouldn't give for free at this stage.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in ALPHA_VANTAGE_API_KEY
```
