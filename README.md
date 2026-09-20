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
- [~] M4 — baseline model: direction classifier + magnitude regressor — **code complete, not yet run** (blocked on scikit-learn install, see below)
- [~] M5 — confidence calibration (Brier score + reliability curve) — **code complete, not yet run**, same blocker
- [~] M6 — backtest evaluation (with honest limitations, not a hype number) — **code complete, not yet run**, same blocker
- [ ] M7 — portfolio polish

## Current blocker: scikit-learn won't install

`pip install scikit-learn` has stalled four separate times on this
machine (connection established, then zero data movement for minutes) -
a real, reproducible network issue with large-wheel downloads today, not
random bad luck. `pandas` and `joblib` installed fine, so M3 and the
parts of M4-M6 that don't need a trained model are written, tested, and
verified. The actual model training/calibration/backtest runs are
blocked purely on that one dependency - re-run `pip install scikit-learn
matplotlib` once the network's cooperative, then `python -m
src.run_pipeline` to execute M4 through M6 in one go.

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
