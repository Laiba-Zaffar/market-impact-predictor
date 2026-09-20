# Market Impact Predictor

Given a financial news event, predict the direction, magnitude, and
calibrated confidence of the stock price move that follows it. Builds on
[fin-event-intel](../fin-event-intel)'s event/entity extraction work, but
needs its own historical dataset — a live feed only teaches a model what
happens *after* you start running it, never what happened before.

## Roadmap

- [x] M0 — repo scaffold
- [ ] M1 — historical news+sentiment ingestion (Alpha Vantage, quota-aware)
- [x] M2a — historical price data (yfinance, 14mo × 21 tickers)
- [x] M2b — price alignment → forward-return labels
- [ ] M3 — dataset construction (time-based split, no lookahead leakage)
- [ ] M4 — baseline model: direction classifier + magnitude regressor
- [ ] M5 — confidence calibration (temperature scaling / isotonic regression)
- [ ] M6 — backtest evaluation (with honest limitations, not a hype number)
- [ ] M7 — portfolio polish

## Design note: baseline before deep learning

M4 starts with classical ML (gradient boosting / logistic regression on
sentiment + relevance + topic features), not a transformer. Same
philosophy as Project 1's M3: ship a simple, fast, CPU-friendly baseline
first, measure where it's weak, then decide if a heavier model is
actually justified — rather than assuming bigger is better before there's
evidence either way.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in ALPHA_VANTAGE_API_KEY
```
