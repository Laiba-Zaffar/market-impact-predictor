# A deliberately diversified basket, not just tech - different sectors
# behave differently around news, which matters for a model that's
# supposed to generalize across "market-moving events" broadly.
#
# One ticker per API call, not batched: Alpha Vantage's `tickers` param
# is an AND filter ("articles mentioning ALL listed tickers"), not an OR -
# confirmed the hard way after a batched backfill run returned near-zero
# results. Learned this from AV's own docs wording after the fact:
# "tickers=COIN,CRYPTO:BTC,FOREX:USD will filter for articles that
# simultaneously mention" all three - not any of them.
TICKER_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL",   # tech
    "JPM", "BAC", "V",                  # financials
    "JNJ", "UNH", "PFE",                # healthcare
    "AMZN", "WMT", "MCD",               # consumer
    "XOM", "CVX",                       # energy
    "BA", "CAT",                        # industrials
    "META", "DIS",                      # communication
    "TSLA", "GM",                       # auto
]

# Market proxy for abnormal-return labels. Measured on this data, ~29% of
# raw forward-return variance is market-wide rather than news-specific -
# so a raw close-to-close label mostly asks the model to predict the
# index from one company's headline. SPY is held in the same `prices`
# table as any other ticker (it just never appears in a news query).
BENCHMARK_TICKER = "SPY"
