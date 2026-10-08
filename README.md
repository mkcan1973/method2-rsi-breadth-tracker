# method2 RSI+Breadth Tracker

Tracks method1's validated RSI(14) mean-reversion signal (buy when a symbol's
14-day RSI drops to <=20, hold until it recovers to >=65), gated by market
breadth: new entries only fire when at least 60% of the S&P 500 + core
ETFs/index universe is trading above its own 50-day moving average.

Two earlier candidate second signals were tried and failed validation before
breadth: a 200-day price-trend filter (structurally anti-correlated with RSI
oversold at every window tested -- a sharp enough drop to trigger RSI<=20
almost always also drags a trailing price MA down with it) and a
volume-confirmation filter (kept a healthy sample size but carried no real
discriminating information). Breadth is a market-wide regime measure,
genuinely independent of any one stock's own price/volume, and it replicated
out of sample -- beating plain RSI on Sharpe/CAGR/max-drawdown in both an
independent train (2016-2021) and test (2021-2026) half -- at the cost of a
lower win rate and a smaller, more concentrated book (~5-7 positions at a
time vs. method1's ~24-26). See the companion research project
(`method1/method2_breadth_filter_train_test.py` and the other
`method1/method2_*` scripts) for the full validation trail.

## Key features

The tracker runs automatically on weekdays via GitHub Actions, once daily
after market close. It generates:

- A web dashboard (`index.html`, served via GitHub Pages) showing open and
  closed positions over the trailing 12 months, with ROI and an equity curve
- `scan_results.csv` -- today's RSI+breadth scan across the full universe
- `trade_log_12mo.csv` -- the full open/closed trade log behind the dashboard
- `method2.sqlite3` -- daily OHLCV price history, pruned to a trailing window
  (enough for RSI/breadth warmup + the 12-month trade log) so this repo's
  database stays small; a separate local copy keeps full multi-year history
  for research/backtesting

## Usage

Signal only -- this places no orders. Run `python scan.py` for today's scan,
or `python web_summary.py` to rebuild the dashboard. Both handle their own
data refresh from Yahoo Finance.
