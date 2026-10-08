"""
Daily signal scan for method2.

Strategy: RSI(14) mean reversion (same entry/exit thresholds validated
for method1: buy <=20, hold until >=65), gated by market breadth -- the
buy signal only fires when at least 60% of the S&P 500 + core ETFs/index
universe is trading above its own 50-day MA. Two earlier candidates
(price vs. its own 200-day MA, volume vs. its own 20-day average) failed
validation: the trend filter was structurally anti-correlated with RSI
oversold at every window tested, and the volume filter carried no real
information. Breadth is a market-wide regime measure, genuinely
orthogonal to any one stock's own price/volume, and it replicated out of
sample: beats plain RSI on Sharpe/CAGR/maxDD in BOTH an independent train
(2016-2021) and test (2021-2026) half -- at the cost of a lower win rate
and fewer, more concentrated positions (~5-7 at a time vs method1's
~24-26). See method1/method2_breadth_filter_train_test.py for that
validation trail.

Signal/alert only: this prints what's actionable today. It places no
orders.
"""

from pathlib import Path

import pandas as pd

import breadth as breadth_mod
import data_db
import fetch_data
import fetch_stock_data
import features
import indicators
from universe import tradeable_tickers
from stock_universe import get_sp500_tickers

RSI_BUY = 20
RSI_EXIT = 65
MIN_BREADTH = 0.60
BREADTH_MA_WINDOW = 50

SUMMARY_CSV = Path(__file__).with_name("scan_results.csv")


def main():
    print("Refreshing ETF/index data...")
    fetch_data.main()
    print("\nRefreshing S&P 500 stock data (incremental)...")
    fetch_stock_data.main()
    print()

    conn = data_db.connect()
    symbols = list(dict.fromkeys(tradeable_tickers() + get_sp500_tickers()))

    bars_by_symbol = {}
    for symbol in symbols:
        bars = data_db.load_bars(conn, symbol)
        if bars is None or len(bars) < 60:
            continue
        bars_by_symbol[symbol] = bars
    conn.close()

    breadth = breadth_mod.compute_breadth(
        {s: b["close"] for s, b in bars_by_symbol.items()}, BREADTH_MA_WINDOW
    )
    today_breadth = breadth.iloc[-1]

    rows = []
    for symbol, bars in bars_by_symbol.items():
        feats = features.build_features(bars)
        feats["breadth_50"] = breadth.reindex(feats.index)
        position = indicators.rsi_breadth_reversion(feats, RSI_BUY, RSI_EXIT, MIN_BREADTH)
        if len(position) < 2:
            continue

        pos_today, pos_yesterday = position.iloc[-1], position.iloc[-2]
        if pos_today == 1 and pos_yesterday == 0:
            status = "NEW_ENTRY"
        elif pos_today == 1 and pos_yesterday == 1:
            status = "HOLDING"
        elif pos_today == 0 and pos_yesterday == 1:
            status = "EXIT"
        else:
            status = "FLAT"

        rows.append(dict(
            symbol=symbol,
            as_of=feats.index[-1].date(),
            close=bars["close"].iloc[-1],
            rsi_14=feats["rsi_14"].iloc[-1],
            status=status,
        ))

    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_CSV, index=False)

    as_of = df["as_of"].iloc[0] if not df.empty else "n/a"
    print(f"Rule: RSI(14) <= {RSI_BUY} -> buy (if breadth >= {MIN_BREADTH:.0%}), "
          f"hold until RSI >= {RSI_EXIT} -> exit")
    print(f"As of: {as_of}  |  breadth today: {today_breadth:.0%} of universe above its 50-day MA "
          f"({'filter OPEN -- new entries allowed' if today_breadth >= MIN_BREADTH else 'filter CLOSED -- no new entries today'})")
    print(f"{len(df)} symbols scanned")
    print()

    counts = df["status"].value_counts()
    print(f"{counts.get('NEW_ENTRY', 0)} new entries today, {counts.get('EXIT', 0)} exits today, "
          f"{counts.get('HOLDING', 0)} positions already held, {counts.get('FLAT', 0)} flat")
    print()

    actionable = df[df["status"].isin(["NEW_ENTRY", "EXIT"])].sort_values(["status", "symbol"])
    if not actionable.empty:
        print("=== Actionable today ===")
        print(actionable.to_string(index=False))
    else:
        print("Nothing new today.")

    holding = df[df["status"] == "HOLDING"].sort_values("symbol")
    if not holding.empty:
        print()
        print(f"=== Currently held ({len(holding)}) -- see {SUMMARY_CSV.name} for the full list ===")
        print(holding.head(15).to_string(index=False))
        if len(holding) > 15:
            print(f"... and {len(holding) - 15} more")

    print()
    print("Execution convention: today's close is the signal, not a fill price --")
    print("if trading this for real, place entries/exits at tomorrow's OPEN.")
    print("Signal only -- no orders placed.")


if __name__ == "__main__":
    main()
