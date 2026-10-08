"""
Traditional technical-indicator strategies (RSI, MACD, and a combination),
plugged into the same walk-forward harness as baselines.py and model.py.

rsi_macd_combo and rsi_overbought_short still use standard textbook
defaults, not fitted to this data. rsi_reversion's defaults are the one
exception: buy<=20/exit>=65 was chosen via threshold_grid.py (a grid
search over buy/exit combos) specifically because it replicated -- it
beat the original textbook 30/55 defaults on every metric in BOTH
independent train (2016-2021) and test (2021-2026) halves, not just on
the combined period the grid search itself ran on. See
threshold_grid.py, threshold_grid_extended.py, and threshold_train_test.py
for that validation. Changing it again should clear the same bar.
"""

import numpy as np
import pandas as pd


def rsi_reversion(full_feats: pd.DataFrame, buy_thresh: float = 20, exit_thresh: float = 65, **_) -> pd.Series:
    """Long-only: buy when RSI drops to oversold, hold until it recovers
    past exit_thresh, flat otherwise. Classic buy-the-dip mean reversion.
    """
    rsi = full_feats["rsi_14"]
    position = np.zeros(len(rsi))
    in_pos = False
    for i, r in enumerate(rsi.to_numpy()):
        if np.isnan(r):
            continue
        if not in_pos and r <= buy_thresh:
            in_pos = True
        elif in_pos and r >= exit_thresh:
            in_pos = False
        position[i] = 1.0 if in_pos else 0.0
    return pd.Series(position, index=rsi.index)


def rsi_trend_reversion(full_feats: pd.DataFrame, buy_thresh: float = 20, exit_thresh: float = 65,
                          trend_window: int = 200, **_) -> pd.Series:
    """rsi_reversion, gated by a long-term trend filter: only take the buy
    signal when price is above its `trend_window`-day moving average.
    Exit logic is unchanged (RSI recovering past exit_thresh) -- this is a
    pure ENTRY filter, not a new exit rule, to isolate what the trend
    condition itself changes.

    Motivation (method2, the first "second signal" on top of method1):
    plain RSI oversold doesn't distinguish ordinary dip-buying noise from
    a stock oversold for a structural reason (fraud, bankruptcy, a
    short-seller report) -- see rsi_reversion_with_stop's docstring and
    the method1 live dashboard's long-held-open-position problem (AON,
    CTVA). A stock already below its 200-day MA is, by definition, in a
    longer-run downtrend; requiring it to be ABOVE that average before
    buying a dip is a trend/regime filter, not a second momentum
    indicator like MACD (already tested and rejected as redundant with
    RSI in rsi_macd_combo) -- it's orthogonal information about the
    longer-term regime the dip is happening in.
    """
    rsi = full_feats["rsi_14"]
    dist_ma = full_feats[f"dist_ma_{trend_window}"]
    position = np.zeros(len(rsi))
    in_pos = False
    rsi_vals, dist_vals = rsi.to_numpy(), dist_ma.to_numpy()
    for i in range(len(rsi)):
        r, d = rsi_vals[i], dist_vals[i]
        if np.isnan(r):
            continue
        if not in_pos and r <= buy_thresh and not np.isnan(d) and d > 0:
            in_pos = True
        elif in_pos and r >= exit_thresh:
            in_pos = False
        position[i] = 1.0 if in_pos else 0.0
    return pd.Series(position, index=rsi.index)


def rsi_volume_reversion(full_feats: pd.DataFrame, buy_thresh: float = 20, exit_thresh: float = 65,
                           min_rel_volume: float = 1.5, **_) -> pd.Series:
    """rsi_reversion, gated by a volume-confirmation filter: only take the
    buy signal when that day's volume is at least `min_rel_volume` times
    its own trailing 20-day average (full_feats["rel_volume_20"]). Exit
    logic is unchanged -- a pure ENTRY filter, same shape as
    rsi_trend_reversion, swapped to a genuinely different second signal
    after the 200-day trend filter turned out to be structurally
    anti-correlated with RSI oversold at every window tested (a sharp
    enough drop to trigger RSI<=20 almost always also drags a trailing
    price MA down with it, so "price above its MA" rarely holds).

    Volume doesn't have that problem -- elevated volume on the oversold
    day is a hypothesis about panic/capitulation selling (often marking a
    tradeable bottom), not a restatement of the price move itself, so
    it's a genuinely different axis of information than RSI.
    """
    rsi = full_feats["rsi_14"]
    rel_vol = full_feats["rel_volume_20"]
    position = np.zeros(len(rsi))
    in_pos = False
    rsi_vals, rel_vol_vals = rsi.to_numpy(), rel_vol.to_numpy()
    for i in range(len(rsi)):
        r, v = rsi_vals[i], rel_vol_vals[i]
        if np.isnan(r):
            continue
        if not in_pos and r <= buy_thresh and not np.isnan(v) and v >= min_rel_volume:
            in_pos = True
        elif in_pos and r >= exit_thresh:
            in_pos = False
        position[i] = 1.0 if in_pos else 0.0
    return pd.Series(position, index=rsi.index)


def rsi_breadth_reversion(full_feats: pd.DataFrame, buy_thresh: float = 20, exit_thresh: float = 65,
                            min_breadth: float = 0.5, **_) -> pd.Series:
    """rsi_reversion, gated by market breadth: only take the buy signal
    when at least `min_breadth` of the whole universe is trading above
    its own 50-day MA (full_feats["breadth_50"], joined in by the caller
    -- see breadth.py). This is a MARKET-WIDE regime filter, not a
    restatement of this symbol's own price/volume, unlike the trend
    filter (anti-correlated with RSI oversold by construction) and the
    volume filter (no real information) tried first for method2.

    Hypothesis: buying an oversold dip is a bad idea during a broad
    market selloff (everything's oversold for the same bad reason) but a
    good idea when the broader market is healthy and this one name
    happens to be dipping on its own.
    """
    rsi = full_feats["rsi_14"]
    breadth = full_feats["breadth_50"]
    position = np.zeros(len(rsi))
    in_pos = False
    rsi_vals, breadth_vals = rsi.to_numpy(), breadth.to_numpy()
    for i in range(len(rsi)):
        r, b = rsi_vals[i], breadth_vals[i]
        if np.isnan(r):
            continue
        if not in_pos and r <= buy_thresh and not np.isnan(b) and b >= min_breadth:
            in_pos = True
        elif in_pos and r >= exit_thresh:
            in_pos = False
        position[i] = 1.0 if in_pos else 0.0
    return pd.Series(position, index=rsi.index)


def rsi_reversion_with_stop(full_bars: pd.DataFrame, full_feats: pd.DataFrame,
                             buy_thresh: float = 30, exit_thresh: float = 55,
                             stop_loss_pct: float = 0.15, **_) -> pd.Series:
    """rsi_reversion, plus a hard stop-loss: exit if price falls stop_loss_pct
    below the entry price, regardless of what RSI is doing. Without this, a
    stock that's oversold for a real fundamental reason (fraud, bankruptcy,
    a short-seller report) rather than ordinary noise can sit in the book
    for months with RSI pinned near zero, never triggering the RSI exit --
    see indicators.py's module docstring history / the method1 ROI
    discussion for why this matters at portfolio scale, not just per trade.
    """
    rsi = full_feats["rsi_14"]
    close = full_bars["close"].reindex(rsi.index)
    position = np.zeros(len(rsi))
    in_pos = False
    entry_price = None
    rsi_vals, close_vals = rsi.to_numpy(), close.to_numpy()
    for i in range(len(rsi)):
        r, c = rsi_vals[i], close_vals[i]
        if np.isnan(r):
            continue
        if not in_pos and r <= buy_thresh:
            in_pos, entry_price = True, c
        elif in_pos and (r >= exit_thresh or c <= entry_price * (1 - stop_loss_pct)):
            in_pos, entry_price = False, None
        position[i] = 1.0 if in_pos else 0.0
    return pd.Series(position, index=rsi.index)


def rsi_overbought_short(full_feats: pd.DataFrame, sell_thresh: float = 70, exit_thresh: float = 45, **_) -> pd.Series:
    """Short-only mirror of rsi_reversion: short when RSI rises to
    overbought, hold the short until it cools back down past exit_thresh,
    flat otherwise. Same mean-reversion logic, opposite direction -- note
    shorting fights the market's long-run upward drift in a way buying
    dips doesn't, so this needs its own out-of-sample check, not an
    assumption that symmetry holds.
    """
    rsi = full_feats["rsi_14"]
    position = np.zeros(len(rsi))
    in_pos = False
    for i, r in enumerate(rsi.to_numpy()):
        if np.isnan(r):
            continue
        if not in_pos and r >= sell_thresh:
            in_pos = True
        elif in_pos and r <= exit_thresh:
            in_pos = False
        position[i] = -1.0 if in_pos else 0.0
    return pd.Series(position, index=rsi.index)


def macd_crossover(full_feats: pd.DataFrame, **_) -> pd.Series:
    """Long/short trend-following: long while MACD line is above its
    signal line, short while below. Always in a position (no flat state).
    """
    macd = full_feats["macd"]
    signal = full_feats["macd_signal"]
    position = pd.Series(np.where(macd > signal, 1.0, -1.0), index=macd.index)
    position[macd.isna() | signal.isna()] = 0.0
    return position


def rsi_macd_combo(full_feats: pd.DataFrame, buy_thresh: float = 35, exit_thresh: float = 55, **_) -> pd.Series:
    """Long-only: enter only when RSI is oversold AND MACD histogram has
    turned positive (momentum confirmation), exit when RSI recovers past
    exit_thresh OR momentum rolls back over -- requiring both indicators to
    agree is the whole point of testing a "combination" rather than either
    alone.
    """
    rsi = full_feats["rsi_14"]
    hist = full_feats["macd_hist"]
    position = np.zeros(len(rsi))
    in_pos = False
    rsi_vals, hist_vals = rsi.to_numpy(), hist.to_numpy()
    for i in range(len(rsi)):
        r, h = rsi_vals[i], hist_vals[i]
        if np.isnan(r) or np.isnan(h):
            continue
        if not in_pos and r <= buy_thresh and h > 0:
            in_pos = True
        elif in_pos and (r >= exit_thresh or h < 0):
            in_pos = False
        position[i] = 1.0 if in_pos else 0.0
    return pd.Series(position, index=rsi.index)
