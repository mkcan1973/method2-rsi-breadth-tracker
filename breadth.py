"""
Market-breadth computation for method2: a market-wide regime signal,
genuinely orthogonal to any single stock's own price/volume history
(unlike the trend and volume filters tried first -- both computed from
the SAME symbol's own series as the RSI entry signal, which is why they
either collapsed the sample or carried no real information).
"""

import pandas as pd


def compute_breadth(close_by_symbol: dict, ma_window: int = 50) -> pd.Series:
    """Fraction of the universe trading above its own `ma_window`-day
    moving average, per date. Symbols are included from whenever their
    own MA first has enough history -- not forced to all 515 having data
    from day one, which would make early-history breadth meaningless.
    """
    above_ma = {}
    for symbol, close in close_by_symbol.items():
        ma = close.rolling(ma_window).mean()
        above_ma[symbol] = (close > ma).astype(float).where(ma.notna())
    return pd.DataFrame(above_ma).mean(axis=1, skipna=True)
