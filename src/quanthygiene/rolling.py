"""Rolling performance metrics.

Answers: is the edge stable through time, or concentrated in one stretch? Every function takes
per-period returns and returns a Series aligned with the input (NaN until the window is full).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_sharpe(returns, window: int = 60, periods_per_year: float = 252) -> pd.Series:
    """Annualised Sharpe ratio over a rolling window."""
    r = pd.Series(np.asarray(returns, dtype=float))
    sd = r.rolling(window).std(ddof=1)
    return r.rolling(window).mean() / sd.where(sd > 0) * np.sqrt(periods_per_year)


def rolling_drawdown(returns, window: int = 60) -> pd.Series:
    """Worst peak-to-trough fractional drawdown inside each rolling window (reported as a positive number)."""
    r = np.asarray(returns, dtype=float)
    out = np.full(len(r), np.nan)
    for i in range(window, len(r) + 1):
        eq = np.cumprod(1 + r[i - window:i])
        eq = np.concatenate([[1.0], eq])
        out[i - 1] = (1 - eq / np.maximum.accumulate(eq)).max()
    return pd.Series(out)


def rolling_hit_rate(returns, window: int = 60) -> pd.Series:
    """Share of positive periods in a rolling window."""
    r = pd.Series(np.asarray(returns, dtype=float))
    return (r > 0).astype(float).rolling(window).mean()
