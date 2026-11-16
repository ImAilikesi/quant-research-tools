"""OHLC volatility estimators: close-to-close, Parkinson, Garman-Klass, Yang-Zhang.

Answers: how volatile is this instrument, using more of the bar than the close alone? Input is a
DataFrame with ``open, high, low, close`` columns. Results are rolling volatility per bar,
annualised by ``periods_per_year``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _check(df: pd.DataFrame) -> None:
    missing = {"open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")


def close_to_close(df: pd.DataFrame, window: int = 20, periods_per_year: float = 252) -> pd.Series:
    """Rolling standard deviation of log close-to-close returns."""
    _check(df)
    r = np.log(df["close"] / df["close"].shift(1))
    return r.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)


def parkinson(df: pd.DataFrame, window: int = 20, periods_per_year: float = 252) -> pd.Series:
    """High-low range estimator; efficient for driftless, gap-free prices."""
    _check(df)
    x = np.log(df["high"] / df["low"]) ** 2 / (4 * np.log(2))
    return np.sqrt(x.rolling(window).mean() * periods_per_year)


def garman_klass(df: pd.DataFrame, window: int = 20, periods_per_year: float = 252) -> pd.Series:
    """Garman-Klass estimator using open, high, low and close; assumes no overnight gap."""
    _check(df)
    hl = np.log(df["high"] / df["low"]) ** 2
    co = np.log(df["close"] / df["open"]) ** 2
    x = 0.5 * hl - (2 * np.log(2) - 1) * co
    return np.sqrt(x.rolling(window).mean() * periods_per_year)


def yang_zhang(df: pd.DataFrame, window: int = 20, periods_per_year: float = 252) -> pd.Series:
    """Yang-Zhang estimator: handles drift and opening gaps; minimum-variance combination."""
    _check(df)
    o, h, lo, c = (np.log(df[k]) for k in ("open", "high", "low", "close"))
    overnight = o - c.shift(1)
    open_close = c - o
    rs = (h - c) * (h - o) + (lo - c) * (lo - o)
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    var_o = overnight.rolling(window).var(ddof=1)
    var_c = open_close.rolling(window).var(ddof=1)
    var_rs = rs.rolling(window).mean()
    return np.sqrt((var_o + k * var_c + (1 - k) * var_rs) * periods_per_year)


ESTIMATORS = {"close-to-close": close_to_close, "parkinson": parkinson,
              "garman-klass": garman_klass, "yang-zhang": yang_zhang}
