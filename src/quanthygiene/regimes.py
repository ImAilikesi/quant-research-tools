"""Performance by volatility regime.

Answers: does the track record depend on the market being calm or turbulent? Each period is
labelled by the tercile of *trailing* volatility measured before it (no look-ahead), and
performance is summarised per label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def volatility_regimes(returns, window: int = 20, n_bins: int = 3) -> pd.Series:
    """Label each period 0..n_bins-1 (0 = calmest) by trailing volatility known before the period.

    Quantile cut-offs use the full sample, so labels describe relative regimes; the trailing window
    itself excludes the labelled period. Early periods without a full window are NaN.
    """
    r = pd.Series(np.asarray(returns, dtype=float))
    vol = r.rolling(window).std(ddof=1).shift(1)
    return pd.qcut(vol, n_bins, labels=False)


def regime_performance(returns, window: int = 20, n_bins: int = 3, periods_per_year: float = 252) -> pd.DataFrame:
    """Mean, volatility, Sharpe, hit rate and count of returns per volatility regime."""
    r = pd.Series(np.asarray(returns, dtype=float))
    lab = volatility_regimes(r, window, n_bins)
    rows = []
    for k in range(n_bins):
        x = r[lab == k]
        sd = x.std(ddof=1)
        rows.append({"regime": k, "n": len(x), "mean": x.mean(), "vol": sd,
                     "sharpe": x.mean() / sd * np.sqrt(periods_per_year) if sd > 0 else 0.0,
                     "hit_rate": (x > 0).mean()})
    return pd.DataFrame(rows)
