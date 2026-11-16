"""FX diagnostics: quote-direction normalisation, diversification, currency exposure.

Answers: are my pairs really diversified or the same USD bet, once quote direction is aligned?
What is my net exposure per currency across all open pair positions?
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_pair(pair: str) -> tuple[str, str]:
    """'EURUSD' or 'EUR/USD' -> ('EUR', 'USD') (base, quote)."""
    p = pair.replace("/", "").replace("_", "").upper()
    if len(p) != 6:
        raise ValueError(f"expected a 6-letter pair, got {pair!r}")
    return p[:3], p[3:]


def normalize_quote_direction(prices: pd.DataFrame, numeraire: str = "USD") -> pd.DataFrame:
    """Express every pair as the price of the *other* currency in ``numeraire``; return log changes.

    EURUSD already is USD per EUR. USDJPY is inverted to USD per JPY. Afterwards a positive return
    means the foreign currency gained against ``numeraire`` for every column, so signs are comparable.
    Columns are renamed to the foreign currency code. Pairs without ``numeraire`` raise an error.
    """
    out = {}
    for pair in prices.columns:
        base, quote = split_pair(pair)
        if quote == numeraire:
            level, name = prices[pair], base
        elif base == numeraire:
            level, name = 1.0 / prices[pair], quote
        else:
            raise ValueError(f"{pair} does not contain numeraire {numeraire}")
        out[name] = np.log(level).diff()
    return pd.DataFrame(out).dropna()


def diversification_check(returns: pd.DataFrame, weights=None) -> dict:
    """Correlation, PCA concentration, effective number of bets and diversification ratio."""
    x = returns.dropna()
    cov = np.cov(x.to_numpy().T)
    vol = np.sqrt(np.diag(cov))
    corr = cov / np.outer(vol, vol)
    eig = np.sort(np.linalg.eigvalsh(corr))[::-1]
    share = eig / eig.sum()
    enb = float(np.exp(-(share[share > 0] * np.log(share[share > 0])).sum()))
    w = np.ones(len(vol)) / len(vol) if weights is None else np.asarray(weights, dtype=float)
    port_vol = float(np.sqrt(w @ cov @ w))
    n = len(vol)
    off = corr[np.triu_indices(n, 1)]
    return {"columns": list(x.columns), "corr": pd.DataFrame(corr, x.columns, x.columns),
            "mean_pairwise_corr": float(off.mean()) if n > 1 else 0.0,
            "pc1_variance_share": float(share[0]), "effective_bets": enb,
            "diversification_ratio": float(w @ vol / port_vol) if port_vol > 0 else float("nan")}


def currency_exposure(positions: dict[str, float], prices: dict[str, float]) -> dict[str, float]:
    """Net exposure per currency from pair positions.

    ``positions``: pair -> signed units of the base currency (negative = short). ``prices``: pair ->
    current quote. Long 100000 EURUSD at 1.10 gives EUR +100000 and USD -110000. Exposures are in
    units of each currency; compare them after converting if you need a single number.
    """
    exp: dict[str, float] = {}
    for pair, units in positions.items():
        base, quote = split_pair(pair)
        exp[base] = exp.get(base, 0.0) + units
        exp[quote] = exp.get(quote, 0.0) - units * prices[pair]
    return exp
