"""Monte Carlo equity paths from a trade list, with confidence bands.

Answers: given my trade P&Ls, what range of equity curves was plausible, and how likely is a loss
or a deep drawdown over the next N trades? Paths are built by resampling trades with replacement
(``mode='bootstrap'``) or by reordering the same trades (``mode='shuffle'``; final P&L is then fixed).
"""

from __future__ import annotations

import numpy as np

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def equity_paths(pnls, n_trades: int | None = None, n_sims: int = 2000, mode: str = "bootstrap",
                 start_equity: float = 100_000.0, seed: int = 0) -> np.ndarray:
    """Simulated equity curves, shape (n_sims, n_trades + 1), starting at ``start_equity``."""
    p = np.asarray(pnls, dtype=float)
    n = n_trades or len(p)
    rng = np.random.default_rng(seed)
    if mode == "bootstrap":
        draws = p[rng.integers(0, len(p), size=(n_sims, n))]
    elif mode == "shuffle":
        if n > len(p):
            raise ValueError("shuffle mode cannot extend beyond the number of trades")
        draws = np.argsort(rng.random((n_sims, len(p))), axis=1)
        draws = p[draws][:, :n]
    else:
        raise ValueError("mode must be 'bootstrap' or 'shuffle'")
    eq = start_equity + np.cumsum(draws, axis=1)
    return np.concatenate([np.full((n_sims, 1), start_equity), eq], axis=1)


def confidence_bands(paths, quantiles=QUANTILES) -> dict[float, np.ndarray]:
    """Pointwise quantile bands across simulated paths."""
    return {q: np.quantile(paths, q, axis=0) for q in quantiles}


def summarize_paths(paths, loss_threshold: float = 0.0, dd_threshold: float = 0.2) -> dict:
    """Probability of finishing below ``start * (1 - loss_threshold)`` and of a drawdown above ``dd_threshold``."""
    start = paths[:, 0]
    final = paths[:, -1]
    peak = np.maximum.accumulate(paths, axis=1)
    max_dd = ((peak - paths) / peak).max(axis=1)
    return {"n_sims": len(paths), "p_final_below_start": float((final < start * (1 - loss_threshold)).mean()),
            "p_max_dd_above_threshold": float((max_dd > dd_threshold).mean()), "dd_threshold": dd_threshold,
            "final_equity_q05_q50_q95": [float(x) for x in np.quantile(final, [0.05, 0.5, 0.95])],
            "max_dd_q50_q95": [float(x) for x in np.quantile(max_dd, [0.5, 0.95])]}
