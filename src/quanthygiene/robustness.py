"""Backtest robustness: does an apparent edge survive scrutiny?

Answers: how uncertain is my Sharpe (bootstrap CI), is it still significant after trying many
variants (Deflated Sharpe Ratio), how likely is my best variant to be a fluke (PBO via CSCV),
and how bad could drawdown have been with a different trade order (reshuffle Monte Carlo).

References: Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio";
Bailey, Borwein, Lopez de Prado, Zhu (2017) "The Probability of Backtest Overfitting".
"""

from __future__ import annotations

import math
from itertools import combinations
from statistics import NormalDist

import numpy as np

_N = NormalDist()
_EULER = 0.5772156649015329


def sharpe(returns, periods_per_year: float = 1.0) -> float:
    """Annualised Sharpe ratio of per-period returns (risk-free rate taken as zero)."""
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else 0.0


def _resample_indices(rng, n: int, n_boot: int, block: int) -> np.ndarray:
    if block <= 1:
        return rng.integers(0, n, size=(n_boot, n))
    n_blocks = -(-n // block)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)) % n
    return idx.reshape(n_boot, -1)[:, :n]


def bootstrap_ci(returns, stat: str = "sharpe", n_boot: int = 5000, block: int = 1,
                 alpha: float = 0.05, periods_per_year: float = 1.0, seed: int = 0) -> dict:
    """Percentile bootstrap CI for ``stat`` in {'sharpe', 'mean', 'total'}.

    ``block > 1`` uses a circular block bootstrap, which keeps short-range autocorrelation.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        raise ValueError("need at least 2 observations")
    rng = np.random.default_rng(seed)
    s = r[_resample_indices(rng, len(r), n_boot, block)]
    if stat == "sharpe":
        sd = s.std(axis=1, ddof=1)
        vals = np.where(sd > 0, s.mean(axis=1) / np.where(sd > 0, sd, 1), 0.0)
        vals = vals * math.sqrt(periods_per_year)
        point = sharpe(r, periods_per_year)
    elif stat == "mean":
        vals, point = s.mean(axis=1), float(r.mean())
    elif stat == "total":
        vals, point = np.prod(1 + s, axis=1) - 1, float(np.prod(1 + r) - 1)
    else:
        raise ValueError("stat must be 'sharpe', 'mean' or 'total'")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"stat": stat, "point": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "alpha": alpha, "n_boot": n_boot, "block": block,
            "p_stat_le_zero": float((vals <= 0).mean())}


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum per-period Sharpe among ``n_trials`` skill-less trials."""
    if n_trials < 2:
        return 0.0
    g = _EULER
    return math.sqrt(sr_variance) * ((1 - g) * _N.inv_cdf(1 - 1 / n_trials)
                                     + g * _N.inv_cdf(1 - 1 / (n_trials * math.e)))


def deflated_sharpe_ratio(returns, n_trials: int, trial_sharpe_var: float | None = None) -> dict:
    """Deflated Sharpe Ratio: probability the true Sharpe is positive after selection bias.

    ``returns``: per-period returns of the selected strategy. ``n_trials``: how many variants were
    tried. ``trial_sharpe_var``: variance of the per-period Sharpe across trials; if omitted the
    iid-null value 1/(T-1) is used. Sharpe values here are per period, not annualised.
    """
    r = np.asarray(returns, dtype=float)
    t = len(r)
    if t < 3:
        raise ValueError("need at least 3 observations")
    sd = r.std(ddof=1)
    sr = float(r.mean() / sd)
    z = (r - r.mean()) / sd
    skew, kurt = float((z ** 3).mean()), float((z ** 4).mean())
    var = trial_sharpe_var if trial_sharpe_var is not None else 1.0 / (t - 1)
    sr0 = expected_max_sharpe(n_trials, var)
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    stat = (sr - sr0) * math.sqrt(t - 1) / math.sqrt(max(denom, 1e-12))
    return {"sharpe_per_period": sr, "sr_benchmark": sr0, "dsr": _N.cdf(stat),
            "n_trials": n_trials, "n_obs": t, "skew": skew, "kurtosis": kurt}


def pbo_cscv(returns_matrix, n_splits: int = 16, max_combos: int = 5000, seed: int = 0) -> dict:
    """Probability of Backtest Overfitting by combinatorially symmetric cross-validation.

    ``returns_matrix``: shape (T, N), one column per tried strategy variant. The rows are cut into
    ``n_splits`` blocks (even); every way of picking half of them as in-sample is scored: the best
    in-sample variant's out-of-sample rank gives a logit. PBO is the share of splits where that
    variant lands at or below the out-of-sample median.
    """
    m = np.asarray(returns_matrix, dtype=float)
    t, n = m.shape
    if n < 2:
        raise ValueError("need at least 2 strategy variants")
    if n_splits % 2 or n_splits < 4 or t < n_splits * 2:
        raise ValueError("n_splits must be even, >= 4, and at most T/2")
    size = t // n_splits
    blocks = m[: size * n_splits].reshape(n_splits, size, n)
    s1, s2 = blocks.sum(axis=1), (blocks ** 2).sum(axis=1)

    def score(sel):
        cnt = len(sel) * size
        mean = s1[sel].sum(axis=0) / cnt
        var = np.maximum(s2[sel].sum(axis=0) / cnt - mean ** 2, 1e-18)
        return mean / np.sqrt(var)

    combos = list(combinations(range(n_splits), n_splits // 2))
    rng = np.random.default_rng(seed)
    if len(combos) > max_combos:
        combos = [combos[i] for i in rng.choice(len(combos), max_combos, replace=False)]
    logits, degr = [], []
    allb = set(range(n_splits))
    for js in combos:
        ins, oos = list(js), sorted(allb - set(js))
        sis, soos = score(ins), score(oos)
        best = int(np.argmax(sis))
        rank = int((soos <= soos[best]).sum())
        w = rank / (n + 1)
        logits.append(math.log(w / (1 - w)))
        degr.append((sis[best], soos[best]))
    logits = np.array(logits)
    d = np.array(degr)
    return {"pbo": float((logits <= 0).mean()), "n_combinations": len(combos),
            "n_variants": n, "mean_logit": float(logits.mean()),
            "mean_in_sample_sharpe": float(d[:, 0].mean()),
            "mean_out_of_sample_sharpe": float(d[:, 1].mean())}


def drawdown_series(equity) -> np.ndarray:
    """Fractional drawdown at each point of an equity curve (0 at new highs)."""
    e = np.asarray(equity, dtype=float)
    return 1.0 - e / np.maximum.accumulate(e)


def max_drawdown(equity) -> float:
    """Maximum fractional drawdown of an equity curve."""
    return float(drawdown_series(equity).max())


def reshuffle_mc(pnls, start_equity: float = 100_000.0, n_sims: int = 5000, seed: int = 0) -> dict:
    """Shuffle trade order to see how much of the realised drawdown was path luck.

    The total P&L is identical in every shuffle; only the order changes. Returns the distribution
    of max drawdown and where the observed drawdown sits in it.
    """
    p = np.asarray(pnls, dtype=float)
    rng = np.random.default_rng(seed)
    observed = max_drawdown(np.concatenate([[start_equity], start_equity + np.cumsum(p)]))
    dds = np.empty(n_sims)
    for i in range(n_sims):
        eq = start_equity + np.cumsum(rng.permutation(p))
        dds[i] = max_drawdown(np.concatenate([[start_equity], eq]))
    q = np.quantile(dds, [0.05, 0.5, 0.95, 0.99])
    return {"n_trades": len(p), "n_sims": n_sims, "observed_max_dd": observed,
            "dd_q05": float(q[0]), "dd_median": float(q[1]), "dd_q95": float(q[2]),
            "dd_q99": float(q[3]), "observed_percentile": float((dds <= observed).mean()),
            "samples": dds}


def drawdown_distribution(returns, horizon: int | None = None, n_sims: int = 5000,
                          block: int = 1, seed: int = 0) -> dict:
    """Max-drawdown distribution from bootstrapped return paths of length ``horizon``."""
    r = np.asarray(returns, dtype=float)
    h = horizon or len(r)
    rng = np.random.default_rng(seed)
    paths = r[_resample_indices(rng, len(r), n_sims, block)[:, :h]] if h <= len(r) else \
        r[rng.integers(0, len(r), size=(n_sims, h))]
    eq = np.cumprod(1 + paths, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((n_sims, 1)), eq], axis=1), axis=1)[:, 1:]
    dd = (1 - eq / peak).max(axis=1)
    q = np.quantile(dd, [0.05, 0.5, 0.95, 0.99])
    return {"horizon": h, "n_sims": n_sims, "dd_q05": float(q[0]), "dd_median": float(q[1]),
            "dd_q95": float(q[2]), "dd_q99": float(q[3]), "samples": dd}
