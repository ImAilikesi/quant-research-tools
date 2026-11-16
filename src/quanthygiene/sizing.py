"""Position sizing: fixed-fractional, (fractional) Kelly, volatility targeting.

Answers: how many units for a given risk budget, what bet fraction does an edge imply, and how
much exposure keeps portfolio volatility on target.
"""

from __future__ import annotations

import math

import numpy as np


def fixed_fractional(equity: float, risk_fraction: float, stop_distance: float,
                     value_per_point: float = 1.0) -> float:
    """Units such that hitting the stop loses ``risk_fraction`` of ``equity``.

    ``stop_distance`` is in price points; ``value_per_point`` is account currency per point per unit.
    """
    if stop_distance <= 0 or value_per_point <= 0:
        raise ValueError("stop_distance and value_per_point must be positive")
    return equity * risk_fraction / (stop_distance * value_per_point)


def kelly_fraction(win_prob: float, payoff_ratio: float) -> float:
    """Kelly fraction for a binary bet paying ``payoff_ratio`` per unit risked. Negative = no bet."""
    if payoff_ratio <= 0:
        raise ValueError("payoff_ratio must be positive")
    return win_prob - (1 - win_prob) / payoff_ratio


def kelly_from_returns(returns) -> float:
    """Continuous-approximation Kelly fraction mean/variance from per-period returns."""
    r = np.asarray(returns, dtype=float)
    var = r.var(ddof=1)
    return float(r.mean() / var) if var > 0 else 0.0


def fractional_kelly(full_kelly: float, fraction: float = 0.5) -> float:
    """Scale a Kelly fraction down (half-Kelly is common); never below zero."""
    return max(0.0, full_kelly * fraction)


def vol_target_exposure(target_vol: float, asset_vol: float, max_leverage: float = 3.0) -> float:
    """Exposure as a multiple of equity so that expected volatility equals ``target_vol``.

    Both volatilities must be on the same (e.g. annualised) basis.
    """
    if asset_vol <= 0:
        raise ValueError("asset_vol must be positive")
    return min(target_vol / asset_vol, max_leverage)


def realized_vol(returns, periods_per_year: float = 252.0) -> float:
    """Annualised standard deviation of per-period returns."""
    return float(np.std(np.asarray(returns, dtype=float), ddof=1) * math.sqrt(periods_per_year))
