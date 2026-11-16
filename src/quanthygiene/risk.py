"""Risk-rule simulation: loss limits, trailing drawdown, profit targets, risk of ruin.

Answers: with this return distribution, how likely am I to breach a static loss limit, a trailing
drawdown limit or a daily loss limit (typical of prop-firm evaluations) before reaching a profit
target? What is my risk of ruin at a given risk per trade?

Rule limits are fractions of initial capital and P&L is additive on initial capital.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Rules:
    """Evaluation rules, as fractions of initial capital. ``None`` disables a rule.

    ``max_loss``: static floor below initial capital. ``max_trailing_dd``: drop from the high-water
    mark (starting at 1.0). ``daily_loss``: drop from the equity at the start of the day.
    ``profit_target``: gain that counts as a pass. ``periods_per_day``: returns per trading day.
    """

    max_loss: float | None = 0.10
    max_trailing_dd: float | None = None
    daily_loss: float | None = 0.05
    profit_target: float | None = 0.10
    periods_per_day: int = 1


def simulate_rules(returns, horizon: int, rules: Rules, n_paths: int = 20000,
                   block: int = 1, seed: int = 0) -> dict:
    """Bootstrap ``returns`` (fractions of initial capital per period) and apply ``rules``.

    Outcome per path is whichever comes first: pass (target reached), breach (any limit hit) or
    timeout. A breach in the same period as a pass counts as a breach.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    rng = np.random.default_rng(seed)
    if block <= 1:
        idx = rng.integers(0, n, size=(n_paths, horizon))
    else:
        nb = -(-horizon // block)
        starts = rng.integers(0, n, size=(n_paths, nb))
        idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(n_paths, -1)[:, :horizon]
    eq0 = np.concatenate([np.ones((n_paths, 1)), 1.0 + np.cumsum(r[idx], axis=1)], axis=1)
    eq = eq0[:, 1:]
    breach = np.zeros_like(eq, dtype=bool)
    if rules.max_loss is not None:
        breach |= eq < 1.0 - rules.max_loss
    if rules.max_trailing_dd is not None:
        breach |= np.maximum.accumulate(eq0, axis=1)[:, 1:] - eq > rules.max_trailing_dd
    if rules.daily_loss is not None:
        day_start = (np.arange(horizon) // rules.periods_per_day) * rules.periods_per_day
        breach |= eq - eq0[:, day_start] < -rules.daily_loss
    hit = np.zeros_like(breach)
    if rules.profit_target is not None:
        hit = eq >= 1.0 + rules.profit_target
    never = horizon + 1
    t_breach = np.where(breach.any(axis=1), breach.argmax(axis=1), never)
    t_pass = np.where(hit.any(axis=1), hit.argmax(axis=1), never)
    breached = t_breach <= t_pass
    breached &= t_breach < never
    passed = (t_pass < t_breach)
    timeout = ~(breached | passed)
    done = np.where(passed, t_pass, np.where(breached, t_breach, horizon)) + 1
    return {"n_paths": n_paths, "horizon": horizon, "p_breach": float(breached.mean()),
            "p_pass": float(passed.mean()), "p_timeout": float(timeout.mean()),
            "median_periods_to_pass": float(np.median(done[passed])) if passed.any() else None,
            "final_equity_q05_q50_q95": [float(x) for x in np.quantile(eq[:, -1], [.05, .5, .95])]}


def risk_of_ruin(r_multiples, risk_per_trade: float, ruin_drawdown: float = 0.5,
                 n_trades: int = 1000, n_paths: int = 10000, seed: int = 0) -> dict:
    """Probability that compounding drawdown reaches ``ruin_drawdown`` within ``n_trades``.

    ``r_multiples``: trade results in units of initial risk (a 2R winner is +2, a stop is -1).
    Each trade changes equity by ``risk_per_trade * R`` (fraction of current equity).
    """
    r = np.asarray(r_multiples, dtype=float)
    rng = np.random.default_rng(seed)
    steps = 1.0 + risk_per_trade * r[rng.integers(0, len(r), size=(n_paths, n_trades))]
    steps = np.maximum(steps, 0.0)
    eq = np.cumprod(steps, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((n_paths, 1)), eq], axis=1), axis=1)[:, 1:]
    ruined = ((1 - eq / peak).max(axis=1) >= ruin_drawdown)
    return {"risk_per_trade": risk_per_trade, "ruin_drawdown": ruin_drawdown,
            "n_trades": n_trades, "p_ruin": float(ruined.mean()),
            "expectancy_R": float(r.mean())}
