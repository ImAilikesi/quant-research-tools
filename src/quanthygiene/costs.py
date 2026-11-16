"""Transaction-cost scenarios and sensitivity for a trades table.

Answers: does the result survive realistic (and worse-than-realistic) spread, commission,
slippage and swap? At what cost multiple does the edge disappear? How concentrated are profits?

Trades need a ``pnl`` column (gross, before the costs modelled here). Optional columns:
``units`` (size per trade, default 1) and ``hold_days`` (default 0). All cost inputs are per unit
of size, in the same currency as ``pnl``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    """Per-unit costs. Slippage is charged on both entry and exit; swap per unit per day held."""

    spread: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    swap_per_day: float = 0.0


SCENARIOS = {
    "none": CostModel(),
    "low": CostModel(spread=0.5, commission=0.2, slippage=0.1),
    "base": CostModel(spread=1.0, commission=0.4, slippage=0.25),
    "stress": CostModel(spread=2.0, commission=0.8, slippage=0.75, swap_per_day=0.05),
}


def trade_costs(trades: pd.DataFrame, model: CostModel, multiplier: float = 1.0) -> pd.Series:
    """Cost of each trade under ``model`` scaled by ``multiplier``."""
    units = trades["units"] if "units" in trades else 1.0
    days = trades["hold_days"] if "hold_days" in trades else 0.0
    per_unit = model.spread + model.commission + 2 * model.slippage + model.swap_per_day * days
    return multiplier * units * per_unit


def summarize(pnl) -> dict:
    """Net-result summary of a P&L series."""
    p = np.asarray(pnl, dtype=float)
    wins, losses = p[p > 0].sum(), -p[p < 0].sum()
    sd = p.std(ddof=1) if len(p) > 1 else 0.0
    return {"trades": len(p), "total": float(p.sum()), "expectancy": float(p.mean()),
            "win_rate": float((p > 0).mean()),
            "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
            "sharpe_per_trade": float(p.mean() / sd) if sd > 0 else 0.0}


def sensitivity(trades: pd.DataFrame, model: CostModel,
                multipliers=(0, 0.5, 1, 1.5, 2, 3, 4)) -> pd.DataFrame:
    """Net results across cost multipliers (0 = gross)."""
    rows = []
    for m in multipliers:
        net = trades["pnl"] - trade_costs(trades, model, m)
        rows.append({"cost_multiplier": m, **summarize(net)})
    return pd.DataFrame(rows)


def breakeven_multiplier(trades: pd.DataFrame, model: CostModel) -> float:
    """Cost multiple at which total net P&L is zero (inf if the model has no cost)."""
    c = trade_costs(trades, model).sum()
    return float(trades["pnl"].sum() / c) if c > 0 else float("inf")


def scenario_table(trades: pd.DataFrame, scenarios: dict | None = None) -> pd.DataFrame:
    """Net results under named cost scenarios."""
    rows = []
    for name, model in (scenarios or SCENARIOS).items():
        rows.append({"scenario": name, **summarize(trades["pnl"] - trade_costs(trades, model))})
    return pd.DataFrame(rows)


def profit_concentration(pnl, top: int = 5) -> dict:
    """Share of gross profit made by the best ``top`` trades, and total without them."""
    p = np.sort(np.asarray(pnl, dtype=float))[::-1]
    gross = p[p > 0].sum()
    k = min(top, len(p))
    return {"top": k, "top_share_of_gross_profit": float(p[:k].sum() / gross) if gross > 0 else 0.0,
            "total_without_top": float(p[k:].sum()), "total": float(p.sum())}
