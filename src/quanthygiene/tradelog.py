"""Trade-log analysis: R-multiples, holding time, streaks, MAE/MFE.

Answers: how are wins and losses distributed, how long are trades held, how long can losing runs
get, and how much of each trade's favourable move was given back?

Required column: ``pnl``. Optional: ``risk`` (currency risked per trade, for R-multiples),
``entry_time`` and ``exit_time`` (holding time), ``mae`` and ``mfe`` (maximum adverse / favourable
excursion in the same currency as ``pnl``, both as positive numbers).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def streaks(pnl) -> dict:
    """Longest winning and losing runs, and the mean run length of each (zeros end a run)."""
    p = np.asarray(pnl, dtype=float)
    out = {}
    for name, cond in (("win", p > 0), ("loss", p < 0)):
        runs, cur = [], 0
        for c in cond:
            if c:
                cur += 1
            elif cur:
                runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
        out[f"max_{name}_streak"] = max(runs, default=0)
        out[f"mean_{name}_streak"] = float(np.mean(runs)) if runs else 0.0
    return out


def r_multiples(trades: pd.DataFrame) -> pd.Series:
    """P&L divided by the risk taken on each trade (needs ``risk``)."""
    return trades["pnl"] / trades["risk"]


def analyze(trades: pd.DataFrame) -> dict:
    """Summary of a trade log; sections appear only when their columns exist."""
    p = trades["pnl"].to_numpy(dtype=float)
    wins, losses = p[p > 0], p[p < 0]
    out = {"trades": len(p), "win_rate": float((p > 0).mean()),
           "avg_win": float(wins.mean()) if len(wins) else 0.0,
           "avg_loss": float(losses.mean()) if len(losses) else 0.0,
           "payoff_ratio": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else None,
           "expectancy": float(p.mean()), **streaks(p)}
    if "risk" in trades:
        r = r_multiples(trades)
        out["expectancy_R"] = float(r.mean())
        out["R_quantiles_05_50_95"] = [float(x) for x in np.quantile(r, [0.05, 0.5, 0.95])]
    if {"entry_time", "exit_time"} <= set(trades.columns):
        hold = pd.to_datetime(trades["exit_time"]) - pd.to_datetime(trades["entry_time"])
        hours = hold.dt.total_seconds() / 3600
        out["holding_hours_median"] = float(hours.median())
        out["holding_hours_mean_winners"] = float(hours[trades["pnl"] > 0].mean())
        out["holding_hours_mean_losers"] = float(hours[trades["pnl"] < 0].mean())
    if {"mae", "mfe"} <= set(trades.columns):
        w = trades["pnl"] > 0
        out["mae_mean_winners"] = float(trades.loc[w, "mae"].mean())
        out["mae_mean_losers"] = float(trades.loc[~w, "mae"].mean())
        gain = trades.loc[w]
        out["winner_capture_ratio"] = float((gain["pnl"] / gain["mfe"]).mean())
    return out
