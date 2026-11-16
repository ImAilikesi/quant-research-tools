"""CSV loading helpers shared by the CLI and the report module."""

from __future__ import annotations

import numpy as np
import pandas as pd


def read_table(path: str) -> pd.DataFrame:
    """Read a CSV; parse a ``date``/``time``/``timestamp``/``exit_time`` column as the index if present."""
    df = pd.read_csv(path)
    for name in ("date", "time", "timestamp", "datetime", "exit_time"):
        if name in df.columns:
            df[name] = pd.to_datetime(df[name], errors="coerce")
            if name != "exit_time":
                df = df.set_index(name)
            break
    return df


def detect_kind(df: pd.DataFrame) -> str:
    """Return ``'trades'``, ``'equity'`` or ``'returns'`` from the column names."""
    cols = {c.lower() for c in df.columns}
    if "pnl" in cols:
        return "trades"
    if "equity" in cols:
        return "equity"
    if cols & {"return", "returns", "ret"}:
        return "returns"
    raise ValueError("CSV needs a 'pnl', 'equity' or 'returns' column")


def returns_from_table(df: pd.DataFrame, start_equity: float = 100_000.0) -> pd.Series:
    """Per-period simple returns from an equity, returns or trades table."""
    kind = detect_kind(df)
    lower = {c.lower(): c for c in df.columns}
    if kind == "equity":
        eq = df[lower["equity"]].astype(float)
        return eq.pct_change().dropna()
    if kind == "returns":
        col = lower.get("returns") or lower.get("return") or lower["ret"]
        return df[col].astype(float).dropna()
    pnl = df[lower["pnl"]].astype(float).to_numpy()
    equity = start_equity + np.cumsum(pnl)
    prev = np.concatenate([[start_equity], equity[:-1]])
    return pd.Series(pnl / prev)
