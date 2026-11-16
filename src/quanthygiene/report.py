"""One-command HTML tearsheet from an equity, returns or trades CSV.

Answers: what does this track record look like, how uncertain is it, and how fragile is it?
The page is self-contained (charts are embedded PNGs).
"""

from __future__ import annotations

import base64
import html
import io
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import _io, costs, robustness  # noqa: E402


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _fig(w=6.2, h=3.0):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig, ax


def chart_equity_drawdown(equity: np.ndarray):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.2, 4.2), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
    a1.plot(equity, color="#1f4e79", lw=1.3)
    a1.set_title("Equity")
    a2.fill_between(range(len(equity)), -robustness.drawdown_series(equity) * 100, color="#b03a2e", alpha=0.6)
    a2.set_title("Drawdown (%)", fontsize=9)
    for a in (a1, a2):
        a.grid(alpha=0.25)
    return fig


def chart_hist(samples, observed, title, xlabel, pct=False):
    fig, ax = _fig()
    s = np.asarray(samples) * (100 if pct else 1)
    ax.hist(s, bins=50, color="#7fa6c9")
    ax.axvline(observed * (100 if pct else 1), color="#b03a2e", lw=2, label="observed")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.legend()
    return fig


def chart_cost_curve(sens: pd.DataFrame):
    fig, ax = _fig()
    ax.plot(sens["cost_multiplier"], sens["total"], marker="o", color="#1f4e79")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("Net total P&L vs cost multiplier")
    ax.set_xlabel("cost multiplier (0 = gross)")
    return fig


def _stats(r: np.ndarray, ppy: float) -> dict:
    eq = np.cumprod(1 + r)
    years = len(r) / ppy
    dd = robustness.max_drawdown(np.concatenate([[1.0], eq]))
    cagr = eq[-1] ** (1 / years) - 1 if years > 0 and eq[-1] > 0 else float("nan")
    return {"Periods": len(r), "Total return": f"{eq[-1] - 1:.2%}", "CAGR": f"{cagr:.2%}",
            "Volatility (ann.)": f"{r.std(ddof=1) * math.sqrt(ppy):.2%}",
            "Sharpe (ann.)": f"{robustness.sharpe(r, ppy):.2f}", "Max drawdown": f"{dd:.2%}",
            "Calmar": f"{cagr / dd:.2f}" if dd > 0 else "n/a"}


def build_tearsheet(df: pd.DataFrame, title: str = "Tearsheet", periods_per_year: float = 252,
                    start_equity: float = 100_000.0, seed: int = 0) -> str:
    """Return the tearsheet as an HTML string."""
    kind = _io.detect_kind(df)
    r = _io.returns_from_table(df, start_equity).to_numpy()
    equity = np.concatenate([[1.0], np.cumprod(1 + r)])
    sections = [("Equity and drawdown", _png(chart_equity_drawdown(equity)))]
    ci = robustness.bootstrap_ci(r, "sharpe", periods_per_year=periods_per_year, block=1, seed=seed)
    stats = _stats(r, periods_per_year)
    stats["Sharpe 95% CI"] = f"{ci['ci_low']:.2f} to {ci['ci_high']:.2f}"
    dd = robustness.drawdown_distribution(r, n_sims=2000, seed=seed)
    observed = robustness.max_drawdown(equity)
    sections.append(("Drawdown if the same returns arrived in another order",
                     _png(chart_hist(dd["samples"], observed, "Bootstrapped max drawdown", "%", True))))
    if kind == "trades":
        pnl = df[[c for c in df.columns if c.lower() == "pnl"][0]].to_numpy(dtype=float)
        mc = robustness.reshuffle_mc(pnl, start_equity, 2000, seed)
        sections.append(("Trade-order reshuffle",
                         _png(chart_hist(mc["samples"], mc["observed_max_dd"],
                                         "Max drawdown, shuffled trades", "%", True))))
        stats["Trade order percentile of observed DD"] = f"{mc['observed_percentile']:.0%}"
        conc = costs.profit_concentration(pnl)
        stats[f"Top {conc['top']} trades' share of gross profit"] = f"{conc['top_share_of_gross_profit']:.0%}"
        tr = pd.DataFrame({"pnl": pnl})
        sections.append(("Cost sensitivity (illustrative base model)",
                         _png(chart_cost_curve(costs.sensitivity(tr, costs.SCENARIOS['base'])))))
    rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>" for k, v in stats.items())
    figs = "".join(f"<h2>{html.escape(t)}</h2><img src='data:image/png;base64,{b}'>" for t, b in sections)
    return (f"<!doctype html><meta charset='utf-8'><title>{html.escape(title)}</title><style>"
            "body{font-family:system-ui,sans-serif;max-width:760px;margin:2em auto;padding:0 1em;color:#222}"
            "table{border-collapse:collapse}td{padding:4px 14px 4px 0;border-bottom:1px solid #eee}"
            "img{max-width:100%}small{color:#666}</style>"
            f"<h1>{html.escape(title)}</h1><table>{rows}</table>{figs}"
            "<p><small>Research diagnostics only. Not investment advice; no performance claim.</small></p>")
