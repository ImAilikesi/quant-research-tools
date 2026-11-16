"""Dark-theme visuals and animated GIFs for research results.

Answers: how do I *see* the uncertainty? Fan charts of simulated equity, animated path
simulations, an animated overfitting race (best of N variants in-sample vs out-of-sample), an
underwater plot, a monthly-returns heatmap and a 3D surface of rule-breach probability.
Animations are written with Pillow (installed with matplotlib); no extra dependencies.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import animation  # noqa: E402

from . import montecarlo, risk  # noqa: E402

BG, FG, GRID = "#0d1117", "#c9d1d9", "#30363d"
ACCENT, WARN, GOOD = "#58a6ff", "#f85149", "#3fb950"


def apply_theme() -> None:
    """Set the dark matplotlib theme used by every function in this module."""
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": GRID, "axes.labelcolor": FG, "text.color": FG,
        "xtick.color": FG, "ytick.color": FG, "grid.color": GRID, "axes.grid": True,
        "grid.alpha": 0.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "font.size": 10,
    })


def _bands(ax, x, paths, alpha=1.0):
    q = {k: np.quantile(paths, k, axis=0) for k in (0.05, 0.25, 0.5, 0.75, 0.95)}
    ax.fill_between(x, q[0.05], q[0.95], color=ACCENT, alpha=0.15 * alpha, lw=0)
    ax.fill_between(x, q[0.25], q[0.75], color=ACCENT, alpha=0.3 * alpha, lw=0)
    ax.plot(x, q[0.5], color=ACCENT, lw=2)


def plot_fan(paths, title: str = "Simulated equity paths", n_lines: int = 40):
    """Fan chart: a few sample paths over 5-95 and 25-75 percent bands and the median."""
    apply_theme()
    paths = np.asarray(paths)
    x = np.arange(paths.shape[1])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for p in paths[:n_lines]:
        ax.plot(x, p, color=FG, alpha=0.08, lw=0.8)
    _bands(ax, x, paths)
    ax.axhline(paths[0, 0], color=GRID, lw=1)
    ax.set_title(title)
    ax.set_xlabel("trade")
    return fig


def plot_underwater(returns, title: str = "Underwater (drawdown)"):
    """Filled drawdown curve over time."""
    apply_theme()
    r = pd.Series(np.asarray(returns, dtype=float))
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1) * 100
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.fill_between(dd.index, dd, 0, color=WARN, alpha=0.6, lw=0)
    ax.set_title(title)
    ax.set_ylabel("%")
    return fig


def plot_monthly_heatmap(returns: pd.Series, title: str = "Monthly returns (%)"):
    """Year x month heatmap of compounded returns; ``returns`` needs a DatetimeIndex."""
    apply_theme()
    m = (1 + returns).groupby([returns.index.year, returns.index.month]).prod() - 1
    tbl = m.unstack().reindex(columns=range(1, 13)) * 100
    fig, ax = plt.subplots(figsize=(7, 0.9 + 0.5 * len(tbl)))
    lim = np.nanmax(np.abs(tbl.to_numpy())) or 1.0
    im = ax.imshow(tbl, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(12), list("JFMAMJJASOND"))
    ax.set_yticks(range(len(tbl)), tbl.index)
    ax.grid(False)
    for i in range(tbl.shape[0]):
        for j in range(12):
            v = tbl.iat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color="black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03)
    return fig


def plot_breach_surface(returns, rules: risk.Rules | None = None, scales=(0.5, 1, 1.5, 2, 3),
                        horizons=(10, 20, 40, 60), n_paths: int = 1500, seed: int = 0):
    """3D surface of breach probability against risk scale and horizon under ``rules``."""
    apply_theme()
    rules = rules or risk.Rules()
    r = np.asarray(returns, dtype=float)
    z = np.array([[risk.simulate_rules(r * s, h, rules, n_paths, seed=seed)["p_breach"] for s in scales]
                  for h in horizons])
    sx, hy = np.meshgrid(scales, horizons)
    fig = plt.figure(figsize=(6.4, 4.6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(sx, hy, z, cmap="inferno", edgecolor="none", alpha=0.95)
    ax.set_xlabel("risk scale")
    ax.set_ylabel("horizon")
    ax.set_zlabel("P(breach)")
    ax.set_title("Rule-breach probability")
    ax.xaxis.set_pane_color((0, 0, 0, 0))
    ax.yaxis.set_pane_color((0, 0, 0, 0))
    ax.zaxis.set_pane_color((0, 0, 0, 0))
    return fig


def animate_fan(pnls, path: str, n_trades: int = 120, n_sims: int = 400, frames: int = 45,
                start_equity: float = 100_000.0, seed: int = 0, fps: int = 12) -> str:
    """GIF: simulated equity paths from ``pnls`` unfolding trade by trade, with growing bands."""
    apply_theme()
    paths = montecarlo.equity_paths(pnls, n_trades, n_sims, "bootstrap", start_equity, seed)
    steps = np.linspace(2, n_trades, frames).astype(int)
    lo, hi = np.quantile(paths, 0.01), np.quantile(paths, 0.99)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))

    def draw(k):
        ax.clear()
        t = steps[k]
        x = np.arange(t + 1)
        for p in paths[:30]:
            ax.plot(x, p[: t + 1], color=FG, alpha=0.12, lw=0.8)
        _bands(ax, x, paths[:, : t + 1])
        ax.axhline(start_equity, color=GRID, lw=1)
        ax.set_xlim(0, n_trades)
        ax.set_ylim(lo, hi)
        ax.set_title(f"Monte Carlo equity paths: trade {t}")
        ax.set_xlabel("trade")

    anim = animation.FuncAnimation(fig, draw, frames=len(steps))
    anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    return path


def animate_overfitting(variants, path: str, frames: int = 40, fps: int = 8) -> str:
    """GIF: pick the best of the first k variants in the first half of the data and watch it fade in the second.

    ``variants``: (T, N) per-period returns of N variants. With skill-less variants the in-sample
    equity of the winner climbs as k grows, while its out-of-sample equity does not follow.
    """
    apply_theme()
    m = np.asarray(variants, dtype=float)
    half = len(m) // 2
    ins, oos = m[:half], m[half:]
    score = ins.mean(axis=0) / ins.std(axis=0, ddof=1)
    ks = np.unique(np.linspace(1, m.shape[1], frames).astype(int))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    curves = []
    for k in ks:
        best = int(np.argmax(score[:k]))
        a = np.cumsum(ins[:, best]) * 100
        curves.append(np.concatenate([a, a[-1] + np.cumsum(oos[:, best]) * 100]))
    top, bottom = max(c.max() for c in curves) + 5, min(c.min() for c in curves) - 5

    def draw(i):
        ax.clear()
        k = ks[i]
        best = int(np.argmax(score[:k]))
        a, b = np.cumsum(ins[:, best]) * 100, np.cumsum(oos[:, best]) * 100
        ax.plot(range(half), a, color=GOOD, lw=2, label="in-sample")
        ax.plot(range(half, len(m)), a[-1] + b, color=WARN, lw=2, label="out-of-sample")
        ax.axvline(half, color=GRID)
        ax.set_ylim(bottom, top)
        ax.set_title(f"Best of {k} skill-less variants")
        ax.set_ylabel("cumulative return (%)")
        ax.legend(loc="lower left")

    anim = animation.FuncAnimation(fig, draw, frames=len(ks))
    anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    return path
