"""Run the library on the synthetic example data and write the README figures to docs/img/."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quanthygiene import clustering, costs, montecarlo, regimes, report, risk, robustness, rolling

here = Path(__file__).parent
out = here.parent / "docs" / "img"
out.mkdir(parents=True, exist_ok=True)
trades = pd.read_csv(here / "synthetic_trades.csv")
returns = pd.read_csv(here / "synthetic_returns.csv")["returns"].to_numpy()
variants = pd.read_csv(here / "synthetic_variants.csv").to_numpy()

# 1. Overfitting: the best of 40 skill-less variants looks good in-sample, not out of sample
sr = [robustness.sharpe(variants[:, j], 252) for j in range(variants.shape[1])]
best = int(np.argmax(sr))
dsr = robustness.deflated_sharpe_ratio(variants[:, best], n_trials=variants.shape[1])
pbo = robustness.pbo_cscv(variants, 16)
fig, ax = plt.subplots(figsize=(6.2, 3.2))
ax.hist(sr, bins=15, color="#7fa6c9")
ax.axvline(sr[best], color="#b03a2e", lw=2, label=f"best variant (SR {sr[best]:.2f})")
ax.set_title(f"40 skill-less variants: DSR={dsr['dsr']:.2f}, PBO={pbo['pbo']:.2f}")
ax.set_xlabel("annualised Sharpe"); ax.legend(); ax.grid(alpha=0.25)
fig.savefig(out / "overfitting.png", dpi=130, bbox_inches="tight")

# 2. Cost sensitivity curve
sens = costs.sensitivity(trades, costs.SCENARIOS["base"])
fig = report.chart_cost_curve(sens)
fig.savefig(out / "cost_sensitivity.png", dpi=130, bbox_inches="tight")

# 3. Breach probability against volatility scaling under prop-style rules
scales = [0.5, 0.75, 1, 1.5, 2, 3]
rules = risk.Rules(max_loss=0.10, max_trailing_dd=None, daily_loss=0.05, profit_target=0.10)
res = [risk.simulate_rules(returns * s, 60, rules, 5000) for s in scales]
fig, ax = plt.subplots(figsize=(6.2, 3.2))
ax.plot(scales, [r["p_pass"] for r in res], marker="o", label="pass")
ax.plot(scales, [r["p_breach"] for r in res], marker="o", label="breach")
ax.set_xlabel("risk scale (x synthetic returns)"); ax.set_title("Pass vs breach probability, 60 periods")
ax.legend(); ax.grid(alpha=0.25)
fig.savefig(out / "breach_probability.png", dpi=130, bbox_inches="tight")
print("figures written to", out)

# 4. Monte Carlo equity bands from the trade list
paths = montecarlo.equity_paths(trades["pnl"], 150, 2000)
bands = montecarlo.confidence_bands(paths)
fig, ax = plt.subplots(figsize=(6.2, 3.2))
ax.fill_between(range(151), bands[0.05], bands[0.95], alpha=0.25, label="5-95%")
ax.fill_between(range(151), bands[0.25], bands[0.75], alpha=0.4, label="25-75%")
ax.plot(bands[0.5], color="#1f4e79", label="median")
ax.set_title("Bootstrapped equity paths, next 150 trades"); ax.set_xlabel("trade"); ax.legend(); ax.grid(alpha=0.25)
fig.savefig(out / "mc_bands.png", dpi=130, bbox_inches="tight")

# 5. Rolling Sharpe and volatility-regime Sharpe
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
a1.plot(rolling.rolling_sharpe(returns, 120), color="#1f4e79"); a1.axhline(0, color="k", lw=0.8)
a1.set_title("Rolling Sharpe (120 periods)"); a1.grid(alpha=0.25)
reg = regimes.regime_performance(returns)
a2.bar(["calm", "mid", "turbulent"], reg["sharpe"], color="#7fa6c9")
a2.set_title("Sharpe by volatility tercile"); a2.grid(alpha=0.25)
fig.savefig(out / "rolling_regimes.png", dpi=130, bbox_inches="tight")

# 6. Strategy correlation matrix ordered by cluster
strat = pd.read_csv(here / "synthetic_strategies.csv")
res = clustering.cluster(strat, 2)
order = [n for g in res["clusters"] for n in g]
fig, ax = plt.subplots(figsize=(4.6, 4))
im = ax.imshow(strat[order].corr(), vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(order)), order); ax.set_yticks(range(len(order)), order)
ax.set_title("Correlation, grouped by cluster"); fig.colorbar(im)
fig.savefig(out / "clusters.png", dpi=130, bbox_inches="tight")
