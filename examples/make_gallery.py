"""Write the animated and 3D visuals used in the README to docs/img/ (synthetic data only)."""
from pathlib import Path

import pandas as pd

from quanthygiene import montecarlo, risk, viz

here = Path(__file__).parent
out = here.parent / "docs" / "img"
out.mkdir(parents=True, exist_ok=True)
trades = pd.read_csv(here / "synthetic_trades.csv")
rets = pd.read_csv(here / "synthetic_returns.csv", parse_dates=["date"]).set_index("date")["returns"]
variants = pd.read_csv(here / "synthetic_variants.csv").to_numpy()

viz.animate_fan(trades["pnl"].to_numpy(), str(out / "mc_fan.gif"))
viz.animate_overfitting(variants, str(out / "overfitting_race.gif"))
for name, fig in {
    "fan.png": viz.plot_fan(montecarlo.equity_paths(trades["pnl"], 150, 1000)),
    "underwater.png": viz.plot_underwater(rets.to_numpy()),
    "monthly_heatmap.png": viz.plot_monthly_heatmap(rets),
    "breach_surface.png": viz.plot_breach_surface(rets.to_numpy(), risk.Rules(0.10, None, 0.05, 0.10)),
}.items():
    fig.savefig(out / name, dpi=130, bbox_inches="tight")
print("gallery written to", out)
