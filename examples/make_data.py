"""Generate synthetic data for the examples (no real market or trade data)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 750
dates = pd.bdate_range("2021-01-01", periods=n)

ret = rng.normal(0.0007, 0.008, n)
pd.DataFrame({"date": dates, "returns": ret}).to_csv("examples/synthetic_returns.csv", index=False)

pnl = np.where(rng.random(300) < 0.45, rng.normal(180, 60, 300), -rng.normal(100, 30, 300))
trades = pd.DataFrame({"exit_time": pd.bdate_range("2021-01-01", periods=300), "pnl": pnl.round(2),
                       "units": rng.integers(1, 4, 300), "hold_days": rng.integers(0, 3, 300)})
trades.to_csv("examples/synthetic_trades.csv", index=False)

variants = rng.normal(0.0, 0.01, (n, 40))  # 40 skill-less variants -> overfitting demo
pd.DataFrame(variants, columns=[f"v{i}" for i in range(40)]).to_csv("examples/synthetic_variants.csv", index=False)

ohlc = []
c = 100.0
for _ in range(300):
    o = c * (1 + rng.normal(0, 0.003))
    path = o * np.exp(np.cumsum(rng.normal(0, 0.004, 20)))
    c = path[-1]
    ohlc.append((o, max(path.max(), o), min(path.min(), o), c))
pd.DataFrame(ohlc, columns=["open", "high", "low", "close"]).to_csv("examples/synthetic_ohlc.csv", index=False)

# synthetic FX levels: independent random walks plus a shared USD factor
usd = np.cumsum(rng.normal(0, 0.004, n))
cols = {}
for pair, base, inv in (("EURUSD", 1.10, False), ("GBPUSD", 1.27, False), ("AUDUSD", 0.68, False), ("USDJPY", 140.0, True)):
    x = np.cumsum(rng.normal(0, 0.003, n)) + (-usd if inv else usd)
    cols[pair] = base * np.exp(x)
pd.DataFrame(cols, index=dates).rename_axis("date").to_csv("examples/synthetic_fx.csv")
