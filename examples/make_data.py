"""Generate synthetic data for the examples (no real market or trade data)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 750
dates = pd.bdate_range("2021-01-01", periods=n)

ret = rng.normal(0.0007, 0.008, n)
pd.DataFrame({"date": dates, "returns": ret}).to_csv("examples/synthetic_returns.csv", index=False)

pnl = np.where(rng.random(300) < 0.45, rng.normal(180, 60, 300), -rng.normal(100, 30, 300))
risk = rng.uniform(80, 120, 300).round(2)
exit_t = pd.bdate_range("2021-01-01", periods=300) + pd.to_timedelta(rng.integers(1, 20, 300), unit="h")
entry_t = exit_t - pd.to_timedelta(rng.integers(1, 48, 300), unit="h")
mfe = np.where(pnl > 0, pnl * rng.uniform(1.1, 1.8, 300), rng.uniform(5, 40, 300)).round(2)
mae = np.where(pnl > 0, rng.uniform(5, 60, 300), -pnl * rng.uniform(1.0, 1.3, 300)).round(2)
trades = pd.DataFrame({"entry_time": entry_t, "exit_time": exit_t, "pnl": pnl.round(2), "risk": risk,
                       "mae": mae, "mfe": mfe, "units": rng.integers(1, 4, 300),
                       "hold_days": rng.integers(0, 3, 300)})
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

# eight synthetic strategy return streams driven by two latent factors (for clustering)
f1, f2 = rng.normal(0, 0.006, n), rng.normal(0, 0.006, n)
strat = {f"S{i}": (f1 if i < 4 else f2) * (1 + 0.2 * i) + rng.normal(0, 0.004, n) for i in range(8)}
pd.DataFrame(strat).to_csv("examples/synthetic_strategies.csv", index=False)
