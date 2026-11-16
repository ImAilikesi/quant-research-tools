"""Command-line interface: ``quanthygiene <command> ...``. Every command prints JSON."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from . import _io, costs, fx, report, risk, robustness, sizing, validation, volatility


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if k != "samples"}
    if isinstance(o, pd.DataFrame):
        return json.loads(o.to_json(orient="split"))
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    return o


def _out(obj) -> None:
    print(json.dumps(_clean(obj), indent=2, default=float))


def _returns(a) -> np.ndarray:
    return _io.returns_from_table(_io.read_table(a.csv), a.start_equity).to_numpy()


def _trades(a) -> pd.DataFrame:
    df = _io.read_table(a.csv)
    if _io.detect_kind(df) != "trades":
        sys.exit("error: this command needs a CSV with a 'pnl' column")
    return df


def cmd_bootstrap(a):
    _out(robustness.bootstrap_ci(_returns(a), a.stat, a.n_boot, a.block,
                                 periods_per_year=a.periods_per_year, seed=a.seed))


def cmd_dsr(a):
    _out(robustness.deflated_sharpe_ratio(_returns(a), a.trials))


def cmd_pbo(a):
    m = pd.read_csv(a.csv).select_dtypes("number").to_numpy()
    _out(robustness.pbo_cscv(m, a.splits, seed=a.seed))


def cmd_reshuffle(a):
    _out(robustness.reshuffle_mc(_trades(a)["pnl"], a.start_equity, a.n_sims, a.seed))


def cmd_costs(a):
    t = _trades(a)
    model = costs.CostModel(a.spread, a.commission, a.slippage, a.swap)
    _out({"sensitivity": costs.sensitivity(t, model), "breakeven_multiplier":
          costs.breakeven_multiplier(t, model), "concentration": costs.profit_concentration(t["pnl"])})


def cmd_breach(a):
    rules = risk.Rules(a.max_loss, a.trailing_dd, a.daily_loss, a.target, a.periods_per_day)
    _out(risk.simulate_rules(_returns(a), a.horizon, rules, a.paths, a.block, a.seed))


def cmd_ruin(a):
    r = _trades(a)["pnl"].to_numpy()
    _out(risk.risk_of_ruin(r / abs(r[r < 0].mean()), a.risk, a.ruin_dd, a.trades, seed=a.seed))


def cmd_size(a):
    _out({"fixed_fractional_units": sizing.fixed_fractional(a.equity, a.risk, a.stop, a.point_value),
          "kelly_binary": sizing.kelly_fraction(a.win_prob, a.payoff) if a.win_prob else None,
          "vol_target_exposure": sizing.vol_target_exposure(a.target_vol, a.asset_vol)
          if a.asset_vol else None})


def cmd_splits(a):
    gen = (validation.purged_kfold(a.n, a.folds, a.horizon, a.embargo) if a.kind == "purged"
           else validation.walk_forward_splits(a.n, a.folds, anchored=a.anchored))
    _out([{"train": [int(x[0]), int(x[-1]), len(x)], "test": [int(y[0]), int(y[-1]), len(y)]}
          for x, y in gen])


def cmd_vol(a):
    df = pd.read_csv(a.csv)
    df.columns = [c.lower() for c in df.columns]
    v = volatility.ESTIMATORS[a.estimator](df, a.window, a.periods_per_year).dropna()
    _out({"estimator": a.estimator, "last": float(v.iloc[-1]), "mean": float(v.mean())})


def cmd_fx(a):
    prices = pd.read_csv(a.csv, index_col=0)
    res = fx.diversification_check(fx.normalize_quote_direction(prices, a.numeraire))
    _out(res)


def cmd_tearsheet(a):
    html = report.build_tearsheet(_io.read_table(a.csv), a.title, a.periods_per_year, a.start_equity)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {a.out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("quanthygiene", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_, csv=True):
        s = sub.add_parser(name, help=help_)
        if csv:
            s.add_argument("csv", help="equity / returns / trades CSV")
            s.add_argument("--start-equity", type=float, default=100_000.0)
        s.add_argument("--seed", type=int, default=0)
        s.set_defaults(fn=fn)
        return s

    s = add("bootstrap", cmd_bootstrap, "bootstrap CI for Sharpe / mean / total return")
    s.add_argument("--stat", default="sharpe"); s.add_argument("--n-boot", type=int, default=5000)
    s.add_argument("--block", type=int, default=1); s.add_argument("--periods-per-year", type=float, default=252)
    s = add("dsr", cmd_dsr, "Deflated Sharpe Ratio"); s.add_argument("--trials", type=int, required=True)
    s = add("pbo", cmd_pbo, "Probability of Backtest Overfitting; CSV columns = strategy variants")
    s.add_argument("--splits", type=int, default=16)
    s = add("reshuffle", cmd_reshuffle, "trade-order reshuffle drawdown Monte Carlo")
    s.add_argument("--n-sims", type=int, default=5000)
    s = add("costs", cmd_costs, "cost sensitivity for a trades CSV (costs per unit)")
    for n, d in (("spread", 1.0), ("commission", 0.4), ("slippage", 0.25), ("swap", 0.0)):
        s.add_argument(f"--{n}", type=float, default=d)
    s = add("breach", cmd_breach, "loss-limit / trailing-drawdown / daily-loss breach Monte Carlo")
    s.add_argument("--horizon", type=int, default=60); s.add_argument("--paths", type=int, default=20000)
    s.add_argument("--block", type=int, default=1); s.add_argument("--periods-per-day", type=int, default=1)
    s.add_argument("--max-loss", type=float, default=0.10); s.add_argument("--trailing-dd", type=float)
    s.add_argument("--daily-loss", type=float, default=0.05); s.add_argument("--target", type=float, default=0.10)
    s = add("ruin", cmd_ruin, "risk of ruin from trade R-multiples")
    s.add_argument("--risk", type=float, default=0.01); s.add_argument("--ruin-dd", type=float, default=0.5)
    s.add_argument("--trades", type=int, default=1000)
    s = add("size", cmd_size, "position size helpers", csv=False)
    s.add_argument("--equity", type=float, default=10_000); s.add_argument("--risk", type=float, default=0.01)
    s.add_argument("--stop", type=float, default=10); s.add_argument("--point-value", type=float, default=1.0)
    s.add_argument("--win-prob", type=float); s.add_argument("--payoff", type=float, default=1.0)
    s.add_argument("--target-vol", type=float, default=0.10); s.add_argument("--asset-vol", type=float)
    s = add("splits", cmd_splits, "walk-forward or purged K-fold index ranges", csv=False)
    s.add_argument("--kind", choices=["walk-forward", "purged"], default="purged")
    s.add_argument("--n", type=int, required=True); s.add_argument("--folds", type=int, default=5)
    s.add_argument("--horizon", type=int, default=0); s.add_argument("--embargo", type=int, default=0)
    s.add_argument("--anchored", action="store_true")
    s = add("vol", cmd_vol, "OHLC volatility estimate (CSV: open,high,low,close)")
    s.add_argument("--estimator", choices=list(volatility.ESTIMATORS), default="yang-zhang")
    s.add_argument("--window", type=int, default=20); s.add_argument("--periods-per-year", type=float, default=252)
    s = add("fx-diversify", cmd_fx, "FX diversification check; CSV = date index + pair price columns")
    s.add_argument("--numeraire", default="USD")
    s = add("tearsheet", cmd_tearsheet, "write an HTML tearsheet")
    s.add_argument("--out", default="tearsheet.html"); s.add_argument("--title", default="Tearsheet")
    s.add_argument("--periods-per-year", type=float, default=252)
    return p


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
