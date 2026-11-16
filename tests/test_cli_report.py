import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quanthygiene import _io, cli, report

EX = Path(__file__).resolve().parent.parent / "examples"


def run(capsys, *argv):
    cli.main([str(a) for a in argv])
    return capsys.readouterr().out


@pytest.mark.parametrize("argv", [
    ["bootstrap", EX / "synthetic_returns.csv", "--n-boot", "300"],
    ["dsr", EX / "synthetic_returns.csv", "--trials", "20"],
    ["pbo", EX / "synthetic_variants.csv", "--splits", "8"],
    ["reshuffle", EX / "synthetic_trades.csv", "--n-sims", "100"],
    ["costs", EX / "synthetic_trades.csv"],
    ["breach", EX / "synthetic_returns.csv", "--paths", "500", "--horizon", "20"],
    ["ruin", EX / "synthetic_trades.csv", "--trades", "50"],
    ["size", "--win-prob", "0.5", "--asset-vol", "0.2"],
    ["splits", "--n", "60", "--folds", "3", "--horizon", "2"],
    ["splits", "--kind", "walk-forward", "--n", "60", "--folds", "3"],
    ["vol", EX / "synthetic_ohlc.csv", "--estimator", "parkinson"],
    ["fx-diversify", EX / "synthetic_fx.csv"],
    ["mc-bands", EX / "synthetic_trades.csv", "--n-sims", "200", "--n-trades", "30"],
    ["mc-bands", EX / "synthetic_trades.csv", "--n-sims", "50", "--mode", "shuffle"],
    ["regimes", EX / "synthetic_returns.csv"],
    ["rolling", EX / "synthetic_returns.csv", "--window", "40"],
    ["tradelog", EX / "synthetic_trades.csv"],
    ["minbtl", "--trials", "50", "--sharpe", "1.5"],
    ["cluster", EX / "synthetic_strategies.csv", "--clusters", "2"],
])
def test_every_cli_command_prints_json(capsys, argv):
    assert json.loads(run(capsys, *argv)) is not None


def test_cli_rejects_non_trades_csv(capsys):
    with pytest.raises(SystemExit):
        cli.main(["reshuffle", str(EX / "synthetic_returns.csv")])


@pytest.mark.parametrize("name", ["synthetic_trades.csv", "synthetic_returns.csv"])
def test_tearsheet_writes_self_contained_html(tmp_path, capsys, name):
    out = tmp_path / "t.html"
    run(capsys, "tearsheet", EX / name, "--out", out)
    page = out.read_text()
    assert page.startswith("<!doctype html>") and "data:image/png;base64" in page and "Sharpe" in page


def test_tearsheet_from_equity():
    eq = pd.DataFrame({"equity": 100 * np.cumprod(1 + np.random.default_rng(0).normal(0.001, 0.01, 200))})
    assert "Max drawdown" in report.build_tearsheet(eq)


def test_io_kinds():
    assert _io.detect_kind(pd.DataFrame({"PnL": [1]})) == "trades"
    assert _io.detect_kind(pd.DataFrame({"equity": [1]})) == "equity"
    with pytest.raises(ValueError):
        _io.detect_kind(pd.DataFrame({"x": [1]}))
    r = _io.returns_from_table(pd.DataFrame({"pnl": [100.0, -50.0]}), 1000.0)
    assert r.iloc[0] == pytest.approx(0.1) and r.iloc[1] == pytest.approx(-50 / 1100)
