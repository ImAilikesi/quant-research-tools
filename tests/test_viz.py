from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quanthygiene import cli, montecarlo, risk, viz

EX = Path(__file__).resolve().parent.parent / "examples"


def test_static_plots_return_figures():
    r = np.random.default_rng(0).normal(0.001, 0.01, 500)
    s = pd.Series(r, index=pd.bdate_range("2022-01-01", periods=500))
    paths = montecarlo.equity_paths(r * 1000, 30, 50)
    assert viz.plot_fan(paths).axes
    assert viz.plot_underwater(r).axes
    assert viz.plot_monthly_heatmap(s).axes
    assert viz.plot_breach_surface(r, risk.Rules(), (1, 2), (5, 10), 100).axes


def test_animations_write_valid_gifs(tmp_path):
    rng = np.random.default_rng(0)
    a = viz.animate_fan(rng.normal(5, 50, 100), str(tmp_path / "a.gif"), 20, 40, 4)
    b = viz.animate_overfitting(rng.normal(0, 0.01, (200, 10)), str(tmp_path / "b.gif"), 4)
    for p in (a, b):
        assert Path(p).read_bytes()[:6] in (b"GIF89a", b"GIF87a")


def test_cli_animate(tmp_path, capsys):
    out = tmp_path / "x.gif"
    cli.main(["animate", str(EX / "synthetic_variants.csv"), "overfit", "--out", str(out)])
    assert out.stat().st_size > 1000
    with pytest.raises(SystemExit):
        cli.main(["animate", str(EX / "synthetic_returns.csv"), "fan"])
