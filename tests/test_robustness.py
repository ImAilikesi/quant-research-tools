import numpy as np
import pytest

from quanthygiene import robustness as rb


def test_sharpe_known_value():
    r = np.array([0.01, -0.01, 0.01, -0.01, 0.02])
    assert rb.sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1))
    assert rb.sharpe(np.zeros(5)) == 0.0
    assert rb.sharpe(r, 252) == pytest.approx(rb.sharpe(r) * np.sqrt(252))


def test_bootstrap_ci_covers_point_and_is_reproducible():
    r = np.random.default_rng(0).normal(0.001, 0.01, 500)
    a = rb.bootstrap_ci(r, "sharpe", n_boot=2000, seed=1)
    assert a["ci_low"] < a["point"] < a["ci_high"]
    assert a == rb.bootstrap_ci(r, "sharpe", n_boot=2000, seed=1)


@pytest.mark.parametrize("stat", ["mean", "total"])
def test_bootstrap_other_stats_and_block(stat):
    r = np.random.default_rng(1).normal(0.001, 0.01, 200)
    out = rb.bootstrap_ci(r, stat, n_boot=500, block=5)
    assert out["ci_low"] < out["ci_high"]


def test_bootstrap_rejects_bad_input():
    with pytest.raises(ValueError):
        rb.bootstrap_ci([0.1], "sharpe")
    with pytest.raises(ValueError):
        rb.bootstrap_ci([0.1, 0.2, 0.3], "nope")


def test_expected_max_sharpe_grows_with_trials():
    assert rb.expected_max_sharpe(1, 0.01) == 0.0
    assert rb.expected_max_sharpe(100, 0.01) > rb.expected_max_sharpe(10, 0.01) > 0


def test_dsr_penalises_more_trials():
    r = np.random.default_rng(2).normal(0.001, 0.01, 750)
    few = rb.deflated_sharpe_ratio(r, 2)["dsr"]
    many = rb.deflated_sharpe_ratio(r, 500)["dsr"]
    assert 0 <= many < few <= 1


def test_dsr_high_for_strong_edge_low_for_noise():
    rng = np.random.default_rng(3)
    assert rb.deflated_sharpe_ratio(rng.normal(0.01, 0.01, 750), 10)["dsr"] > 0.99
    assert rb.deflated_sharpe_ratio(rng.normal(0.0, 0.01, 750), 100)["dsr"] < 0.5


def test_pbo_noise_is_high_and_real_edge_is_low():
    rng = np.random.default_rng(4)
    noise = rng.normal(0, 0.01, (800, 30))
    assert rb.pbo_cscv(noise, 8)["pbo"] > 0.3
    edge = noise.copy()
    edge[:, 0] += 0.01  # one variant with a persistent, large edge
    assert rb.pbo_cscv(edge, 8)["pbo"] < 0.1


def test_pbo_validates_arguments():
    with pytest.raises(ValueError):
        rb.pbo_cscv(np.zeros((100, 1)), 8)
    with pytest.raises(ValueError):
        rb.pbo_cscv(np.zeros((100, 3)), 7)


def test_drawdown_known_curve():
    eq = [100, 120, 90, 130, 117]
    assert rb.max_drawdown(eq) == pytest.approx(0.25)
    assert rb.drawdown_series(eq)[0] == 0


def test_reshuffle_total_invariant_and_percentile():
    pnl = np.random.default_rng(5).normal(5, 50, 100)
    out = rb.reshuffle_mc(pnl, 10_000, 300, seed=0)
    assert 0 <= out["observed_percentile"] <= 1
    assert out["dd_q05"] <= out["dd_median"] <= out["dd_q95"] <= out["dd_q99"]


def test_drawdown_distribution_ordering_and_horizon():
    r = np.random.default_rng(6).normal(0.0005, 0.01, 300)
    short = rb.drawdown_distribution(r, 50, 500)
    longer = rb.drawdown_distribution(r, 600, 500)
    assert longer["dd_median"] > short["dd_median"]
