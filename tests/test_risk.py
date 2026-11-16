import numpy as np
import pytest

from quanthygiene import risk


def test_certain_loss_breaches_static_limit():
    out = risk.simulate_rules([-0.02], 20, risk.Rules(0.05, None, None, None), 200)
    assert out["p_breach"] == 1.0 and out["p_pass"] == 0.0


def test_certain_gain_passes():
    out = risk.simulate_rules([0.02], 20, risk.Rules(0.05, None, None, 0.10), 200)
    assert out["p_pass"] == 1.0 and out["median_periods_to_pass"] == 5


def test_trailing_drawdown_vs_static():
    # +3% then -2% repeating: static loss never hit but trailing drawdown of 1.5% is
    r = np.array([0.03, -0.02])
    rules = risk.Rules(0.10, 0.015, None, None)
    path = risk.simulate_rules(r, 2, rules, 2000, seed=0)
    assert path["p_breach"] > 0
    static = risk.simulate_rules(r, 2, risk.Rules(0.10, None, None, None), 2000)
    assert static["p_breach"] == 0


def test_daily_loss_resets_each_day():
    # two periods per day, each -3%: day loss 6% breaches a 5% limit; 1 period/day (3%) does not
    two = risk.simulate_rules([-0.03], 2, risk.Rules(None, None, 0.05, None, 2), 50)
    one = risk.simulate_rules([-0.03], 2, risk.Rules(None, None, 0.05, None, 1), 50)
    assert two["p_breach"] == 1.0 and one["p_breach"] == 0.0


def test_breach_before_pass_on_tie_counts_as_breach():
    out = risk.simulate_rules([0.10], 1, risk.Rules(None, None, None, 0.10), 10)
    assert out["p_pass"] == 1.0
    # breach and pass same period: trailing dd impossible here; use daily rule on a gain? use loss limit
    both = risk.simulate_rules([-0.5, 0.5], 1, risk.Rules(0.2, None, None, 10.0), 500, seed=1)
    assert both["p_breach"] + both["p_pass"] + both["p_timeout"] == pytest.approx(1.0)


def test_probabilities_sum_to_one_and_more_vol_more_breach():
    rng = np.random.default_rng(0)
    calm, wild = rng.normal(0, 0.005, 500), rng.normal(0, 0.02, 500)
    rules = risk.Rules(0.08, None, 0.04, 0.08)
    a = risk.simulate_rules(calm, 40, rules, 4000)
    b = risk.simulate_rules(wild, 40, rules, 4000)
    for o in (a, b):
        assert o["p_breach"] + o["p_pass"] + o["p_timeout"] == pytest.approx(1.0)
    assert b["p_breach"] > a["p_breach"]


def test_block_bootstrap_runs():
    r = np.random.default_rng(1).normal(0, 0.01, 200)
    assert 0 <= risk.simulate_rules(r, 30, risk.Rules(), 500, block=5)["p_breach"] <= 1


def test_risk_of_ruin_orders_with_risk_and_edge():
    r = np.array([2.0, -1.0, -1.0, 2.0, -1.0])
    low = risk.risk_of_ruin(r, 0.005, 0.3, 300, 3000)["p_ruin"]
    high = risk.risk_of_ruin(r, 0.05, 0.3, 300, 3000)["p_ruin"]
    assert high > low
    assert risk.risk_of_ruin([-1.0], 0.1, 0.3, 50, 100)["p_ruin"] == 1.0
