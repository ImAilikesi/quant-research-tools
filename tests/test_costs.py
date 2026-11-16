import pandas as pd
import pytest

from quanthygiene import costs


@pytest.fixture
def trades():
    return pd.DataFrame({"pnl": [100.0, -50.0, 80.0, -20.0], "units": [1, 2, 1, 1], "hold_days": [0, 1, 2, 0]})


def test_trade_costs_formula(trades):
    m = costs.CostModel(spread=1, commission=0.5, slippage=0.25, swap_per_day=0.1)
    c = costs.trade_costs(trades, m)
    assert c.iloc[0] == pytest.approx(1 * (1 + 0.5 + 0.5))
    assert c.iloc[1] == pytest.approx(2 * (1 + 0.5 + 0.5) + 0.1 * 2 * 1)
    assert costs.trade_costs(trades, m, 2).sum() == pytest.approx(2 * c.sum())


def test_sensitivity_monotone_and_gross_first(trades):
    s = costs.sensitivity(trades, costs.CostModel(spread=1))
    assert s["total"].is_monotonic_decreasing
    assert s.loc[0, "total"] == trades["pnl"].sum()


def test_breakeven_multiplier_zeroes_total(trades):
    m = costs.CostModel(spread=2)
    k = costs.breakeven_multiplier(trades, m)
    assert (trades["pnl"] - costs.trade_costs(trades, m, k)).sum() == pytest.approx(0, abs=1e-9)
    assert costs.breakeven_multiplier(trades, costs.CostModel()) == float("inf")


def test_summarize_and_scenarios(trades):
    s = costs.summarize(trades["pnl"])
    assert s["profit_factor"] == pytest.approx(180 / 70)
    t = costs.scenario_table(trades)
    assert list(t["scenario"]) == list(costs.SCENARIOS)
    assert t.set_index("scenario").loc["stress", "total"] < t.set_index("scenario").loc["none", "total"]


def test_profit_concentration():
    c = costs.profit_concentration([100, 10, 10, -5], top=1)
    assert c["top_share_of_gross_profit"] == pytest.approx(100 / 120)
    assert c["total_without_top"] == 15
