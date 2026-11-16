import numpy as np
import pandas as pd
import pytest

from quanthygiene import fx, volatility


def make_ohlc(n=400, sigma=0.01, seed=0):
    rng = np.random.default_rng(seed)
    rows, c = [], 100.0
    for _ in range(n):
        o = c
        path = o * np.exp(np.cumsum(rng.normal(0, sigma / np.sqrt(20), 20)))
        c = path[-1]
        rows.append((o, max(path.max(), o), min(path.min(), o), c))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"])


@pytest.mark.parametrize("name", list(volatility.ESTIMATORS))
def test_estimators_recover_true_vol(name):
    df = make_ohlc(sigma=0.01)
    v = volatility.ESTIMATORS[name](df, 60, 1).dropna()
    assert v.mean() == pytest.approx(0.01, rel=0.25)


def test_estimators_scale_with_vol():
    lo = volatility.parkinson(make_ohlc(sigma=0.005), 30).mean()
    hi = volatility.parkinson(make_ohlc(sigma=0.02), 30).mean()
    assert hi > 2 * lo


def test_missing_columns():
    with pytest.raises(ValueError):
        volatility.parkinson(pd.DataFrame({"close": [1, 2]}))


def test_normalise_inverts_usd_base_pairs():
    idx = pd.RangeIndex(3)
    prices = pd.DataFrame({"EURUSD": [1.0, 1.1, 1.21], "USDJPY": [100.0, 100.0, 50.0]}, index=idx)
    out = fx.normalize_quote_direction(prices)
    assert list(out.columns) == ["EUR", "JPY"]
    assert out["EUR"].iloc[0] == pytest.approx(np.log(1.1))
    assert out["JPY"].iloc[1] == pytest.approx(np.log(2.0))  # USDJPY halves -> JPY doubles in USD
    with pytest.raises(ValueError):
        fx.normalize_quote_direction(pd.DataFrame({"EURGBP": [1.0, 1.1]}))


def test_split_pair_formats():
    assert fx.split_pair("eur/usd") == ("EUR", "USD")
    with pytest.raises(ValueError):
        fx.split_pair("EURO")


def test_diversification_identical_vs_independent():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, 1000)
    same = pd.DataFrame({"A": base, "B": base * 2, "C": base * 0.5})
    indep = pd.DataFrame(rng.normal(0, 0.01, (1000, 3)), columns=list("ABC"))
    s, i = fx.diversification_check(same), fx.diversification_check(indep)
    assert s["effective_bets"] < 1.2 and s["pc1_variance_share"] > 0.99
    assert i["effective_bets"] > 2.5 and i["diversification_ratio"] > 1.4
    assert s["mean_pairwise_corr"] > 0.99


def test_currency_exposure_netting():
    exp = fx.currency_exposure({"EURUSD": 100_000, "GBPUSD": -50_000, "USDJPY": 10_000}, {"EURUSD": 1.1, "GBPUSD": 1.25, "USDJPY": 150})
    assert exp["EUR"] == 100_000
    assert exp["USD"] == pytest.approx(-110_000 + 62_500 + 10_000)
    assert exp["JPY"] == pytest.approx(-1_500_000)
