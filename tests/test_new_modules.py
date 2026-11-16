import numpy as np
import pandas as pd
import pytest

from quanthygiene import clustering, montecarlo, regimes, robustness, rolling, tradelog


def test_equity_paths_shapes_and_start():
    p = np.random.default_rng(0).normal(5, 50, 100)
    paths = montecarlo.equity_paths(p, 40, 200, start_equity=1000)
    assert paths.shape == (200, 41) and (paths[:, 0] == 1000).all()


def test_shuffle_mode_keeps_final_pnl():
    p = np.random.default_rng(0).normal(5, 50, 60)
    paths = montecarlo.equity_paths(p, mode="shuffle", n_sims=50, start_equity=0)
    assert np.allclose(paths[:, -1], p.sum())
    with pytest.raises(ValueError):
        montecarlo.equity_paths(p, 100, mode="shuffle")
    with pytest.raises(ValueError):
        montecarlo.equity_paths(p, mode="x")


def test_bands_ordered_and_summary_extremes():
    paths = montecarlo.equity_paths(np.random.default_rng(1).normal(0, 50, 80), 80, 500)
    b = montecarlo.confidence_bands(paths)
    assert (b[0.05] <= b[0.5]).all() and (b[0.5] <= b[0.95]).all()
    win = montecarlo.summarize_paths(montecarlo.equity_paths([10.0], 20, 50))
    assert win["p_final_below_start"] == 0 and win["p_max_dd_above_threshold"] == 0
    lose = montecarlo.summarize_paths(montecarlo.equity_paths([-10.0], 20, 50))
    assert lose["p_final_below_start"] == 1


def test_regimes_use_no_lookahead_and_split_evenly():
    r = np.random.default_rng(0).normal(0, 0.01, 600)
    lab = regimes.volatility_regimes(r, 20)
    assert lab.iloc[:20].isna().all() and lab.dropna().nunique() == 3
    # a shock at t must raise the label of t+1, not of t
    rng = np.random.default_rng(5)
    r = np.concatenate([rng.normal(0, 0.001, 300), [0.5], rng.normal(0, 0.01, 300)])
    lab = regimes.volatility_regimes(r, 20)
    assert lab.iloc[300] == 0 and lab.iloc[301] == 2


def test_regime_performance_detects_calm_edge():
    rng = np.random.default_rng(1)
    vol = np.where((np.arange(3000) // 100) % 2 == 0, 0.004, 0.02)
    r = rng.normal(0, 1, 3000) * vol
    r[vol == 0.004] += 0.002  # edge only in calm blocks
    perf = regimes.regime_performance(r, 20)
    assert perf.loc[0, "sharpe"] > perf.loc[2, "sharpe"]
    assert perf["n"].sum() > 2900


def test_rolling_metrics_against_manual():
    r = np.random.default_rng(2).normal(0.001, 0.01, 200)
    s = rolling.rolling_sharpe(r, 50, 1)
    assert s.iloc[:49].isna().all()
    assert s.iloc[60] == pytest.approx(r[11:61].mean() / r[11:61].std(ddof=1))
    h = rolling.rolling_hit_rate(r, 50)
    assert h.iloc[60] == pytest.approx((r[11:61] > 0).mean())
    d = rolling.rolling_drawdown(np.array([0.1, -0.5, 0.1, 0.1]), 3)
    assert d.iloc[2] == pytest.approx(0.5) and np.isnan(d.iloc[0])


def test_streaks():
    out = tradelog.streaks([1, 1, -1, -1, -1, 0, 1, -1])
    assert out["max_win_streak"] == 2 and out["max_loss_streak"] == 3
    assert out["mean_win_streak"] == pytest.approx(1.5)
    assert tradelog.streaks([])["max_win_streak"] == 0


def test_tradelog_analyze_full_columns():
    t = pd.DataFrame({"pnl": [100.0, -50.0, 60.0, -40.0], "risk": [50.0] * 4,
                      "entry_time": pd.to_datetime(["2024-01-01 00:00", "2024-01-02 00:00", "2024-01-03 00:00", "2024-01-04 00:00"]),
                      "exit_time": pd.to_datetime(["2024-01-01 10:00", "2024-01-02 02:00", "2024-01-03 04:00",
                                                   "2024-01-04 02:00"]),
                      "mae": [10.0, 50.0, 5.0, 40.0], "mfe": [200.0, 5.0, 120.0, 0.0]})
    a = tradelog.analyze(t)
    assert a["win_rate"] == 0.5 and a["payoff_ratio"] == pytest.approx(80 / 45)
    assert a["expectancy_R"] == pytest.approx(0.35)
    assert a["holding_hours_mean_winners"] == pytest.approx(7.0) and a["holding_hours_mean_losers"] == pytest.approx(2.0)
    assert a["winner_capture_ratio"] == pytest.approx(0.5)
    assert tradelog.r_multiples(t).iloc[0] == 2.0


def test_tradelog_minimal_columns():
    a = tradelog.analyze(pd.DataFrame({"pnl": [1.0, 2.0]}))
    assert a["payoff_ratio"] is None and "expectancy_R" not in a


def test_min_backtest_length():
    assert robustness.min_backtest_length(100, 1.0) == pytest.approx(2 * np.log(100))
    assert robustness.min_backtest_length(100, 2.0) == pytest.approx(robustness.min_backtest_length(100, 1.0) / 4)
    assert robustness.min_backtest_length(1000, 1.0) > robustness.min_backtest_length(10, 1.0)
    with pytest.raises(ValueError):
        robustness.min_backtest_length(1, 1.0)


def test_cluster_recovers_two_factor_groups():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 800), rng.normal(0, 1, 800)
    df = pd.DataFrame({"a1": a + 0.2 * rng.normal(size=800), "a2": a + 0.2 * rng.normal(size=800),
                       "b1": b + 0.2 * rng.normal(size=800), "b2": b + 0.2 * rng.normal(size=800)})
    res = clustering.cluster(df, 2)
    assert sorted(sorted(g) for g in res["clusters"]) == [["a1", "a2"], ["b1", "b2"]]
    assert len(res["merges"]) == 2
    assert res["labels"]["a1"] == res["labels"]["a2"] != res["labels"]["b1"]
    d = clustering.correlation_distance(df)
    assert d.loc["a1", "a1"] == 0 and d.loc["a1", "a2"] < d.loc["a1", "b1"]
    with pytest.raises(ValueError):
        clustering.cluster(df, 9)
