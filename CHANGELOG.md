# Changelog

## 0.2.0
- Added `montecarlo` (equity bands from trades), `regimes` (volatility terciles), `rolling` metrics,
  `tradelog` analysis (R-multiples, streaks, holding time, MAE/MFE), `clustering`, and
  `min_backtest_length`; new CLI commands `mc-bands`, `regimes`, `rolling`, `tradelog`, `minbtl`, `cluster`.

## 0.1.0
- Initial release: robustness (bootstrap CI, Deflated Sharpe, PBO/CSCV, reshuffle and drawdown Monte Carlo),
  costs, prop-style risk rules and risk of ruin, position sizing, walk-forward and purged K-fold splits,
  OHLC volatility estimators, FX diagnostics, HTML tearsheet, and the `quanthygiene` CLI.
