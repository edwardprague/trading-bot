# Results Log

| Version | Date | Strategy | Instrument | Timeframe | Notes | Trades | Win Rate | Profit Factor | Net P&L | Max Drawdown | Sharpe Ratio |
|---------|------|----------|------------|-----------|-------|--------|----------|---------------|---------|--------------|-------------|
| v1 | 2026-03-25 10:20 | Trend Following | EURUSD=X | 5m | Baseline — 5-minute EURUSD short only time filter hours 01 02 16 17 18 | 382 | 34.3% | 1.03 | +$7412.88 | -25.36% | 0.08 |
| v3 | 2026-03-25 16:35 | Trend Following | EURUSD=X | 5m | Regime filter — 60 day test to visually verify range detection before full run | 31 | 41.9% | 1.43 | +$7953.25 | -6.79% | 0.67 |
| v3 | 2026-03-25 17:04 | Trend Following | EURUSD=X | 5m | Regime filter — added net displacement condition to fix inverted detection | 33 | 45.5% | 1.64 | +$12314.56 | -6.79% | 0.96 |
