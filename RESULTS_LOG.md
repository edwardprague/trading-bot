# Results Log

| Version | Date | Strategy | Instrument | Timeframe | Notes | Trades | Win Rate | Profit Factor | Net P&L | Max Drawdown | Sharpe Ratio |
|---------|------|----------|------------|-----------|-------|--------|----------|---------------|---------|--------------|-------------|
| v1 | 2026-03-23 13:23 | Trend Following | EURUSD=X | 1h | Baseline — EMA 20/50/200 swing stop 1:2 RRR | 191 | 35.1% | 1.07 | +$838.66 | -13.00% | 0.36 |
| v1 | 2026-03-23 13:44 | Trend Following | EURUSD=X | 1h | Baseline — EMA 20/50/200 swing stop 1:2 RRR | 191 | 35.1% | 1.07 | +$838.66 | -13.00% | 0.36 |
| v3 | 2026-03-23 13:48 | Trend Following | EURUSD=X | 1h | Increased EMA fast from 50 to 100 — wider separation between trend filter and entry signal | 191 | 31.9% | 0.92 | -$938.80 | -22.50% | -0.29 |
| v3 | 2026-03-23 18:03 | Trend Following | EURUSD=X | 1h | Increased EMA fast from 50 to 100 — wider separation between trend filter and entry signal | 191 | 31.4% | 0.90 | -$1205.30 | -22.50% | -0.40 |
| v5 | 2026-03-23 18:27 | Trend Following | EURUSD=X | 1h | Short only — testing direction asymmetry identified in v4 diagnostics | 93 | 38.7% | 1.25 | -$207.10 | -5.85% | 0.75 |
| v6 | 2026-03-24 03:52 | Trend Following | EURUSD=X | 1h | Short only plus time filter — blocking worst hours 06 09 14 20 23 keeping best hours 16-19 and 00-04 | 59 | 40.7% | 1.37 | -$101.67 | -5.85% | 0.80 |
| v6 | 2026-03-24 08:44 | Trend Following | EURUSD=X | 1h | Short only plus time filter — blocking worst hours 06 09 14 20 23 keeping best hours 16-19 and 00-04 | 59 | 40.7% | 1.37 | -$101.67 | -5.85% | 0.80 |
| v6 | 2026-03-24 08:56 | Trend Following | EURUSD=X | 1h | Short only plus time filter — blocking worst hours 06 09 14 20 23 keeping best hours 16-19 and 00-04 | 59 | 40.7% | 1.37 | -$101.67 | -5.85% | 0.80 |
| v9 | 2026-03-24 09:39 | Trend Following | EURUSD=X | 1h | Fixed time filter — short only strict hours 16-19 and 00-04 only | 59 | 40.7% | 1.37 | -$101.67 | -5.85% | 0.80 |
| v9 | 2026-03-24 09:39 | Trend Following | EURUSD=X | 1h | Fixed time filter — short only strict hours 16-19 and 00-04 only | 59 | 40.7% | 1.37 | -$101.67 | -5.85% | 0.80 |
| v11 | 2026-03-24 09:45 | Trend Following | EURUSD=X | 1h | Time filter timezone fix | 59 | 40.7% | 1.37 | -$101.67 | -5.85% | 0.80 |
| v12 | 2026-03-24 09:54 | Trend Following | EURUSD=X | 1h | Time filter bug fix — added continue to actually block trades outside allowed hours | 59 | 40.7% | 1.37 | -$101.67 | -5.85% | 0.80 |
| v12 | 2026-03-24 09:59 | Trend Following | EURUSD=X | 1h | Time filter bug fix — added continue to actually block trades outside allowed hours | 59 | 40.7% | 1.37 | -$101.67 | -5.85% | 0.80 |
| v13 | 2026-03-24 10:02 | Trend Following | EURUSD=X | 1h | Fixed time of day display — was showing incorrect hours in diagnostic table | 59 | 40.7% | 1.37 | -$100.98 | -5.85% | 0.80 |
