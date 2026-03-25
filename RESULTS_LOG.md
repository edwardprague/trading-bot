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
| v15 | 2026-03-24 10:05 | Trend Following | EURUSD=X | 1h | Tightened time filter — removed worst hours 00 and 19, keeping 01-04 and 16-18 | 54 | 42.6% | 1.49 | +$201.12 | -5.85% | 0.95 |
| v16 | 2026-03-24 10:10 | Trend Following | EURUSD=X | 1h | Increased RRR from 2.0 to 2.5 — testing if wider target improves results given 42.6% win rate | 53 | 37.7% | 1.53 | -$284.73 | -5.85% | 0.95 |
| v17 | 2026-03-24 10:35 | Trend Following | EURUSD=X | 1h | Reverted RRR to 2.0, added RRR and parameter sensitivity diagnostics | 54 | 42.6% | 1.49 | +$201.12 | -5.85% | 0.95 |
| v18 | 2026-03-24 10:57 | Trend Following | EURUSD=X | 1h | Added dollar drawdown amounts and daily drawdown tracking — aligning with FTMO $100k challenge parameters | 54 | 42.6% | 1.49 | +$201.12 | -5.85% | 0.95 |
| v18 | 2026-03-24 11:09 | Trend Following | EURUSD=X | 1h | Scaled to $100k capital — aligning with FTMO challenge parameters, added position size tracking | 54 | 42.6% | 1.49 | +$2011.15 | -5.85% | 0.95 |
| v20 | 2026-03-24 13:44 | Trend Following | GBPUSD=X | 1h | Testing same strategy on GBPUSD — checking if edge generalises across instruments | 48 | 45.8% | 1.65 | +$5450.17 | -5.85% | 1.19 |
| v21 | 2026-03-24 14:40 | Trend Following | GBPUSD=X | 1h | Added Rolling Profit Factor diagnostic — visualising regime performance over time | 48 | 45.8% | 1.65 | +$5450.17 | -5.85% | 1.19 |
| v22 | 2026-03-24 14:56 | Trend Following | EURUSD=X | 1h | Back to EURUSD — viewing Rolling PF chart | 54 | 42.6% | 1.49 | +$2011.15 | -5.85% | 0.95 |
| v23 | 2026-03-24 15:55 | Trend Following | EURUSD=X | 5m | Switched to Massive API — 5-minute EURUSD data | 454 | 33.0% | 0.97 | +$2786.30 | -31.16% | -0.03 |
| v23 | 2026-03-24 16:02 | Trend Following | EURUSD=X | 5m | Switched to Massive API — 5-minute EURUSD data | 454 | 33.0% | 0.97 | +$2786.30 | -31.16% | -0.03 |
| v23 | 2026-03-24 16:08 | Trend Following | EURUSD=X | 5m | Switched to Massive API — 5-minute EURUSD data | 454 | 33.0% | 0.97 | +$2786.30 | -31.16% | -0.03 |
| v23 | 2026-03-24 18:29 | Trend Following | EURUSD=X | 5m | Switched to Massive API — 5-minute EURUSD data | 454 | 33.0% | 0.97 | +$2786.30 | -31.16% | -0.03 |
| v23 | 2026-03-24 18:29 | Trend Following | EURUSD=X | 5m | Switched to Massive API — 5-minute EURUSD data | 454 | 33.0% | 0.97 | +$2786.30 | -31.16% | -0.03 |
| v23 | 2026-03-24 18:31 | Trend Following | EURUSD=X | 5m | Switched to Massive API — 5-minute EURUSD data | 454 | 33.0% | 0.97 | +$2786.30 | -31.16% | -0.03 |
| v23 | 2026-03-25 02:16 | Trend Following | EURUSD=X | 5m | Switched to Massive API — 5-minute EURUSD data | 455 | 33.0% | 0.97 | +$2786.30 | -31.16% | -0.03 |
| v23 | 2026-03-25 02:17 | Trend Following | EURUSD=X | 5m | Switched to Massive API — 5-minute EURUSD data | 455 | 33.0% | 0.97 | +$2786.30 | -31.16% | -0.03 |
| v23 | 2026-03-25 02:37 | Trend Following | EURUSD=X | 5m | Switched to Massive API — 5-minute EURUSD data | 455 | 33.0% | 0.97 | +$2786.30 | -31.16% | -0.03 |
| v23 | 2026-03-25 03:06 | Trend Following | EURUSD=X | 5m | Switched to Massive API — 5-minute EURUSD data | 455 | 33.0% | 0.97 | +$2786.30 | -31.16% | -0.03 |
| v23 | 2026-03-25 03:16 | Trend Following | EURUSD=X | 5m | Switched to Massive API — 5-minute EURUSD data | 455 | 33.0% | 0.97 | +$2786.30 | -31.16% | -0.03 |
| v35 | 2026-03-25 03:29 | Trend Following | EURUSD=X | 5m | Fixed P&L reconciliation — consistent compounding calculation | 455 | 33.0% | 0.97 | +$2786.30 | -31.16% | -0.03 |
| v36 | 2026-03-25 03:40 | Trend Following | EURUSD=X | 5m | Removed misleading direction P&L — replaced with avg P&L per trade and trade share | 455 | 33.0% | 0.97 | +$2786.30 | -31.16% | -0.03 |
| v36 | 2026-03-25 03:53 | Trend Following | EURUSD=X | 5m | Removed misleading direction P&L — replaced with avg P&L per trade and trade share | 455 | 33.0% | 0.97 | +$2786.30 | -31.16% | -0.03 |
| v36 | 2026-03-25 03:53 | Trend Following | EURUSD=X | 5m | Removed misleading direction P&L — replaced with avg P&L per trade and trade share | 455 | 33.0% | 0.97 | +$2786.30 | -31.16% | -0.03 |
