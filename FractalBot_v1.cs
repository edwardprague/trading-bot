// FractalBot_v1.cs
// ════════════════════════════════════════════════════════════════
// cTrader Automate cBot — Strategy v1: Fractal Geometry Entries
// Generated from Trading Bot Dashboard
//
// Instrument : GBPUSD
// Direction  : Short Only
// Entry      : N2 fractal (higher-low = long, lower-high = short)
// Stop       : Fractal price +/- 7 pip offset
// Target     : Entry +/- (stop distance x 1/1)
// Risk       : 1% of equity per trade
// Time filter: UTC hours 1,2,3,7,9,13,15,18,19,20
// Daily limit: Max 3 losing trades per UTC day
// EMAs       : 8 / 20 / 50 (chart reference only)
// ════════════════════════════════════════════════════════════════

using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Indicators;

namespace FractalBot
{
    [Robot(
        TimeZone = TimeZones.UTC,
        AccessRights = AccessRights.None
    )]
    public class FractalBot_v1 : Robot
    {
        // ── Parameters ───────────────────────────────────────────────────

        [Parameter("EMA Short Period", Group = "EMAs", DefaultValue = 8, MinValue = 2)]
        public int EmaShortPeriod { get; set; }

        [Parameter("EMA Mid Period", Group = "EMAs", DefaultValue = 20, MinValue = 2)]
        public int EmaMidPeriod { get; set; }

        [Parameter("EMA Long Period", Group = "EMAs", DefaultValue = 50, MinValue = 2)]
        public int EmaLongPeriod { get; set; }

        [Parameter("Fractal Stop Offset (pips)", Group = "Stop", DefaultValue = 7, MinValue = 1)]
        public double FractalStopPips { get; set; }

        [Parameter("Min Stop (pips)", Group = "Stop", DefaultValue = 5, MinValue = 1)]
        public double MinStopPips { get; set; }

        [Parameter("Max Stop (pips)", Group = "Stop", DefaultValue = 200, MinValue = 1)]
        public double MaxStopPips { get; set; }

        [Parameter("Risk Percent", Group = "Risk", DefaultValue = 1.0, MinValue = 0.1, MaxValue = 10.0)]
        public double RiskPercent { get; set; }

        [Parameter("RRR Risk", Group = "Risk", DefaultValue = 1, MinValue = 1)]
        public int RrrRisk { get; set; }

        [Parameter("RRR Reward", Group = "Risk", DefaultValue = 1, MinValue = 1)]
        public int RrrReward { get; set; }

        [Parameter("Max Daily Losses", Group = "Risk", DefaultValue = 3, MinValue = 0)]
        public int MaxDailyLosses { get; set; }

        [Parameter("Allowed Hours UTC (comma-separated)", Group = "Filters",
                   DefaultValue = "1,2,3,7,9,13,15,18,19,20")]
        public string AllowedHoursUtc { get; set; }


        // ── Indicator instances ──────────────────────────────────────────

        private ExponentialMovingAverage _emaShort;
        private ExponentialMovingAverage _emaMid;
        private ExponentialMovingAverage _emaLong;


        // ── Fractal tracking state ───────────────────────────────────────

        private double _lastFractalHighPrice  = double.NaN;
        private double _priorFractalHighPrice = double.NaN;
        private double _lastFractalLowPrice   = double.NaN;
        private double _priorFractalLowPrice  = double.NaN;


        // ── Runtime state ────────────────────────────────────────────────

        private int[]    _allowedHours;
        private int      _dailyLossCount;
        private DateTime _currentDay;
        private int      _lastCloseBar = 0;


        // ── Lifecycle: OnStart ───────────────────────────────────────────

        protected override void OnStart()
        {
            _emaShort = Indicators.ExponentialMovingAverage(Bars.ClosePrices, EmaShortPeriod);
            _emaMid   = Indicators.ExponentialMovingAverage(Bars.ClosePrices, EmaMidPeriod);
            _emaLong  = Indicators.ExponentialMovingAverage(Bars.ClosePrices, EmaLongPeriod);

            _allowedHours   = ParseHours(AllowedHoursUtc);
            _currentDay     = Server.Time.Date;
            _dailyLossCount = 0;

            Positions.Closed += OnPositionClosed;

            Print("FractalBot v1 started");
            Print($"  EMAs {EmaShortPeriod}/{EmaMidPeriod}/{EmaLongPeriod}  |  " +
                  $"Stop offset {FractalStopPips} pips  |  " +
                  $"RRR {RrrRisk}:{RrrReward}  |  Risk {RiskPercent}%  |  " +
                  $"Max daily losses {MaxDailyLosses}");
            Print($"  Allowed UTC hours: [{string.Join(", ", _allowedHours)}]");
        }


        // ── Main logic: OnBar ────────────────────────────────────────────
        //
        // Execution order mirrors Python backtest:
        //   1. Daily loss limit reset
        //   2. Fractal detection (confirmed at bar i-2)
        //   3. Entry signal generation (higher-low / lower-high)
        //   4. Direction filter
        //   5. Daily loss limit check
        //   6. Time filter (allowed hours)
        //   7. Stop validation & order execution

        protected override void OnBar()
        {
            // Need at least 5 completed bars for fractal detection
            if (Bars.Count < 6)
                return;

            // ── 1. Daily loss limit reset ────────────────────────────────
            DateTime today = Server.Time.Date;
            if (today != _currentDay)
            {
                _currentDay     = today;
                _dailyLossCount = 0;
            }

            // ── 2. Fractal detection ─────────────────────────────────────
            // Python: at bar i, confirm fractal at bar i-2 (needs i-4..i).
            // cTrader: Last(1) = just-closed bar = Python's bar i.
            //          Fractal candidate = Last(3) = Python's bar i-2.
            //          Surrounding bars  = Last(5), Last(4), Last(2), Last(1).

            double fHigh = Bars.HighPrices.Last(3);
            double fLow  = Bars.LowPrices.Last(3);

            bool isPivotHigh = fHigh > Bars.HighPrices.Last(5)
                            && fHigh > Bars.HighPrices.Last(4)
                            && fHigh > Bars.HighPrices.Last(2)
                            && fHigh > Bars.HighPrices.Last(1);

            bool isPivotLow  = fLow < Bars.LowPrices.Last(5)
                            && fLow < Bars.LowPrices.Last(4)
                            && fLow < Bars.LowPrices.Last(2)
                            && fLow < Bars.LowPrices.Last(1);

            bool newHighConfirmed = false;
            bool newLowConfirmed  = false;

            if (isPivotHigh)
            {
                _priorFractalHighPrice = _lastFractalHighPrice;
                _lastFractalHighPrice  = fHigh;
                newHighConfirmed = true;
                Print($"[FRACTAL HIGH] {Bars.OpenTimes.Last(3):yyyy-MM-dd HH:mm} UTC | Price {fHigh:F5} | Prior {_priorFractalHighPrice:F5}");
            }

            if (isPivotLow)
            {
                _priorFractalLowPrice = _lastFractalLowPrice;
                _lastFractalLowPrice  = fLow;
                newLowConfirmed = true;
                Print($"[FRACTAL LOW]  {Bars.OpenTimes.Last(3):yyyy-MM-dd HH:mm} UTC | Price {fLow:F5} | Prior {_priorFractalLowPrice:F5}");
            }


            // ── Skip if already in a trade ───────────────────────────────
            if (Positions.Find(Symbol.Name) != null)
                return;

            // ── One-bar cooldown after close ─────────────────────────────
            if (Bars.Count <= _lastCloseBar + 1)
                return;

            double close = Bars.ClosePrices.Last(1);

            // ── 3. Entry signals ─────────────────────────────────────────
            // Long: new higher-low fractal confirmed
            //   (last fractal low > prior fractal low)
            bool longSig = newLowConfirmed
                        && !double.IsNaN(_lastFractalLowPrice)
                        && !double.IsNaN(_priorFractalLowPrice)
                        && _lastFractalLowPrice > _priorFractalLowPrice;

            // Short: new lower-high fractal confirmed
            //   (last fractal high < prior fractal high)
            bool shortSig = newHighConfirmed
                         && !double.IsNaN(_lastFractalHighPrice)
                         && !double.IsNaN(_priorFractalHighPrice)
                         && _lastFractalHighPrice < _priorFractalHighPrice;


            // ── 4. Direction filter ──────────────────────────────────────
            longSig = false;  // short_only

            if (!longSig && !shortSig)
                return;


            // ── 5. Daily loss limit ──────────────────────────────────────
            if (_dailyLossCount >= MaxDailyLosses)
                return;


            // ── 6. Time filter ───────────────────────────────────────────
            if (!IsAllowedHour(Server.Time.Hour))
                return;


            // ── 7. Stop validation & execution ───────────────────────────
            double fractalStopOffset = FractalStopPips * Symbol.PipSize;
            double minStopPrice      = MinStopPips * Symbol.PipSize;
            double maxStopPrice      = MaxStopPips * Symbol.PipSize;
            double rrr               = (double)RrrReward / RrrRisk;

            if (longSig)
            {
                double slPrice = _lastFractalLowPrice - fractalStopOffset;
                double dist    = close - slPrice;

                if (dist >= minStopPrice && dist <= maxStopPrice)
                {
                    double stopPips = dist / Symbol.PipSize;
                    double tpPips   = stopPips * rrr;

                    double riskAmount = Account.Equity * (RiskPercent / 100.0);
                    double volume     = Symbol.NormalizeVolumeInUnits(riskAmount / dist);

                    var result = ExecuteMarketOrder(
                        TradeType.Buy, Symbol.Name, volume,
                        Symbol.Name, stopPips, tpPips
                    );

                    if (result.IsSuccessful)
                        Print($"[LONG]  {Server.Time:yyyy-MM-dd HH:mm} UTC | " +
                              $"Entry ~{close:F5} | SL {stopPips:F1} pips | " +
                              $"TP {tpPips:F1} pips | {volume:F0} units");
                    else
                        Print($"[ORDER FAILED] {Server.Time:yyyy-MM-dd HH:mm} — {result.Error}");
                }
            }
            else if (shortSig)
            {
                double slPrice = _lastFractalHighPrice + fractalStopOffset;
                double dist    = slPrice - close;

                if (dist >= minStopPrice && dist <= maxStopPrice)
                {
                    double stopPips = dist / Symbol.PipSize;
                    double tpPips   = stopPips * rrr;

                    double riskAmount = Account.Equity * (RiskPercent / 100.0);
                    double volume     = Symbol.NormalizeVolumeInUnits(riskAmount / dist);

                    var result = ExecuteMarketOrder(
                        TradeType.Sell, Symbol.Name, volume,
                        Symbol.Name, stopPips, tpPips
                    );

                    if (result.IsSuccessful)
                        Print($"[SHORT] {Server.Time:yyyy-MM-dd HH:mm} UTC | " +
                              $"Entry ~{close:F5} | SL {stopPips:F1} pips | " +
                              $"TP {tpPips:F1} pips | {volume:F0} units");
                    else
                        Print($"[ORDER FAILED] {Server.Time:yyyy-MM-dd HH:mm} — {result.Error}");
                }
            }
        }


        // ── Daily loss tracking ──────────────────────────────────────────

        private void OnPositionClosed(PositionClosedEventArgs args)
        {
            if (args.Position.Label != Symbol.Name)
                return;

            _lastCloseBar = Bars.Count;

            if (args.Position.NetProfit < 0)
                _dailyLossCount++;
        }


        // ── Helpers ──────────────────────────────────────────────────────

        private int[] ParseHours(string hoursString)
        {
            var result = new List<int>();
            if (string.IsNullOrWhiteSpace(hoursString))
                return result.ToArray();
            foreach (var part in hoursString.Split(','))
            {
                if (int.TryParse(part.Trim(), out int hour))
                    result.Add(hour);
            }
            return result.ToArray();
        }

        private bool IsAllowedHour(int hour)
        {
            foreach (int h in _allowedHours)
            {
                if (h == hour) return true;
            }
            return false;
        }
    }
}
