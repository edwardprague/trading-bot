// TrendFollowerBot.cs
// ===================
// cTrader Automate cBot — Trend Following (EMA 20/50/200 + Swing Stop)
//
// Mirrors Python strategy.py v6 exactly:
//   Instrument : EURUSD, 5-minute bars
//   Direction  : Short only
//   Trend      : EMA50 < EMA200
//   Entry      : Close crosses below EMA20
//   Stop       : Swing high over 20 bars (excluding current bar)
//   Target     : Entry - (stop distance × RRR)
//   Risk       : 1% of equity per trade
//   Time filter: UTC hours 1, 2, 16, 17, 18
//   Daily limit: No new entries once realised loss >= $2,500
//
// This version: structure and signal detection only.
// OnBar() detects and logs signals — no orders are placed yet.
// ─────────────────────────────────────────────────────────────────────────────

using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Indicators;

namespace TrendFollower
{
    [Robot(
        TimeZone = TimeZones.UTC,           // Server.Time will be UTC throughout
        AccessRights = AccessRights.None
    )]
    public class TrendFollowerBot : Robot
    {
        // ── Parameters ───────────────────────────────────────────────────────
        // Grouped to match strategy.py configuration block.

        [Parameter("EMA Slow Period", Group = "EMAs", DefaultValue = 200, MinValue = 2)]
        public int EmaSlowPeriod { get; set; }

        [Parameter("EMA Fast Period", Group = "EMAs", DefaultValue = 50, MinValue = 2)]
        public int EmaFastPeriod { get; set; }

        [Parameter("EMA Entry Period", Group = "EMAs", DefaultValue = 20, MinValue = 2)]
        public int EmaEntryPeriod { get; set; }

        // Python: SWING_LOOKBACK = 20
        [Parameter("Swing Lookback (bars)", Group = "Stop", DefaultValue = 20, MinValue = 1)]
        public int SwingLookback { get; set; }

        // Python: MIN_STOP = 0.0005  (5 pips)
        [Parameter("Min Stop (pips)", Group = "Stop", DefaultValue = 5, MinValue = 1)]
        public double MinStopPips { get; set; }

        // Python: MAX_STOP = 0.0200  (200 pips)
        [Parameter("Max Stop (pips)", Group = "Stop", DefaultValue = 200, MinValue = 1)]
        public double MaxStopPips { get; set; }

        // Python: RISK_PCT = 0.01
        [Parameter("Risk Percent", Group = "Risk", DefaultValue = 1.0, MinValue = 0.1, MaxValue = 10.0)]
        public double RiskPercent { get; set; }

        // Python: RRR = 2.0
        [Parameter("RRR", Group = "Risk", DefaultValue = 2.0, MinValue = 0.5)]
        public double Rrr { get; set; }

        // Python: MAX_DAILY_LOSS = 2500.0
        [Parameter("Max Daily Loss ($)", Group = "Risk", DefaultValue = 2500.0, MinValue = 0)]
        public double MaxDailyLoss { get; set; }

        // Python: TIME_FILTER_HOURS = [1, 2, 16, 17, 18]
        // cTrader doesn't support array parameters — using a comma-separated string.
        [Parameter("Time Filter Hours UTC (comma-separated)", Group = "Filters", DefaultValue = "1,2,16,17,18")]
        public string TimeFilterHours { get; set; }


        // ── Indicator instances ───────────────────────────────────────────────

        private ExponentialMovingAverage _emaSlow;    // EMA 200 — trend baseline
        private ExponentialMovingAverage _emaFast;    // EMA 50  — trend confirmation
        private ExponentialMovingAverage _emaEntry;   // EMA 20  — entry crossover signal


        // ── Runtime state ─────────────────────────────────────────────────────

        private int[]    _allowedHours;         // parsed from TimeFilterHours parameter
        private double   _dailyRealizedPnl;     // closed-trade P&L accumulated for _currentDay
        private DateTime _currentDay;           // UTC date currently being tracked


        // ── Lifecycle: OnStart ────────────────────────────────────────────────

        protected override void OnStart()
        {
            // Initialise EMA indicators on the Close price series.
            // Equivalent to Python: df.Close.ewm(span=N, adjust=False).mean()
            _emaSlow  = Indicators.ExponentialMovingAverage(Bars.ClosePrices, EmaSlowPeriod);
            _emaFast  = Indicators.ExponentialMovingAverage(Bars.ClosePrices, EmaFastPeriod);
            _emaEntry = Indicators.ExponentialMovingAverage(Bars.ClosePrices, EmaEntryPeriod);

            // Parse the time filter hours string into an int array.
            _allowedHours = ParseHours(TimeFilterHours);

            // Initialise daily loss tracking state.
            _currentDay       = Server.Time.Date;
            _dailyRealizedPnl = 0.0;

            Print("TrendFollowerBot started — signal detection only (no orders)");
            Print($"  EMA {EmaSlowPeriod}/{EmaFastPeriod}/{EmaEntryPeriod}  |  " +
                  $"Swing {SwingLookback} bars  |  Stop {MinStopPips}–{MaxStopPips} pips  |  " +
                  $"RRR 1:{Rrr}  |  Risk {RiskPercent}%  |  Max daily loss ${MaxDailyLoss}");
            Print($"  Time filter: UTC hours [{string.Join(", ", _allowedHours)}]");
        }


        // ── Main logic: OnBar ─────────────────────────────────────────────────
        // Fires once per completed bar. Uses Last(1) for the just-closed bar
        // and Last(2) for the bar before it — mirroring Python's current bar 'c'
        // and previous bar 'cp'.

        protected override void OnBar()
        {
            // ── 1. Time filter ────────────────────────────────────────────────
            // Python: if entry_hour not in TIME_FILTER_HOURS: continue
            // Server.Time is UTC because [Robot(TimeZone = TimeZones.UTC)].
            int barHour = Server.Time.Hour;
            if (!IsAllowedHour(barHour))
                return;


            // ── 2. Daily loss limit ───────────────────────────────────────────
            // Python resets _daily_loss_pnl to 0 when the UTC date changes.
            // _dailyRealizedPnl is updated in OnTrade() once orders are added.
            DateTime today = Server.Time.Date;
            if (today != _currentDay)
            {
                _currentDay       = today;
                _dailyRealizedPnl = 0.0;
            }

            // Python: if _daily_loss_pnl <= -MAX_DAILY_LOSS: continue
            if (_dailyRealizedPnl <= -MaxDailyLoss)
            {
                Print($"[{Server.Time:HH:mm} UTC] Daily loss limit reached " +
                      $"(${_dailyRealizedPnl:F2}) — no new entries today.");
                return;
            }


            // ── 3. Trend filter ───────────────────────────────────────────────
            // Python: trend_down = fast < slow  (EMA50 < EMA200)
            double emaFastNow = _emaFast.Result.Last(1);
            double emaSlowNow = _emaSlow.Result.Last(1);

            bool trendDown = emaFastNow < emaSlowNow;
            if (!trendDown)
                return;


            // ── 4. EMA20 crossover — short signal ─────────────────────────────
            // Python:
            //   short_sig_raw = trend_down and cp > enp and c < en
            //     c   = current bar close  = Last(1)
            //     cp  = previous bar close = Last(2)
            //     en  = EMA20 at current bar  = Last(1)
            //     enp = EMA20 at previous bar = Last(2)
            double closeCurrent   = Bars.ClosePrices.Last(1);
            double closePrevious  = Bars.ClosePrices.Last(2);
            double ema20Current   = _emaEntry.Result.Last(1);
            double ema20Previous  = _emaEntry.Result.Last(2);

            // Previous close was ABOVE EMA20, current close is BELOW EMA20.
            bool crossedBelowEma20 = closePrevious > ema20Previous
                                  && closeCurrent  < ema20Current;

            if (!crossedBelowEma20)
                return;


            // ── 5. Swing high calculation ─────────────────────────────────────
            // Python: df["s_high"] = df.High.shift(1).rolling(SWING_LOOKBACK).max()
            // shift(1) excludes the current bar, so we want the maximum High
            // over the SwingLookback bars that preceded the current completed bar.
            //
            // cTrader Last() indexing:
            //   Last(1) = current completed bar   (excluded — mirrors shift(1))
            //   Last(2) = 1 bar ago
            //   Last(SwingLookback + 1) = SwingLookback bars ago
            double swingHigh = double.MinValue;
            for (int j = 2; j <= SwingLookback + 1; j++)
            {
                double h = Bars.HighPrices.Last(j);
                if (h > swingHigh)
                    swingHigh = h;
            }


            // ── 6. Stop distance validation ───────────────────────────────────
            // Python: dist = s_hi - c  /  MIN_STOP <= dist <= MAX_STOP
            // Convert pip limits to price using Symbol.PipSize (e.g. 0.00001 for EURUSD).
            double entryPrice   = closeCurrent;
            double stopDistance = swingHigh - entryPrice;

            double minStopPrice = MinStopPips * Symbol.PipSize;
            double maxStopPrice = MaxStopPips * Symbol.PipSize;

            if (stopDistance < minStopPrice || stopDistance > maxStopPrice)
                return;


            // ── 7. Calculate trade levels ─────────────────────────────────────
            // Python:
            //   sl   = s_hi
            //   tp   = c - dist * RRR
            //   size = (cash * RISK_PCT) / dist
            double stopLevel       = swingHigh;
            double takeProfitLevel = entryPrice - stopDistance * Rrr;
            double stopPips        = stopDistance / Symbol.PipSize;

            // Position size in units (same formula as Python).
            // Convert to standard lots (1 lot = 100,000 units) for reference.
            double riskAmount    = Account.Equity * (RiskPercent / 100.0);
            double positionUnits = riskAmount / stopDistance;
            double lots          = positionUnits / 100_000.0;


            // ── 8. Log signal — no order placed yet ───────────────────────────
            Print($"───── SHORT SIGNAL ─────────────────────────────────────");
            Print($"  Time      : {Server.Time:yyyy-MM-dd HH:mm} UTC");
            Print($"  Entry     : {entryPrice:F5}");
            Print($"  Swing High: {swingHigh:F5}");
            Print($"  Stop      : {stopLevel:F5}  ({stopPips:F1} pips above entry)");
            Print($"  Target    : {takeProfitLevel:F5}  (RRR 1:{Rrr})");
            Print($"  Risk $    : {riskAmount:F2}  |  Size: {positionUnits:F0} units  ({lots:F2} lots)");
            Print($"  EMA50={emaFastNow:F5}  EMA200={emaSlowNow:F5}  EMA20={ema20Current:F5}");
            Print($"────────────────────────────────────────────────────────");
        }


        // ── Daily P&L tracking ────────────────────────────────────────────────
        // Placeholder — wired up here so the daily loss limit has a data source
        // once order execution is added in the next session.
        // Python updates _daily_loss_pnl inside the trade exit block.
        // In cTrader this will be driven by OnTrade() or Position.NetProfit.
        protected override void OnTrade(TradeResult tradeResult)
        {
            // When orders are added: accumulate closed P&L into _dailyRealizedPnl here.
            // e.g. if (tradeResult.IsSuccessful && tradeResult.Position != null)
            //          _dailyRealizedPnl += tradeResult.Position.GrossProfit;
        }


        // ── Helpers ───────────────────────────────────────────────────────────

        /// <summary>
        /// Parses a comma-separated string of UTC hours into an int array.
        /// e.g. "1,2,16,17,18" → [1, 2, 16, 17, 18]
        /// </summary>
        private int[] ParseHours(string hoursString)
        {
            var result = new List<int>();
            foreach (var part in hoursString.Split(','))
            {
                if (int.TryParse(part.Trim(), out int hour))
                    result.Add(hour);
            }
            return result.ToArray();
        }

        /// <summary>
        /// Returns true if the given UTC hour is in the allowed hours list.
        /// Using a loop rather than LINQ to avoid any dependency issues.
        /// </summary>
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
