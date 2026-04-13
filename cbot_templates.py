"""
cbot_templates.py  —  C# cBot template generator
==================================================
Generates parameterised cTrader Automate cBot (.cs) files for each strategy version.
Each version has its own C# template that faithfully mirrors the Python backtest logic.

Called by server.py's /generate_cbot endpoint.
"""


def generate_cbot(strategy_version, params):
    """
    Generate a C# cBot source file for the given strategy version.

    Args:
        strategy_version: "v1", "v2", etc.
        params: dict with keys: ema_short, ema_mid, ema_long, stop_loss_pips,
                rrr_risk, rrr_reward, trade_direction, blocked_hours,
                max_daily_losses, instrument

    Returns:
        tuple of (filename, cs_code_string)
    """
    generators = {
        "v1": _generate_v1,
        "v2": _generate_v2,
    }

    gen = generators.get(strategy_version)
    if gen is None:
        raise ValueError(f"No cBot template for strategy version '{strategy_version}'")

    return gen(params)


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _blocked_to_allowed(blocked_hours_str):
    """Convert a blocked-hours CSV string to an allowed-hours CSV string."""
    if not blocked_hours_str or not blocked_hours_str.strip():
        return ",".join(str(h) for h in range(24))
    blocked = set()
    for part in blocked_hours_str.split(","):
        part = part.strip()
        if part:
            try:
                blocked.add(int(part))
            except ValueError:
                pass
    allowed = sorted(h for h in range(24) if h not in blocked)
    return ",".join(str(h) for h in allowed)


def _p(params, key, default):
    """Safely extract a parameter, converting to the right type."""
    val = params.get(key)
    if val is None or val == "":
        return default
    if isinstance(default, int):
        return int(val)
    if isinstance(default, float):
        return float(val)
    return str(val)


# ── v1: Fractal geometry entries — no EMA alignment ─────────────────────────────

def _generate_v1(params):
    p = params
    ema_short      = _p(p, "ema_short", 8)
    ema_mid        = _p(p, "ema_mid", 20)
    ema_long       = _p(p, "ema_long", 40)
    stop_pips      = _p(p, "stop_loss_pips", 15)
    rrr_risk       = _p(p, "rrr_risk", 1)
    rrr_reward     = _p(p, "rrr_reward", 2)
    max_dd         = _p(p, "max_daily_losses", 2)
    direction      = _p(p, "trade_direction", "both")
    blocked        = _p(p, "blocked_hours", "4,5,6,8,10,11,14,17")
    instrument     = _p(p, "instrument", "EURUSD")
    allowed_hours  = _blocked_to_allowed(blocked)

    # Direction booleans for C# code
    allow_long  = direction != "short_only"
    allow_short = direction != "long_only"

    filename = f"FractalBot_v1_{instrument}.cs"

    cs = f'''\
// FractalBot_v1_{instrument}.cs
// ════════════════════════════════════════════════════════════════
// cTrader Automate cBot — Strategy v1: Fractal Geometry Entries
// Generated from Trading Bot Dashboard
//
// Instrument : {instrument}
// Direction  : {direction.replace("_", " ").title()}
// Entry      : N2 fractal (higher-low = long, lower-high = short)
// Stop       : Fractal price +/- {stop_pips} pip offset
// Target     : Entry +/- (stop distance x {rrr_reward}/{rrr_risk})
// Risk       : 1% of equity per trade
// Time filter: UTC hours {allowed_hours}
// Daily limit: Max {max_dd} losing trades per UTC day
// EMAs       : {ema_short} / {ema_mid} / {ema_long} (chart reference only)
// ════════════════════════════════════════════════════════════════

using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Indicators;

namespace FractalBot
{{
    [Robot(
        TimeZone = TimeZones.UTC,
        AccessRights = AccessRights.None
    )]
    public class FractalBot_v1 : Robot
    {{
        // ── Parameters ───────────────────────────────────────────────────

        [Parameter("EMA Short Period", Group = "EMAs", DefaultValue = {ema_short}, MinValue = 2)]
        public int EmaShortPeriod {{ get; set; }}

        [Parameter("EMA Mid Period", Group = "EMAs", DefaultValue = {ema_mid}, MinValue = 2)]
        public int EmaMidPeriod {{ get; set; }}

        [Parameter("EMA Long Period", Group = "EMAs", DefaultValue = {ema_long}, MinValue = 2)]
        public int EmaLongPeriod {{ get; set; }}

        [Parameter("Fractal Stop Offset (pips)", Group = "Stop", DefaultValue = {stop_pips}, MinValue = 1)]
        public double FractalStopPips {{ get; set; }}

        [Parameter("Min Stop (pips)", Group = "Stop", DefaultValue = 5, MinValue = 1)]
        public double MinStopPips {{ get; set; }}

        [Parameter("Max Stop (pips)", Group = "Stop", DefaultValue = 200, MinValue = 1)]
        public double MaxStopPips {{ get; set; }}

        [Parameter("Risk Percent", Group = "Risk", DefaultValue = 1.0, MinValue = 0.1, MaxValue = 10.0)]
        public double RiskPercent {{ get; set; }}

        [Parameter("RRR Risk", Group = "Risk", DefaultValue = {rrr_risk}, MinValue = 1)]
        public int RrrRisk {{ get; set; }}

        [Parameter("RRR Reward", Group = "Risk", DefaultValue = {rrr_reward}, MinValue = 1)]
        public int RrrReward {{ get; set; }}

        [Parameter("Max Daily Losses", Group = "Risk", DefaultValue = {max_dd}, MinValue = 0)]
        public int MaxDailyLosses {{ get; set; }}

        [Parameter("Allowed Hours UTC (comma-separated)", Group = "Filters",
                   DefaultValue = "{allowed_hours}")]
        public string AllowedHoursUtc {{ get; set; }}


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
        {{
            _emaShort = Indicators.ExponentialMovingAverage(Bars.ClosePrices, EmaShortPeriod);
            _emaMid   = Indicators.ExponentialMovingAverage(Bars.ClosePrices, EmaMidPeriod);
            _emaLong  = Indicators.ExponentialMovingAverage(Bars.ClosePrices, EmaLongPeriod);

            _allowedHours   = ParseHours(AllowedHoursUtc);
            _currentDay     = Server.Time.Date;
            _dailyLossCount = 0;

            Positions.Closed += OnPositionClosed;

            Print("FractalBot v1 started");
            Print($"  EMAs {{EmaShortPeriod}}/{{EmaMidPeriod}}/{{EmaLongPeriod}}  |  " +
                  $"Stop offset {{FractalStopPips}} pips  |  " +
                  $"RRR {{RrrRisk}}:{{RrrReward}}  |  Risk {{RiskPercent}}%  |  " +
                  $"Max daily losses {{MaxDailyLosses}}");
            Print($"  Allowed UTC hours: [{{string.Join(", ", _allowedHours)}}]");
        }}


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
        {{
            // Need at least 5 completed bars for fractal detection
            if (Bars.Count < 6)
                return;

            // ── 1. Daily loss limit reset ────────────────────────────────
            DateTime today = Server.Time.Date;
            if (today != _currentDay)
            {{
                _currentDay     = today;
                _dailyLossCount = 0;
            }}

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
            {{
                _priorFractalHighPrice = _lastFractalHighPrice;
                _lastFractalHighPrice  = fHigh;
                newHighConfirmed = true;
            }}

            if (isPivotLow)
            {{
                _priorFractalLowPrice = _lastFractalLowPrice;
                _lastFractalLowPrice  = fLow;
                newLowConfirmed = true;
            }}


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
            {"if (!longSig && !shortSig) return;" if not allow_long and not allow_short else
             "longSig = false;  // short_only" if not allow_long else
             "shortSig = false;  // long_only" if not allow_short else
             "// Direction: both (no filter)"}

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
            {{
                double slPrice = _lastFractalLowPrice - fractalStopOffset;
                double dist    = close - slPrice;

                if (dist >= minStopPrice && dist <= maxStopPrice)
                {{
                    double stopPips = dist / Symbol.PipSize;
                    double tpPips   = stopPips * rrr;

                    double riskAmount = Account.Equity * (RiskPercent / 100.0);
                    double volume     = Symbol.NormalizeVolumeInUnits(riskAmount / dist);

                    var result = PlaceLimitOrder(
                        TradeType.Buy, Symbol.Name, volume,
                        close, Symbol.Name, stopPips, tpPips,
                        Server.Time.AddMinutes(5)
                    );

                    if (result.IsSuccessful)
                        Print($"[LONG LIMIT]  {{Server.Time:yyyy-MM-dd HH:mm}} UTC | " +
                              $"Entry ~{{close:F5}} | SL {{stopPips:F1}} pips | " +
                              $"TP {{tpPips:F1}} pips | {{volume:F0}} units");
                    else
                        Print($"[ORDER FAILED] {{Server.Time:yyyy-MM-dd HH:mm}} — {{result.Error}}");
                }}
            }}
            else if (shortSig)
            {{
                double slPrice = _lastFractalHighPrice + fractalStopOffset;
                double dist    = slPrice - close;

                if (dist >= minStopPrice && dist <= maxStopPrice)
                {{
                    double stopPips = dist / Symbol.PipSize;
                    double tpPips   = stopPips * rrr;

                    double riskAmount = Account.Equity * (RiskPercent / 100.0);
                    double volume     = Symbol.NormalizeVolumeInUnits(riskAmount / dist);

                    var result = PlaceLimitOrder(
                        TradeType.Sell, Symbol.Name, volume,
                        close, Symbol.Name, stopPips, tpPips,
                        Server.Time.AddMinutes(5)
                    );

                    if (result.IsSuccessful)
                        Print($"[SHORT LIMIT] {{Server.Time:yyyy-MM-dd HH:mm}} UTC | " +
                              $"Entry ~{{close:F5}} | SL {{stopPips:F1}} pips | " +
                              $"TP {{tpPips:F1}} pips | {{volume:F0}} units");
                    else
                        Print($"[ORDER FAILED] {{Server.Time:yyyy-MM-dd HH:mm}} — {{result.Error}}");
                }}
            }}
        }}


        // ── Daily loss tracking ──────────────────────────────────────────

        private void OnPositionClosed(PositionClosedEventArgs args)
        {{
            if (args.Position.Label != Symbol.Name)
                return;

            _lastCloseBar = Bars.Count;

            if (args.Position.NetProfit < 0)
                _dailyLossCount++;
        }}


        // ── Helpers ──────────────────────────────────────────────────────

        private int[] ParseHours(string hoursString)
        {{
            var result = new List<int>();
            if (string.IsNullOrWhiteSpace(hoursString))
                return result.ToArray();
            foreach (var part in hoursString.Split(','))
            {{
                if (int.TryParse(part.Trim(), out int hour))
                    result.Add(hour);
            }}
            return result.ToArray();
        }}

        private bool IsAllowedHour(int hour)
        {{
            foreach (int h in _allowedHours)
            {{
                if (h == hour) return true;
            }}
            return false;
        }}
    }}
}}
'''
    return (filename, cs)


# ── v2: Fractal geometry + EMA position filter ──────────────────────────────────

def _generate_v2(params):
    p = params
    ema_short      = _p(p, "ema_short", 8)
    ema_mid        = _p(p, "ema_mid", 20)
    ema_long       = _p(p, "ema_long", 40)
    stop_pips      = _p(p, "stop_loss_pips", 15)
    rrr_risk       = _p(p, "rrr_risk", 1)
    rrr_reward     = _p(p, "rrr_reward", 2)
    max_dd         = _p(p, "max_daily_losses", 2)
    direction      = _p(p, "trade_direction", "both")
    blocked        = _p(p, "blocked_hours", "4,5,6,8,10,11,14,17")
    instrument     = _p(p, "instrument", "EURUSD")
    allowed_hours  = _blocked_to_allowed(blocked)

    allow_long  = direction != "short_only"
    allow_short = direction != "long_only"

    filename = f"FractalBot_v2_{instrument}.cs"

    cs = f'''\
// FractalBot_v2_{instrument}.cs
// ════════════════════════════════════════════════════════════════
// cTrader Automate cBot — Strategy v2: Fractal + EMA Position Filter
// Generated from Trading Bot Dashboard
//
// Instrument : {instrument}
// Direction  : {direction.replace("_", " ").title()}
// Entry      : N2 fractal (higher-low = long, lower-high = short)
//              + EMA position filter (long above / short below EMA Long)
// Stop       : Fractal price +/- {stop_pips} pip offset
// Target     : Entry +/- (stop distance x {rrr_reward}/{rrr_risk})
// Risk       : 1% of equity per trade
// Time filter: UTC hours {allowed_hours}
// Daily limit: Max {max_dd} losing trades per UTC day
// EMAs       : {ema_short} / {ema_mid} / {ema_long}
// ════════════════════════════════════════════════════════════════

using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Indicators;

namespace FractalBot
{{
    [Robot(
        TimeZone = TimeZones.UTC,
        AccessRights = AccessRights.None
    )]
    public class FractalBot_v2 : Robot
    {{
        // ── Parameters ───────────────────────────────────────────────────

        [Parameter("EMA Short Period", Group = "EMAs", DefaultValue = {ema_short}, MinValue = 2)]
        public int EmaShortPeriod {{ get; set; }}

        [Parameter("EMA Mid Period", Group = "EMAs", DefaultValue = {ema_mid}, MinValue = 2)]
        public int EmaMidPeriod {{ get; set; }}

        [Parameter("EMA Long Period", Group = "EMAs", DefaultValue = {ema_long}, MinValue = 2)]
        public int EmaLongPeriod {{ get; set; }}

        [Parameter("Fractal Stop Offset (pips)", Group = "Stop", DefaultValue = {stop_pips}, MinValue = 1)]
        public double FractalStopPips {{ get; set; }}

        [Parameter("Min Stop (pips)", Group = "Stop", DefaultValue = 5, MinValue = 1)]
        public double MinStopPips {{ get; set; }}

        [Parameter("Max Stop (pips)", Group = "Stop", DefaultValue = 200, MinValue = 1)]
        public double MaxStopPips {{ get; set; }}

        [Parameter("Risk Percent", Group = "Risk", DefaultValue = 1.0, MinValue = 0.1, MaxValue = 10.0)]
        public double RiskPercent {{ get; set; }}

        [Parameter("RRR Risk", Group = "Risk", DefaultValue = {rrr_risk}, MinValue = 1)]
        public int RrrRisk {{ get; set; }}

        [Parameter("RRR Reward", Group = "Risk", DefaultValue = {rrr_reward}, MinValue = 1)]
        public int RrrReward {{ get; set; }}

        [Parameter("Max Daily Losses", Group = "Risk", DefaultValue = {max_dd}, MinValue = 0)]
        public int MaxDailyLosses {{ get; set; }}

        [Parameter("Allowed Hours UTC (comma-separated)", Group = "Filters",
                   DefaultValue = "{allowed_hours}")]
        public string AllowedHoursUtc {{ get; set; }}


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
        {{
            _emaShort = Indicators.ExponentialMovingAverage(Bars.ClosePrices, EmaShortPeriod);
            _emaMid   = Indicators.ExponentialMovingAverage(Bars.ClosePrices, EmaMidPeriod);
            _emaLong  = Indicators.ExponentialMovingAverage(Bars.ClosePrices, EmaLongPeriod);

            _allowedHours   = ParseHours(AllowedHoursUtc);
            _currentDay     = Server.Time.Date;
            _dailyLossCount = 0;

            Positions.Closed += OnPositionClosed;

            Print("FractalBot v2 started (EMA position filter active)");
            Print($"  EMAs {{EmaShortPeriod}}/{{EmaMidPeriod}}/{{EmaLongPeriod}}  |  " +
                  $"Stop offset {{FractalStopPips}} pips  |  " +
                  $"RRR {{RrrRisk}}:{{RrrReward}}  |  Risk {{RiskPercent}}%  |  " +
                  $"Max daily losses {{MaxDailyLosses}}");
            Print($"  Allowed UTC hours: [{{string.Join(", ", _allowedHours)}}]");
        }}


        // ── Main logic: OnBar ────────────────────────────────────────────
        //
        // Execution order mirrors Python backtest:
        //   1. Daily loss limit reset
        //   2. Fractal detection (confirmed at bar i-2)
        //   3. Entry signal generation (higher-low / lower-high)
        //   4. Direction filter
        //   5. EMA position filter (v2: long above EMA Long, short below)
        //   6. Daily loss limit check
        //   7. Time filter (allowed hours)
        //   8. Stop validation & order execution

        protected override void OnBar()
        {{
            if (Bars.Count < 6)
                return;

            // ── 1. Daily loss limit reset ────────────────────────────────
            DateTime today = Server.Time.Date;
            if (today != _currentDay)
            {{
                _currentDay     = today;
                _dailyLossCount = 0;
            }}

            // ── 2. Fractal detection ─────────────────────────────────────

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
            {{
                _priorFractalHighPrice = _lastFractalHighPrice;
                _lastFractalHighPrice  = fHigh;
                newHighConfirmed = true;
            }}

            if (isPivotLow)
            {{
                _priorFractalLowPrice = _lastFractalLowPrice;
                _lastFractalLowPrice  = fLow;
                newLowConfirmed = true;
            }}


            // ── Skip if already in a trade ───────────────────────────────
            if (Positions.Find(Symbol.Name) != null)
                return;

            // ── One-bar cooldown after close ─────────────────────────────
            if (Bars.Count <= _lastCloseBar + 1)
                return;

            double close = Bars.ClosePrices.Last(1);


            // ── 3. Entry signals ─────────────────────────────────────────

            bool longSig = newLowConfirmed
                        && !double.IsNaN(_lastFractalLowPrice)
                        && !double.IsNaN(_priorFractalLowPrice)
                        && _lastFractalLowPrice > _priorFractalLowPrice;

            bool shortSig = newHighConfirmed
                         && !double.IsNaN(_lastFractalHighPrice)
                         && !double.IsNaN(_priorFractalHighPrice)
                         && _lastFractalHighPrice < _priorFractalHighPrice;


            // ── 4. Direction filter ──────────────────────────────────────
            {"if (!longSig && !shortSig) return;" if not allow_long and not allow_short else
             "longSig = false;  // short_only" if not allow_long else
             "shortSig = false;  // long_only" if not allow_short else
             "// Direction: both (no filter)"}


            // ── 5. EMA position filter (v2) ──────────────────────────────
            // Long entries require close above EMA Long.
            // Short entries require close below EMA Long.
            double emaLongVal = _emaLong.Result.Last(1);
            if (longSig && close <= emaLongVal)
                longSig = false;
            if (shortSig && close >= emaLongVal)
                shortSig = false;

            if (!longSig && !shortSig)
                return;


            // ── 6. Daily loss limit ──────────────────────────────────────
            if (_dailyLossCount >= MaxDailyLosses)
                return;


            // ── 7. Time filter ───────────────────────────────────────────
            if (!IsAllowedHour(Server.Time.Hour))
                return;


            // ── 8. Stop validation & execution ───────────────────────────
            double fractalStopOffset = FractalStopPips * Symbol.PipSize;
            double minStopPrice      = MinStopPips * Symbol.PipSize;
            double maxStopPrice      = MaxStopPips * Symbol.PipSize;
            double rrr               = (double)RrrReward / RrrRisk;

            if (longSig)
            {{
                double slPrice = _lastFractalLowPrice - fractalStopOffset;
                double dist    = close - slPrice;

                if (dist >= minStopPrice && dist <= maxStopPrice)
                {{
                    double stopPips = dist / Symbol.PipSize;
                    double tpPips   = stopPips * rrr;

                    double riskAmount = Account.Equity * (RiskPercent / 100.0);
                    double volume     = Symbol.NormalizeVolumeInUnits(riskAmount / dist);

                    var result = PlaceLimitOrder(
                        TradeType.Buy, Symbol.Name, volume,
                        close, Symbol.Name, stopPips, tpPips,
                        Server.Time.AddMinutes(5)
                    );

                    if (result.IsSuccessful)
                        Print($"[LONG LIMIT]  {{Server.Time:yyyy-MM-dd HH:mm}} UTC | " +
                              $"Entry ~{{close:F5}} | SL {{stopPips:F1}} pips | " +
                              $"TP {{tpPips:F1}} pips | {{volume:F0}} units | " +
                              $"EMA Long {{emaLongVal:F5}}");
                    else
                        Print($"[ORDER FAILED] {{Server.Time:yyyy-MM-dd HH:mm}} — {{result.Error}}");
                }}
            }}
            else if (shortSig)
            {{
                double slPrice = _lastFractalHighPrice + fractalStopOffset;
                double dist    = slPrice - close;

                if (dist >= minStopPrice && dist <= maxStopPrice)
                {{
                    double stopPips = dist / Symbol.PipSize;
                    double tpPips   = stopPips * rrr;

                    double riskAmount = Account.Equity * (RiskPercent / 100.0);
                    double volume     = Symbol.NormalizeVolumeInUnits(riskAmount / dist);

                    var result = PlaceLimitOrder(
                        TradeType.Sell, Symbol.Name, volume,
                        close, Symbol.Name, stopPips, tpPips,
                        Server.Time.AddMinutes(5)
                    );

                    if (result.IsSuccessful)
                        Print($"[SHORT LIMIT] {{Server.Time:yyyy-MM-dd HH:mm}} UTC | " +
                              $"Entry ~{{close:F5}} | SL {{stopPips:F1}} pips | " +
                              $"TP {{tpPips:F1}} pips | {{volume:F0}} units | " +
                              $"EMA Long {{emaLongVal:F5}}");
                    else
                        Print($"[ORDER FAILED] {{Server.Time:yyyy-MM-dd HH:mm}} — {{result.Error}}");
                }}
            }}
        }}


        // ── Daily loss tracking ──────────────────────────────────────────

        private void OnPositionClosed(PositionClosedEventArgs args)
        {{
            if (args.Position.Label != Symbol.Name)
                return;

            _lastCloseBar = Bars.Count;

            if (args.Position.NetProfit < 0)
                _dailyLossCount++;
        }}


        // ── Helpers ──────────────────────────────────────────────────────

        private int[] ParseHours(string hoursString)
        {{
            var result = new List<int>();
            if (string.IsNullOrWhiteSpace(hoursString))
                return result.ToArray();
            foreach (var part in hoursString.Split(','))
            {{
                if (int.TryParse(part.Trim(), out int hour))
                    result.Add(hour);
            }}
            return result.ToArray();
        }}

        private bool IsAllowedHour(int hour)
        {{
            foreach (int h in _allowedHours)
            {{
                if (h == hour) return true;
            }}
            return false;
        }}
    }}
}}
'''
    return (filename, cs)
