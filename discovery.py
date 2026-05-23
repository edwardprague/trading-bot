"""
discovery.py — Phase 1: Random search over the v2 parameter space
==================================================================

Runs N random-sampled backtest trials by launching strategy.py (routed
to strategy_v2.py via STRATEGY_VERSION=v2) as a subprocess for each trial
with DISCOVERY_MODE=1. The strategy module short-circuits its usual
report.html / chart / git side effects and writes a slim metrics JSON
to the path in DISCOVERY_METRICS_OUT. We harvest that file, evaluate
pass/fail against the Phase 1 objective (PF >= 1.5, trades >= 50,
max DD <= 15%), and append the result to data/discovery_results.json
atomically after every trial so the Flask /api/discovery/status endpoint
can poll progress while the run is in flight.

Phase 1 scope (intentionally narrow):
  - Random search only (no Optuna, no walk-forward, no clustering)
  - Single sequential subprocess (no parallelism)
  - Fixed instrument / interval / direction / blocked hours / slippage

Usage:
    # Normal random search (200 trials over default range)
    python3 discovery.py --trials 200 --start 2025-07-01 --end 2025-12-31

    # Single-trial mode for sanity-checking a known parameter set
    python3 discovery.py --once \
        --ema-long 50 --stop-loss 10 --rrr-reward 1 --max-dll 3 \
        --ema-filter on \
        --macro staircase_down,strong_down \
        --micro ranging_medium,ranging_wide,transitioning,trending_fast_down,trending_slow_down

    # Headless from a JSON config (used by server.py /api/discovery/run)
    python3 discovery.py --config-json /path/to/config.json
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

BASE_DIR             = Path(__file__).parent
STRATEGY_FILE        = BASE_DIR / "strategy.py"
DATA_DIR             = BASE_DIR / "data"
RESULTS_FILE_DEFAULT = DATA_DIR / "discovery_results.json"
DISCOVERY_TMP_DIR    = DATA_DIR / ".discovery_tmp"

# ── Search space (Phase 1) ────────────────────────────────────────────────────
EMA_LONG_RANGE      = (10, 200)
STOP_LOSS_RANGE     = (5, 50)
RRR_REWARD_RANGE    = (1, 5)
# MAX_DAILY_LOSS_RANGE was retired (May 2026). The 1–5 sweep over
# MAX_DAILY_LOSSES (daily-loss-stop count) was a legacy of an earlier
# search-space design and no longer matches how the strategy is used —
# the daily-loss stop is now treated as a fixed strategy safety rail
# rather than a tunable Discovery dimension. We pin it to the strategy's
# documented default (2) via FIXED_MAX_DAILY_LOSSES below so every trial
# still receives a valid value through the env.

ALL_MACRO_REGIMES = ["strong_down", "staircase_down", "flat", "staircase_up", "strong_up"]
ALL_MICRO_REGIMES = [
    "trending_fast_down", "trending_medium_down", "trending_slow_down",
    "trending_fast_up",   "trending_medium_up",   "trending_slow_up",
    "ranging_narrow", "ranging_medium", "ranging_wide", "transitioning",
]

# ── Fixed constants (Phase 1) ─────────────────────────────────────────────────
FIXED_INSTRUMENT        = "GBPUSD"
FIXED_INTERVAL          = "5m"
FIXED_DIRECTION         = "short_only"
FIXED_EMA_SHORT         = 8
FIXED_EMA_MID           = 20
FIXED_BLOCKED_HOURS     = "4,5,6,8,10,11,14,17"  # v3's current values
FIXED_APPLY_SLIPPAGE    = "true"
FIXED_SPREAD_PIPS       = "1.0"
FIXED_SL_SLIPPAGE       = "1.0"
FIXED_STRATEGY_VER      = "v2"
# May 2026: MAX_DAILY_LOSSES is no longer searched. Pinned to the
# strategy's documented default so every trial receives a valid env var.
FIXED_MAX_DAILY_LOSSES  = 2

# ── Objective function (Phase 1) ──────────────────────────────────────────────
OBJ_PROFIT_FACTOR_MIN     = 1.5
OBJ_TRADES_MIN            = 50
OBJ_MAX_DRAWDOWN_MAX      = 10.0   # DD1 cap (peak-to-trough, percent). May 2026
                                   # tightened from 15% → 10% so the default
                                   # passing posture matches the risk profile
                                   # the user actually wants.
OBJ_MAX_DAILY_DRAWDOWN_MAX = 5.0   # DD2 cap (worst single-day drawdown, percent).
                                   # New passing criterion (May 2026).

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_TRIALS = 200
DEFAULT_START  = "2025-07-01"
DEFAULT_END    = "2025-12-31"


# ── Config-override helpers ───────────────────────────────────────────────────
# Bug fix / feature (May 2026): the Discovery Settings panel on /discovery
# now exposes Instrument / Interval / Direction / Blocked Hours and the
# passing-criteria thresholds as editable fields. The server.py
# /api/discovery/run handler writes them into the config JSON we pick up
# via --config-json; the helpers below resolve config → values with the
# original Phase 1 constants as defaults so the CLI --once path keeps
# working with no JSON.

def resolve_settings(config):
    """Pull the editable fixed-constants out of the run config (with
    FIXED_* defaults). Returns a dict consumed by sample_params + build_env.

    May 2026 — added `use_ema_filter` and `rrr_reward_max`. EMA filter was
    promoted from a per-trial searched dimension to a per-run fixed
    constant (transparency: every trial in a run carries the same setting).
    `rrr_reward_max` exposes the previously-hardcoded RRR upper bound of
    5 as an editable per-run knob — the lower bound stays fixed at 1.

    May 2026 (Regime Filters) — added `allowed_macro_regimes` /
    `allowed_micro_regimes`. These constrain the per-trial random subset
    search to the user-selected pool: every trial's allow-list is sampled
    from these lists, so toggled-off regimes (the complement) are guaranteed
    locked out of every trial. Missing / null → defaults to the full
    canonical set, preserving the legacy "any combination of 5/10" behaviour
    for CLI callers that don't pass these keys."""
    cfg = config or {}
    # use_ema_filter: accept bool or string ("true"/"false"/"on"/"off").
    raw_ema = cfg.get("use_ema_filter")
    if isinstance(raw_ema, bool):
        ema_on = raw_ema
    elif isinstance(raw_ema, str):
        ema_on = raw_ema.strip().lower() in ("true", "1", "on", "yes")
    else:
        ema_on = True  # default matches the historical p=0.5 sampling default-ish + strategy default
    # rrr_reward_max: int, defaulted to the original Phase 1 upper bound.
    try:
        rrr_max = int(cfg.get("rrr_reward_max")) if cfg.get("rrr_reward_max") is not None else RRR_REWARD_RANGE[1]
    except (TypeError, ValueError):
        rrr_max = RRR_REWARD_RANGE[1]
    # Clamp into a sensible band — sample_params calls rng.randint(1, max)
    # so a max < 1 would crash; an absurdly large max would inflate variance.
    if rrr_max < 1:
        rrr_max = 1
    if rrr_max > 100:
        rrr_max = 100

    # ── Regime Filters ───────────────────────────────────────────────────
    # Accept list / CSV string / None. Intersect with the canonical sets so a
    # stale key from an older results file (e.g. a renamed regime) can't
    # poison the search. Empty after filtering → fall back to the canonical
    # set; the server endpoint already rejects user-facing empty lists, this
    # is the CLI safety net.
    def _norm_regime_list(raw, canonical):
        if raw is None:
            return list(canonical)
        if isinstance(raw, str):
            items = [s.strip() for s in raw.split(",")]
        elif isinstance(raw, (list, tuple)):
            items = [str(s).strip() for s in raw]
        else:
            return list(canonical)
        valid = [s for s in items if s in canonical]
        # Preserve canonical order regardless of how the caller supplied them
        # so downstream display + CSV-equality checks are stable.
        ordered = [k for k in canonical if k in valid]
        return ordered if ordered else list(canonical)

    macro_allowed = _norm_regime_list(cfg.get("allowed_macro_regimes"), ALL_MACRO_REGIMES)
    micro_allowed = _norm_regime_list(cfg.get("allowed_micro_regimes"), ALL_MICRO_REGIMES)

    return {
        "instrument":      (cfg.get("instrument")    or FIXED_INSTRUMENT).upper(),
        "interval":         cfg.get("interval")      or FIXED_INTERVAL,
        "direction":        cfg.get("direction")     or FIXED_DIRECTION,
        "blocked_hours":    cfg.get("blocked_hours") or FIXED_BLOCKED_HOURS,
        "use_ema_filter":   ema_on,
        "rrr_reward_max":   rrr_max,
        "allowed_macro_regimes": macro_allowed,
        "allowed_micro_regimes": micro_allowed,
    }


def resolve_thresholds(config):
    """Pull the editable passing-criteria thresholds out of the run config
    (with OBJ_* defaults). Returns a dict consumed by evaluate_objective.

    May 2026 — added max_daily_dd_pct (DD2 cap, default 5%). The existing
    max_dd_pct now defaults to 10% (was 15%) to match the tighter risk
    posture the user wants by default. Both values are still editable per
    run via the Discovery Settings panel's Passing Criteria section."""
    cfg = config or {}
    return {
        "min_pf":           float(cfg.get("min_pf")           if cfg.get("min_pf")           is not None else OBJ_PROFIT_FACTOR_MIN),
        "min_trades":       int(  cfg.get("min_trades")       if cfg.get("min_trades")       is not None else OBJ_TRADES_MIN),
        "max_dd_pct":       float(cfg.get("max_dd_pct")       if cfg.get("max_dd_pct")       is not None else OBJ_MAX_DRAWDOWN_MAX),
        "max_daily_dd_pct": float(cfg.get("max_daily_dd_pct") if cfg.get("max_daily_dd_pct") is not None else OBJ_MAX_DAILY_DRAWDOWN_MAX),
    }


# ── Sampling ──────────────────────────────────────────────────────────────────

def _sample_subset(labels, rng):
    """Bernoulli p=0.5 per label; resample if the subset comes back empty.
    Empty allow-lists would block 100% of trades, which is degenerate, so we
    guarantee at least one label."""
    while True:
        picked = [lbl for lbl in labels if rng.random() < 0.5]
        if picked:
            return picked


def sample_params(rng, settings=None):
    """Draw a single parameter set uniformly from the Phase 1 search space.
    `settings` (from resolve_settings) supplies blocked_hours, use_ema_filter
    (fixed per-run since May 2026), and rrr_reward_max (the editable upper
    bound of the RRR Reward sweep, lower bound fixed at 1). The rest of the
    per-trial dimensions are sampled. blocked_hours, use_ema_filter, and
    rrr_reward_max are NOT randomised — Discovery applies the same value
    to every trial in a run per the user's intent (transparency over
    search-space breadth).

    May 2026: each trial's params dict is now self-describing — the
    per-run fixed constants (instrument, interval, direction,
    blocked_hours, use_ema_filter, max_daily_losses) are folded into
    every returned record. Previously these lived only at the run's
    top-level config dict and downstream consumers (the trial detail
    page, version assignment) would silently fall back to defaults when
    the run used non-default values. Storing them per trial makes the
    record self-describing without changing search behaviour — they're
    still identical across every trial in a single run.

    max_daily_losses was retired from the SEARCH space (it was a legacy
    1–5 sweep over a strategy safety rail, not a true tunable) and is
    pinned to FIXED_MAX_DAILY_LOSSES; we still emit the field so the
    trial detail page can render it instead of showing an em-dash."""
    s = settings or resolve_settings(None)
    rrr_lo = RRR_REWARD_RANGE[0]
    rrr_hi = int(s.get("rrr_reward_max") or RRR_REWARD_RANGE[1])
    if rrr_hi < rrr_lo:
        rrr_hi = rrr_lo  # guard against an upper < lower edit
    # Per-run regime allow-lists (Regime Filters, May 2026). Sample each
    # trial's allow-list from this pool instead of the full canonical set
    # so toggled-off regimes are guaranteed locked out of every trial.
    # Falls back to the canonical set when settings omits the keys (CLI
    # --once / legacy callers).
    macro_pool = s.get("allowed_macro_regimes") or ALL_MACRO_REGIMES
    micro_pool = s.get("allowed_micro_regimes") or ALL_MICRO_REGIMES
    return {
        "ema_long":              rng.randint(*EMA_LONG_RANGE),
        "stop_loss_pips":        rng.randint(*STOP_LOSS_RANGE),
        "rrr_risk":              1,
        "rrr_reward":            rng.randint(rrr_lo, rrr_hi),
        "use_ema_filter":        bool(s.get("use_ema_filter", True)),
        "allowed_macro_regimes": _sample_subset(macro_pool, rng),
        "allowed_micro_regimes": _sample_subset(micro_pool, rng),
        "blocked_hours":         s["blocked_hours"],
        # Per-run fixed constants (May 2026) — duplicated into every trial's
        # params dict so the trial record is self-describing.
        "instrument":            s["instrument"],
        "interval":              s["interval"],
        "direction":             s["direction"],
        "max_daily_losses":      FIXED_MAX_DAILY_LOSSES,
    }


# ── Trial execution ───────────────────────────────────────────────────────────

def build_env(params, start_date, end_date, settings=None):
    """Build the env dict for one strategy_v2 subprocess call. `settings`
    (from resolve_settings) supplies instrument/interval/direction/blocked
    hours — defaults via resolve_settings(None) for CLI / --once paths."""
    s = settings or resolve_settings(None)
    env = os.environ.copy()
    env.update({
        "DISCOVERY_MODE":        "1",
        "STRATEGY_VERSION":      FIXED_STRATEGY_VER,
        "RUN_MODE":              "date_range",
        "RUN_START_DATE":        start_date,
        "RUN_END_DATE":          end_date,
        "INSTRUMENT":            s["instrument"],
        "INTERVAL":              s["interval"],
        "TRADE_DIRECTION":       s["direction"],
        "EMA_SHORT":             str(FIXED_EMA_SHORT),
        "EMA_MID":               str(FIXED_EMA_MID),
        "EMA_LONG":              str(params["ema_long"]),
        "FRACTAL_STOP_PIPS":     str(params["stop_loss_pips"]),
        "RRR_RISK":              str(params["rrr_risk"]),
        "RRR_REWARD":            str(params["rrr_reward"]),
        # MAX_DAILY_LOSSES is fixed per-run now (May 2026) — see
        # FIXED_MAX_DAILY_LOSSES at the top of this file.
        "MAX_DAILY_LOSSES":      str(FIXED_MAX_DAILY_LOSSES),
        "USE_EMA_FILTER":        "true" if params["use_ema_filter"] else "false",
        "BLOCKED_HOURS_UTC":     params["blocked_hours"],
        "ALLOWED_MACRO_REGIMES": ",".join(params["allowed_macro_regimes"]),
        "ALLOWED_MICRO_REGIMES": ",".join(params["allowed_micro_regimes"]),
        "APPLY_SLIPPAGE":        FIXED_APPLY_SLIPPAGE,
        "SPREAD_PIPS":           FIXED_SPREAD_PIPS,
        "SL_SLIPPAGE_PIPS":      FIXED_SL_SLIPPAGE,
    })
    return env


def evaluate_objective(metrics, thresholds=None):
    """Return (pass, reasons[]). PF=None (∞) treated as passing the PF bar.

    `thresholds` (from resolve_thresholds) supplies min_pf / min_trades /
    max_dd_pct / max_daily_dd_pct — defaults to the Phase 1 OBJ_*
    constants. Both drawdown checks compare as absolute magnitudes:
    strategy_v2's compute_metrics writes drawdown with a negative-sign
    convention, but the objective "max DD ≤ X%" reads naturally as a
    positive cap — without the abs() here, a catastrophic -50% drawdown
    would silently pass because -50 < X for any reasonable X.

    DD1 (max_drawdown):       peak-to-trough drawdown across the run.
    DD2 (max_daily_drawdown): worst single-day drawdown (stored as a
                              {dollar, pct} dict by strategy_v2; we read
                              .pct). New since May 2026."""
    t = thresholds or resolve_thresholds(None)
    reasons = []
    pf       = metrics.get("profit_factor")
    trades   = metrics.get("total_trades", 0) or 0
    max_dd   = abs(metrics.get("max_drawdown", 0.0) or 0.0)
    # DD2 is stored as {"dollar": …, "pct": …}; coerce defensively because
    # legacy metric records may omit it or store a flat scalar.
    mdd_raw  = metrics.get("max_daily_drawdown")
    if isinstance(mdd_raw, dict):
        max_daily_dd = abs(float(mdd_raw.get("pct") or 0.0))
    elif mdd_raw is None:
        max_daily_dd = 0.0
    else:
        max_daily_dd = abs(float(mdd_raw))

    if pf is not None and pf < t["min_pf"]:
        reasons.append(f"profit_factor {pf:.2f} < {t['min_pf']}")
    if trades < t["min_trades"]:
        reasons.append(f"trades {trades} < {t['min_trades']}")
    if max_dd > t["max_dd_pct"]:
        reasons.append(f"max_drawdown {max_dd:.1f}% > {t['max_dd_pct']}%")
    if max_daily_dd > t["max_daily_dd_pct"]:
        reasons.append(f"max_daily_drawdown {max_daily_dd:.1f}% > {t['max_daily_dd_pct']}%")
    return (len(reasons) == 0, reasons)


def run_trial(trial_num, params, start_date, end_date, settings=None, thresholds=None):
    """Run a single strategy_v2 subprocess; return a trial-result dict.
    `settings` and `thresholds` (from resolve_settings / resolve_thresholds)
    are threaded through to build_env + evaluate_objective so editable
    Discovery Settings reach the subprocess + the pass/fail evaluation."""
    DISCOVERY_TMP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DISCOVERY_TMP_DIR / f"trial_{uuid.uuid4().hex}.json"
    env = build_env(params, start_date, end_date, settings=settings)
    env["DISCOVERY_METRICS_OUT"] = str(out_path)

    started = time.time()
    metrics = None
    error   = None
    try:
        proc = subprocess.run(
            [sys.executable, str(STRATEGY_FILE)],
            env=env,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=600,  # 10 min safety cap per trial
        )
        if proc.returncode != 0:
            # Strategy aborted (e.g. no data, no trades after filtering, exception).
            # Truncate stderr so the results file doesn't bloat on noisy failures.
            tail = (proc.stderr or "").strip().splitlines()[-8:]
            error = "subprocess exited %d: %s" % (proc.returncode, " | ".join(tail))
        elif out_path.exists():
            with open(out_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            # Normalize the sign convention: strategy_v2 stores drawdown as a
            # negative number. Phase 1 treats max DD as a positive magnitude
            # everywhere downstream (objective check, UI display, threshold
            # tooltips), so we abs() it once here at the boundary.
            if metrics.get("max_drawdown") is not None:
                metrics["max_drawdown"] = abs(metrics["max_drawdown"])
        else:
            error = "metrics file not written"
    except subprocess.TimeoutExpired:
        error = "trial timed out after 600s"
    except Exception as e:
        error = f"trial crashed: {e}"
    finally:
        try:
            if out_path.exists():
                out_path.unlink()
        except OSError:
            pass

    duration = round(time.time() - started, 2)

    if metrics is None:
        # Treat any failure as a non-passing trial with zero trades. Recorded
        # so the results table can show the failure for debugging without
        # actually counting it as a candidate.
        metrics = {
            "total_trades": 0, "win_rate": 0.0, "profit_factor": None,
            "net_profit": 0.0, "net_profit_pct": 0.0, "max_drawdown": 0.0,
        }
        passed   = False
        reasons  = ["trial errored"]
    else:
        passed, reasons = evaluate_objective(metrics, thresholds=thresholds)

    return {
        "id":           f"t{trial_num}_{uuid.uuid4().hex[:8]}",
        "trial":        trial_num,
        "params":       params,
        "metrics":      metrics,
        "pass":         passed,
        "fail_reasons": reasons,
        "error":        error,
        "duration_sec": duration,
    }


# ── Results-file management ──────────────────────────────────────────────────

def _atomic_write(path, payload):
    """Write JSON atomically so a polling reader never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def _best_passing(trials):
    """Highest profit_factor among passing trials (PF=None counts as ∞ → best)."""
    passing = [t for t in trials if t.get("pass")]
    if not passing:
        return None
    def _key(t):
        pf = t.get("metrics", {}).get("profit_factor")
        # None (∞) ranks above any finite value; otherwise sort by PF descending
        return (pf is None, pf if pf is not None else float("inf"))
    passing.sort(key=_key, reverse=True)
    return passing[0]


def _load_runs(results_path):
    """Read the discovery results file as a list of runs. Handles:
      • missing/empty/invalid file → []
      • new array-of-runs schema   → returned as-is
      • legacy single-object schema (one run dict at the top level) →
        wrapped into [dict] so the caller transparently sees the array
        format. Doesn't persist the migration here — init_results_file's
        atomic write below does that on the next run start.
    """
    if not results_path.exists():
        return []
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and data.get("run_id"):
        return [data]
    return []


def _write_run_into_array(results_path, run_payload):
    """Read the runs array, replace (or prepend) the entry matching
    run_payload['run_id'], write back atomically. If a concurrent DELETE
    removed this run from the file between writes, we re-prepend it so the
    in-flight run's progress isn't silently dropped."""
    runs = _load_runs(results_path)
    rid = run_payload.get("run_id")
    for i, r in enumerate(runs):
        if r.get("run_id") == rid:
            runs[i] = run_payload
            break
    else:
        runs.insert(0, run_payload)
    _atomic_write(results_path, runs)


def init_results_file(results_path, run_id, config):
    """Start a new run: prepend a fresh run entry at the front of the
    array (newest first) and persist. Returns the in-memory payload dict
    that append_trial / finalize will keep updating."""
    payload = {
        "run_id":          run_id,
        "started_at":      datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "finished_at":     None,
        "config":          config,
        "status":          "running",
        "trials_complete": 0,
        "trials_total":    config.get("trials", 0),
        "best":            None,
        "trials":          [],
    }
    runs = _load_runs(results_path)
    runs.insert(0, payload)
    _atomic_write(results_path, runs)
    return payload


def append_trial(results_path, payload, trial_record):
    """Append one trial to the in-flight run's entry and persist the array."""
    payload["trials"].append(trial_record)
    payload["trials_complete"] = len(payload["trials"])
    payload["best"] = _best_passing(payload["trials"])
    _write_run_into_array(results_path, payload)


def finalize(results_path, payload, status="complete", error=None):
    payload["status"]      = status
    payload["finished_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    if error:
        payload["error"] = error
    _write_run_into_array(results_path, payload)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv):
    p = argparse.ArgumentParser(description="Phase 1 random search over strategy_v2 parameter space")
    p.add_argument("--trials",       type=int, default=DEFAULT_TRIALS, help="Number of trials")
    p.add_argument("--start",        type=str, default=DEFAULT_START,  help="Start date YYYY-MM-DD")
    p.add_argument("--end",          type=str, default=DEFAULT_END,    help="End date YYYY-MM-DD")
    p.add_argument("--seed",         type=int, default=None,           help="Random seed (optional, for reproducibility)")
    p.add_argument("--results-file", type=str, default=str(RESULTS_FILE_DEFAULT))
    p.add_argument("--config-json",  type=str, default=None, help="Path to JSON file with {trials,start,end,seed}")

    # Single-trial sanity-check mode
    p.add_argument("--once",       action="store_true", help="Run a single trial with explicit params (sanity check)")
    p.add_argument("--ema-long",   type=int)
    p.add_argument("--stop-loss",  type=int)
    p.add_argument("--rrr-reward", type=int)
    p.add_argument("--max-dll",    type=int)
    p.add_argument("--ema-filter", choices=["on", "off"], default="on")
    p.add_argument("--macro",      type=str, help="Comma-separated macro regime labels")
    p.add_argument("--micro",      type=str, help="Comma-separated micro regime labels")

    args = p.parse_args(argv)

    args.full_config = {}  # picked up by main() for resolve_settings/thresholds
    if args.config_json:
        with open(args.config_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        args.full_config = cfg
        args.trials = int(cfg.get("trials", args.trials))
        args.start  = cfg.get("start",  args.start)
        args.end    = cfg.get("end",    args.end)
        if cfg.get("seed") is not None:
            args.seed = int(cfg["seed"])
        if cfg.get("results_file"):
            args.results_file = cfg["results_file"]
    return args


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])

    # Resolve editable settings + thresholds from --config-json (with the
    # Phase 1 constants as defaults if anything's missing). Both modes
    # (--once and normal random search) use these.
    settings   = resolve_settings(args.full_config)
    thresholds = resolve_thresholds(args.full_config)

    # ── --once: deterministic single-trial mode (sanity-check) ───────────────
    if args.once:
        # max_dll was retired from required args (May 2026) — daily-loss-stop
        # is now a fixed strategy safety rail, not a Discovery dimension.
        # The flag still parses (for back-compat) but is silently ignored.
        if None in (args.ema_long, args.stop_loss, args.rrr_reward, args.macro, args.micro):
            print("--once requires --ema-long, --stop-loss, --rrr-reward, --macro, --micro", file=sys.stderr)
            return 2
        params = {
            "ema_long":              args.ema_long,
            "stop_loss_pips":        args.stop_loss,
            "rrr_risk":              1,
            "rrr_reward":            args.rrr_reward,
            "use_ema_filter":        args.ema_filter == "on",
            "allowed_macro_regimes": [s.strip() for s in args.macro.split(",") if s.strip()],
            "allowed_micro_regimes": [s.strip() for s in args.micro.split(",") if s.strip()],
            "blocked_hours":         settings["blocked_hours"],
            # Self-describing fixed constants (May 2026) — same fields
            # sample_params writes so the --once trial record matches the
            # shape of full-search runs.
            "instrument":            settings["instrument"],
            "interval":              settings["interval"],
            "direction":             settings["direction"],
            "max_daily_losses":      FIXED_MAX_DAILY_LOSSES,
        }
        print(f"[discovery] --once trial: range {args.start} → {args.end}")
        print(f"[discovery] settings: {json.dumps(settings)}")
        print(f"[discovery] thresholds: {json.dumps(thresholds)}")
        print(f"[discovery] params: {json.dumps(params, indent=2)}")
        rec = run_trial(1, params, args.start, args.end, settings=settings, thresholds=thresholds)
        print(f"[discovery] result: pass={rec['pass']} reasons={rec['fail_reasons']} error={rec['error']}")
        print(f"[discovery] metrics: {json.dumps(rec['metrics'], indent=2)}")
        return 0 if rec["pass"] else 1

    # ── Normal random-search mode ────────────────────────────────────────────
    seed = args.seed if args.seed is not None else int(time.time())
    rng  = random.Random(seed)
    run_id = "discovery_" + datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    # Persist settings + thresholds into the run's config so the stored
    # results record reflects what was actually used (and downstream
    # consumers — trial detail page, etc. — can read them).
    config = {
        "trials": args.trials, "start": args.start, "end": args.end,
        "seed":   seed,
        "instrument":     settings["instrument"],
        "interval":       settings["interval"],
        "direction":      settings["direction"],
        "blocked_hours":  settings["blocked_hours"],
        # May 2026: persist the per-run EMA-filter choice (promoted from a
        # searched dimension to a fixed constant) and the RRR Reward upper
        # bound (lower bound stays fixed at 1) so the run header can
        # display them and a future re-run can reproduce them exactly.
        "use_ema_filter": settings["use_ema_filter"],
        "rrr_reward_max": settings["rrr_reward_max"],
        # Regime Filters (May 2026). Persisted into the run record so
        # completed-run blocks can show which regimes were in / out of the
        # search space for this run. Stored as ordered lists for stable
        # JSON round-tripping; resolve_settings re-orders to canonical on
        # the way back in.
        "allowed_macro_regimes": settings["allowed_macro_regimes"],
        "allowed_micro_regimes": settings["allowed_micro_regimes"],
        "min_pf":           thresholds["min_pf"],
        "min_trades":       thresholds["min_trades"],
        "max_dd_pct":       thresholds["max_dd_pct"],
        "max_daily_dd_pct": thresholds["max_daily_dd_pct"],
    }
    results_path = Path(args.results_file)
    payload = init_results_file(results_path, run_id, config)
    print(f"[discovery] run_id={run_id} trials={args.trials} range={args.start}→{args.end} seed={seed}")
    print(f"[discovery] settings: {json.dumps(settings)}")
    print(f"[discovery] thresholds: {json.dumps(thresholds)}")

    try:
        for i in range(1, args.trials + 1):
            params = sample_params(rng, settings=settings)
            print(f"[discovery] trial {i}/{args.trials} …", flush=True)
            rec = run_trial(i, params, args.start, args.end, settings=settings, thresholds=thresholds)
            verdict = "PASS" if rec["pass"] else ("FAIL: " + ", ".join(rec["fail_reasons"]))
            pf = rec["metrics"].get("profit_factor")
            pf_s = "inf" if pf is None else f"{pf:.2f}"
            print(f"[discovery]   {verdict}  pf={pf_s}  trades={rec['metrics']['total_trades']}  "
                  f"dd={rec['metrics']['max_drawdown']:.1f}%  ({rec['duration_sec']}s)", flush=True)
            append_trial(results_path, payload, rec)
        finalize(results_path, payload, "complete")
    except KeyboardInterrupt:
        finalize(results_path, payload, "cancelled", error="interrupted by user")
        return 130
    except Exception as e:
        finalize(results_path, payload, "error", error=str(e))
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
