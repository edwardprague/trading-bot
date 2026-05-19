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
MAX_DAILY_LOSS_RANGE = (1, 5)

ALL_MACRO_REGIMES = ["strong_down", "staircase_down", "flat", "staircase_up", "strong_up"]
ALL_MICRO_REGIMES = [
    "trending_fast_down", "trending_medium_down", "trending_slow_down",
    "trending_fast_up",   "trending_medium_up",   "trending_slow_up",
    "ranging_narrow", "ranging_medium", "ranging_wide", "transitioning",
]

# ── Fixed constants (Phase 1) ─────────────────────────────────────────────────
FIXED_INSTRUMENT     = "GBPUSD"
FIXED_INTERVAL       = "5m"
FIXED_DIRECTION      = "short_only"
FIXED_EMA_SHORT      = 8
FIXED_EMA_MID        = 20
FIXED_BLOCKED_HOURS  = "4,5,6,8,10,11,14,17"  # v3's current values
FIXED_APPLY_SLIPPAGE = "true"
FIXED_SPREAD_PIPS    = "1.0"
FIXED_SL_SLIPPAGE    = "1.0"
FIXED_STRATEGY_VER   = "v2"

# ── Objective function (Phase 1) ──────────────────────────────────────────────
OBJ_PROFIT_FACTOR_MIN = 1.5
OBJ_TRADES_MIN        = 50
OBJ_MAX_DRAWDOWN_MAX  = 15.0   # percent

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_TRIALS = 200
DEFAULT_START  = "2025-07-01"
DEFAULT_END    = "2025-12-31"


# ── Sampling ──────────────────────────────────────────────────────────────────

def _sample_subset(labels, rng):
    """Bernoulli p=0.5 per label; resample if the subset comes back empty.
    Empty allow-lists would block 100% of trades, which is degenerate, so we
    guarantee at least one label."""
    while True:
        picked = [lbl for lbl in labels if rng.random() < 0.5]
        if picked:
            return picked


def sample_params(rng):
    """Draw a single parameter set uniformly from the Phase 1 search space."""
    return {
        "ema_long":              rng.randint(*EMA_LONG_RANGE),
        "stop_loss_pips":        rng.randint(*STOP_LOSS_RANGE),
        "rrr_risk":              1,
        "rrr_reward":            rng.randint(*RRR_REWARD_RANGE),
        "max_daily_losses":      rng.randint(*MAX_DAILY_LOSS_RANGE),
        "use_ema_filter":        rng.random() < 0.5,
        "allowed_macro_regimes": _sample_subset(ALL_MACRO_REGIMES, rng),
        "allowed_micro_regimes": _sample_subset(ALL_MICRO_REGIMES, rng),
        "blocked_hours":         FIXED_BLOCKED_HOURS,
    }


# ── Trial execution ───────────────────────────────────────────────────────────

def build_env(params, start_date, end_date):
    """Build the env dict for one strategy_v2 subprocess call."""
    env = os.environ.copy()
    env.update({
        "DISCOVERY_MODE":        "1",
        "STRATEGY_VERSION":      FIXED_STRATEGY_VER,
        "RUN_MODE":              "date_range",
        "RUN_START_DATE":        start_date,
        "RUN_END_DATE":          end_date,
        "INSTRUMENT":            FIXED_INSTRUMENT,
        "INTERVAL":              FIXED_INTERVAL,
        "TRADE_DIRECTION":       FIXED_DIRECTION,
        "EMA_SHORT":             str(FIXED_EMA_SHORT),
        "EMA_MID":               str(FIXED_EMA_MID),
        "EMA_LONG":              str(params["ema_long"]),
        "FRACTAL_STOP_PIPS":     str(params["stop_loss_pips"]),
        "RRR_RISK":              str(params["rrr_risk"]),
        "RRR_REWARD":            str(params["rrr_reward"]),
        "MAX_DAILY_LOSSES":      str(params["max_daily_losses"]),
        "USE_EMA_FILTER":        "true" if params["use_ema_filter"] else "false",
        "BLOCKED_HOURS_UTC":     params["blocked_hours"],
        "ALLOWED_MACRO_REGIMES": ",".join(params["allowed_macro_regimes"]),
        "ALLOWED_MICRO_REGIMES": ",".join(params["allowed_micro_regimes"]),
        "APPLY_SLIPPAGE":        FIXED_APPLY_SLIPPAGE,
        "SPREAD_PIPS":           FIXED_SPREAD_PIPS,
        "SL_SLIPPAGE_PIPS":      FIXED_SL_SLIPPAGE,
    })
    return env


def evaluate_objective(metrics):
    """Return (pass, reasons[]). PF=None (∞) treated as passing the PF bar.

    Max drawdown is compared as an absolute magnitude: strategy_v2's
    compute_metrics writes drawdown with a negative-sign convention, but
    the Phase 1 objective "max DD ≤ 15%" reads naturally as a positive cap
    — without the abs() here, a catastrophic -50% drawdown would silently
    pass because -50 < 15."""
    reasons = []
    pf       = metrics.get("profit_factor")
    trades   = metrics.get("total_trades", 0) or 0
    max_dd   = abs(metrics.get("max_drawdown", 0.0) or 0.0)

    if pf is not None and pf < OBJ_PROFIT_FACTOR_MIN:
        reasons.append(f"profit_factor {pf:.2f} < {OBJ_PROFIT_FACTOR_MIN}")
    if trades < OBJ_TRADES_MIN:
        reasons.append(f"trades {trades} < {OBJ_TRADES_MIN}")
    if max_dd > OBJ_MAX_DRAWDOWN_MAX:
        reasons.append(f"max_drawdown {max_dd:.1f}% > {OBJ_MAX_DRAWDOWN_MAX}%")
    return (len(reasons) == 0, reasons)


def run_trial(trial_num, params, start_date, end_date):
    """Run a single strategy_v2 subprocess; return a trial-result dict."""
    DISCOVERY_TMP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DISCOVERY_TMP_DIR / f"trial_{uuid.uuid4().hex}.json"
    env = build_env(params, start_date, end_date)
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
        passed, reasons = evaluate_objective(metrics)

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

    if args.config_json:
        with open(args.config_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
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

    # ── --once: deterministic single-trial mode (sanity-check) ───────────────
    if args.once:
        if None in (args.ema_long, args.stop_loss, args.rrr_reward, args.max_dll, args.macro, args.micro):
            print("--once requires --ema-long, --stop-loss, --rrr-reward, --max-dll, --macro, --micro", file=sys.stderr)
            return 2
        params = {
            "ema_long":              args.ema_long,
            "stop_loss_pips":        args.stop_loss,
            "rrr_risk":              1,
            "rrr_reward":            args.rrr_reward,
            "max_daily_losses":      args.max_dll,
            "use_ema_filter":        args.ema_filter == "on",
            "allowed_macro_regimes": [s.strip() for s in args.macro.split(",") if s.strip()],
            "allowed_micro_regimes": [s.strip() for s in args.micro.split(",") if s.strip()],
            "blocked_hours":         FIXED_BLOCKED_HOURS,
        }
        print(f"[discovery] --once trial: range {args.start} → {args.end}")
        print(f"[discovery] params: {json.dumps(params, indent=2)}")
        rec = run_trial(1, params, args.start, args.end)
        print(f"[discovery] result: pass={rec['pass']} reasons={rec['fail_reasons']} error={rec['error']}")
        print(f"[discovery] metrics: {json.dumps(rec['metrics'], indent=2)}")
        return 0 if rec["pass"] else 1

    # ── Normal random-search mode ────────────────────────────────────────────
    seed = args.seed if args.seed is not None else int(time.time())
    rng  = random.Random(seed)
    run_id = "discovery_" + datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    config = {
        "trials": args.trials, "start": args.start, "end": args.end,
        "seed":   seed,
    }
    results_path = Path(args.results_file)
    payload = init_results_file(results_path, run_id, config)
    print(f"[discovery] run_id={run_id} trials={args.trials} range={args.start}→{args.end} seed={seed}")

    try:
        for i in range(1, args.trials + 1):
            params = sample_params(rng)
            print(f"[discovery] trial {i}/{args.trials} …", flush=True)
            rec = run_trial(i, params, args.start, args.end)
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
