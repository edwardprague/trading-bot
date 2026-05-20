"""
server.py — Trading Bot Dashboard
==================================
Serves report.html at http://localhost:8080 and injects a run bar with:
  • "Run New Version" — increments version, 730-day full run
  • "Run Date Range"  — runs current version with selected date range
  • Start/end date pickers that persist via localStorage

Usage:
    source venv/bin/activate
    python3 server.py
    # Then open http://localhost:8080

Note: port 8080 is used instead of 5000 because macOS Monterey and later
reserves port 5000 for AirPlay Receiver, which intercepts requests before
they reach Flask.
"""

import os
import sys
import json
import re
import glob
import shutil
import subprocess
import threading
from pathlib import Path
from cbot_templates import generate_cbot

# ── Auto-install Flask if missing ─────────────────────────────────────────────
try:
    from flask import Flask, Response, jsonify, request, send_from_directory, abort
except ImportError:
    print("  Flask not found — installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"],
                          stdout=subprocess.DEVNULL)
    from flask import Flask, Response, jsonify, request, send_from_directory, abort

from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).parent
REPORT_FILE   = BASE_DIR / "report.html"
STRATEGY_FILE = BASE_DIR / "strategy.py"
RESULTS_DIR   = BASE_DIR / "results"
DATA_DIR      = BASE_DIR / "data"

# Versions store — server-side source of truth for the user's strategy
# profiles. Each version bundles:
#   - id, name (display label)
#   - strategy_version (which strategy_vN.py module to invoke)
#   - regime_state (per-version macro/micro allow-lists)
# The active_version_id field drives global context: /run, /run_range,
# /run_batch use the active version's params unless the request payload
# explicitly overrides them. /run_regime_analysis writes its toggle state
# back into the active version so switching versions never cross-pollutes
# regime configuration.
VERSIONS_FILE             = DATA_DIR / "versions.json"
LEGACY_REGIME_STATE_FILE  = DATA_DIR / "regime_filter_state.json"

# Canonical regime-key lists — mirrors MACRO_REGIME_ORDER + REGIME_ORDER in
# regime_analysis.py. Used as the "all active" default when a new version is
# created without a base to copy from: every key listed = every regime allowed,
# which is functionally equivalent to "no gate" but renders all RA toggles as
# ON (clearer UX than an empty allow-list).
_ALL_MACRO_KEYS = ["strong_down", "staircase_down", "flat", "staircase_up", "strong_up"]
_ALL_MICRO_KEYS = [
    "trending_fast_down", "trending_medium_down", "trending_slow_down",
    "trending_fast_up",   "trending_medium_up",   "trending_slow_up",
    "ranging_narrow", "ranging_medium", "ranging_wide", "transitioning",
]

_DEFAULT_VERSIONS = {
    "active_version_id": "v2",
    "versions": [
        {
            "id": "v1",
            "name": "v1",
            "strategy_version": "v1",
            "regime_state": {
                "allowed_macro_regimes": [],
                "allowed_micro_regimes": [],
            },
        },
        {
            "id": "v2",
            "name": "v2",
            "strategy_version": "v2",
            "regime_state": {
                "allowed_macro_regimes": ["staircase_down", "strong_down"],
                "allowed_micro_regimes": ["ranging_medium", "ranging_wide"],
            },
        },
    ],
}

# One-time rename of seeded entries from the earlier descriptive names.
# Applied on every _read_versions() call until the rename takes effect,
# then persisted — afterwards the dict lookups simply miss and nothing
# happens. Cheap correctness for users already on the previous schema.
_VERSION_NAME_MIGRATIONS = {
    "v1 — Fractal Only":     "v1",
    "v2 — EMA + Regime Gates": "v2",
}

app = Flask(__name__)


# ── Versions storage — server-side source of truth ────────────────────────────

def _migrate_legacy_regime_state(data):
    """One-time migration: if the old data/regime_filter_state.json exists,
    fold its values into v2's regime_state. Caller passes a freshly-seeded
    `data` dict that will be persisted afterwards."""
    try:
        if not LEGACY_REGIME_STATE_FILE.exists():
            return
        with open(LEGACY_REGIME_STATE_FILE, "r", encoding="utf-8") as f:
            legacy = json.load(f)
        if not isinstance(legacy, dict):
            return
        for v in data.get("versions", []):
            if v.get("id") != "v2":
                continue
            if "allowed_macro_regimes" in legacy:
                v["regime_state"]["allowed_macro_regimes"] = list(legacy["allowed_macro_regimes"])
            if "allowed_micro_regimes" in legacy:
                v["regime_state"]["allowed_micro_regimes"] = list(legacy["allowed_micro_regimes"])
            break
    except (OSError, ValueError) as e:
        print(f"  [versions] legacy migration skipped: {e}", file=sys.stderr)


def _read_versions():
    """Load data/versions.json. If absent or malformed, seed with the two
    default versions (v1, v2), migrate any legacy regime state into v2,
    persist, and return the fresh dict. Also rewrites any seeded entries
    that still carry the old descriptive names (e.g. 'v1 — Fractal Only')
    so they use the current simple 'vN' naming scheme."""
    if VERSIONS_FILE.exists():
        try:
            with open(VERSIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("versions"), list) and data["versions"]:
                # Best-effort one-time rename of seeded entries to the simple
                # naming scheme. Idempotent: subsequent loads no-op.
                renamed = False
                for v in data["versions"]:
                    new_name = _VERSION_NAME_MIGRATIONS.get(v.get("name"))
                    if new_name and v.get("name") != new_name:
                        v["name"] = new_name
                        renamed = True
                if renamed:
                    _write_versions(data)
                return data
        except (OSError, ValueError) as e:
            print(f"  [versions] read failed, reseeding: {e}", file=sys.stderr)
    import copy as _copy
    data = _copy.deepcopy(_DEFAULT_VERSIONS)
    _migrate_legacy_regime_state(data)
    _write_versions(data)
    return data


def _next_version_name(versions):
    """Return the next sequential 'vN' name. May 2026: with
    renumber-on-delete in place, the existing versions are always a
    contiguous v1..vN sequence — so the next name is just v(len+1).

    Falls back to scanning for the max numeric suffix when the list
    contains any non-conforming names (legacy / hand-edited entries) so
    we never produce a duplicate id."""
    nums = []
    saw_non_vn = False
    for v in versions or []:
        m = re.match(r"^v(\d+)$", (v.get("name") or "").strip())
        if m:
            nums.append(int(m.group(1)))
        else:
            saw_non_vn = True
    if saw_non_vn:
        # Be conservative: fall back to legacy max+1 behaviour.
        return "v" + str((max(nums) if nums else 0) + 1)
    return "v" + str(len(versions or []) + 1)


def _write_versions(data):
    """Atomic write of versions.json. Best-effort — failures log to stderr."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = VERSIONS_FILE.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(VERSIONS_FILE)
    except OSError as e:
        print(f"  [versions] write failed: {e}", file=sys.stderr)


def _get_active_version():
    """Return the active version dict, or the first version if active_id
    is stale, or None if no versions exist."""
    data = _read_versions()
    active_id = data.get("active_version_id")
    for v in data.get("versions", []):
        if v.get("id") == active_id:
            return v
    versions = data.get("versions", [])
    return versions[0] if versions else None


def _set_active_version(version_id):
    """Switch the global active version. Returns True if version exists."""
    data = _read_versions()
    if not any(v.get("id") == version_id for v in data.get("versions", [])):
        return False
    data["active_version_id"] = version_id
    _write_versions(data)
    return True


def _make_version_id(name, existing_ids):
    """Slugify a display name into a unique id."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_") or "version"
    candidate = slug
    n = 1
    while candidate in existing_ids:
        n += 1
        candidate = f"{slug}_{n}"
    return candidate


def _add_version():
    """Append a new BLANK / UNASSIGNED version. Name auto-generated as
    'v<N+1>'. The slot starts with no strategy module, no params, and no
    regime state — Discovery fills all three in a single one-time write
    via _assign_version_from_trial. Until then the version is intentionally
    invisible in the BD + RA dropdowns (those filter on params != null),
    so users can't accidentally run a backtest against an empty profile.
    The Versions page is the only surface where unassigned versions show
    up, with the 'Awaiting parameter assignment from Discovery' placeholder.
    """
    data = _read_versions()
    versions = data.get("versions", [])
    existing_ids = {v.get("id") for v in versions}
    name = _next_version_name(versions)
    new_id = _make_version_id(name, existing_ids)

    new_version = {
        "id": new_id,
        "name": name,
        "strategy_version": None,
        "params": None,
        "regime_state": None,
    }
    versions.append(new_version)
    data["versions"] = versions
    _write_versions(data)
    return new_version


def _assign_version_from_trial(version_id, trial_params, run_config=None):
    """One-time write of a Discovery trial's params into an existing
    unassigned version slot. Refuses if the target already has params
    (no update endpoint — re-assignment requires a new version).

    `run_config` is the parent Discovery run's config dict — supplies
    instrument + interval (which live at the run level, not the trial
    level). Defaults to GBPUSD / 5m when missing.

    Maps the trial's flat params dict to the new params schema:
      ema_long, use_ema_filter, fractal_stop_pips (renamed from
      stop_loss_pips), rrr_reward, max_daily_losses, trade_direction,
      blocked_hours, instrument, interval, assigned_at.
    Also writes regime_state from the trial's allowed_macro/micro lists,
    and stamps strategy_version='v2' (Discovery's fixed base for now).

    Returns (True, version_dict) on success, (False, error_string) on
    failure.
    """
    data = _read_versions()
    target = None
    for v in data.get("versions", []):
        if v.get("id") == version_id:
            target = v
            break
    if target is None:
        return False, f"Version '{version_id}' not found"
    if target.get("params") is not None:
        return False, "Version is already assigned"

    tp = trial_params or {}
    rc = run_config   or {}
    target["strategy_version"] = "v2"
    target["params"] = {
        "ema_long":         tp.get("ema_long"),
        "use_ema_filter":   bool(tp.get("use_ema_filter", True)),
        "fractal_stop_pips": tp.get("stop_loss_pips"),
        "rrr_reward":       tp.get("rrr_reward"),
        "max_daily_losses": tp.get("max_daily_losses"),
        # Bug fix (May 2026): Phase 1 Discovery holds trade_direction and
        # blocked_hours fixed, but they're still active inputs in the
        # backtest. Without recording them here the BD's user-tunable
        # dropdown / blocked-hours checkboxes would silently override the
        # Discovery setting on the next run, producing materially different
        # results (the original Bug 2 symptom). Discovery's sampled params
        # carry blocked_hours; trade_direction is hardcoded to "short_only"
        # in build_env, so we record it explicitly here.
        "trade_direction":  tp.get("trade_direction") or "short_only",
        "blocked_hours":    tp.get("blocked_hours") or "4,5,6,8,10,11,14,17",
        # Structural change (May 2026): instrument + interval are per-version.
        # Pulled from the parent Discovery run's config (defaults if missing).
        "instrument":       (rc.get("instrument") or "GBPUSD").upper(),
        "interval":          rc.get("interval")   or "5m",
        "assigned_at":      datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    target["regime_state"] = {
        "allowed_macro_regimes": list(tp.get("allowed_macro_regimes") or []),
        "allowed_micro_regimes": list(tp.get("allowed_micro_regimes") or []),
        "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    _write_versions(data)
    return True, target


def _count_runs_per_version_name():
    """Return {version_name: run_count} by parsing report.html's
    versions-data <script> block. Used by /api/versions GET to stamp a
    run_count onto each version so the Versions page can show the
    'all runs will be deleted' warning before issuing the delete.
    Failures (no report.html, parse errors) return {} — the page degrades
    to a plain confirm in that case."""
    if not REPORT_FILE.exists():
        return {}
    try:
        html = REPORT_FILE.read_text(encoding="utf-8")
        m = re.search(
            r'<script[^>]+id=["\']versions-data["\'][^>]*>([\s\S]*?)</script>',
            html,
        )
        if not m:
            return {}
        buckets = json.loads(m.group(1).strip())
        return {
            (v.get("name") or ""): len(v.get("runs") or [])
            for v in buckets
        }
    except (OSError, ValueError):
        return {}


# ── Version rename / renumber ────────────────────────────────────────────
# Atomic helpers (May 2026) that rewrite a version's id+name across BOTH
# versions.json AND report.html's embedded versions-data buckets. These
# back two flows: a one-off Task-1 rename (v8 → v1) and the recurring
# auto-renumber-on-delete that produces a contiguous v1..vN sequence after
# each removal.

def _delete_bucket_from_report(name):
    """Remove a bucket from report.html's <script id="versions-data"> array
    and clean up its sidecar files. Factored out of /delete_version POST so
    _delete_version can call it directly (May 2026)."""
    if not name or not REPORT_FILE.exists():
        return
    try:
        html = REPORT_FILE.read_text(encoding="utf-8")
    except OSError:
        return
    match = re.search(
        r'(<script[^>]+id=["\']versions-data["\'][^>]*>)([\s\S]*?)(</script>)',
        html,
    )
    if not match:
        return
    try:
        buckets = json.loads(match.group(2).strip())
    except (json.JSONDecodeError, ValueError):
        return
    new_buckets = [b for b in buckets if b.get("name") != name]
    if len(new_buckets) == len(buckets):
        return  # nothing to drop
    new_json = json.dumps(new_buckets, indent=2, ensure_ascii=False)
    new_html = html[:match.start(2)] + "\n" + new_json + "\n" + html[match.end(2):]
    try:
        REPORT_FILE.write_text(new_html, encoding="utf-8")
    except OSError:
        pass
    # RESULTS_LOG and results/<name>_* sidecars — best-effort
    results_log = BASE_DIR / "RESULTS_LOG.md"
    if results_log.exists():
        try:
            lines = results_log.read_text(encoding="utf-8").splitlines(keepends=True)
            new_lines = [l for l in lines if not re.match(r'^\|\s*' + re.escape(name) + r'\s*\|', l)]
            results_log.write_text("".join(new_lines), encoding="utf-8")
        except OSError:
            pass
    results_dir = BASE_DIR / "results"
    if results_dir.is_dir():
        for f in results_dir.iterdir():
            if f.name.startswith(name + "_") or f.name.startswith(name + "."):
                try:
                    f.unlink()
                except OSError:
                    pass


def _rename_buckets_in_report(rename_map):
    """Rewrite bucket names inside report.html's versions-data block based
    on `rename_map` (old_name → new_name). Uses a two-pass temp-id swap so
    intermediate collisions (e.g. rename {v2:v1, v3:v2}) don't merge two
    distinct buckets. Buckets whose names aren't in the map are left
    untouched; orphan buckets stay orphan."""
    if not rename_map or not REPORT_FILE.exists():
        return
    try:
        html = REPORT_FILE.read_text(encoding="utf-8")
    except OSError:
        return
    match = re.search(
        r'(<script[^>]+id=["\']versions-data["\'][^>]*>)([\s\S]*?)(</script>)',
        html,
    )
    if not match:
        return
    try:
        buckets = json.loads(match.group(2).strip())
    except (json.JSONDecodeError, ValueError):
        return
    # Pass 1: map every renamable bucket to a temp namespace so we never
    # have two buckets with the same name at any intermediate step.
    TMP_PREFIX = "__rename_tmp__"
    for b in buckets:
        nm = b.get("name")
        if nm in rename_map:
            b["name"] = TMP_PREFIX + rename_map[nm]
    # Pass 2: strip the temp prefix.
    for b in buckets:
        nm = b.get("name") or ""
        if nm.startswith(TMP_PREFIX):
            b["name"] = nm[len(TMP_PREFIX):]
    new_json = json.dumps(buckets, indent=2, ensure_ascii=False)
    new_html = html[:match.start(2)] + "\n" + new_json + "\n" + html[match.end(2):]
    try:
        REPORT_FILE.write_text(new_html, encoding="utf-8")
    except OSError:
        pass


def _apply_version_rename(rename_map):
    """Atomically rewrite versions.json AND report.html for `rename_map`
    (old_id → new_id). Updates id, name, and active_version_id where
    applicable. No-op for empty maps. Returns the rename map (for caller
    convenience — used by DELETE response so the client can rewrite
    localStorage discovery_trial_assignments)."""
    if not rename_map:
        return {}
    data = _read_versions()
    versions = data.get("versions", [])
    for v in versions:
        old = v.get("id")
        if old in rename_map:
            new_id = rename_map[old]
            v["id"] = new_id
            # name has always tracked id (v1, v7, …); keep them in sync so
            # the BD sidebar bucket lookup (by name) keeps working.
            if (v.get("name") or "").strip() == old or not v.get("name"):
                v["name"] = new_id
    active = data.get("active_version_id")
    if active in rename_map:
        data["active_version_id"] = rename_map[active]
    _write_versions(data)
    _rename_buckets_in_report(rename_map)
    return rename_map


def _prune_orphan_buckets():
    """Drop any report.html versions-data buckets whose name doesn't
    correspond to an ASSIGNED version in versions.json. An unassigned
    version can't have backtest runs (no params → no strategy subprocess
    → nothing to save), so any bucket whose name matches an unassigned
    slot — or doesn't match any version at all — is orphan data from a
    previously-deleted version that happened to share the name. Returns
    a list of (name, run_count) tuples describing what was dropped, for
    logging.

    Sidecar cleanup (RESULTS_LOG row, results/<name>_* files) is delegated
    to _delete_bucket_from_report for each dropped name to keep the same
    "delete a version" semantics as the existing /delete_version flow."""
    if not REPORT_FILE.exists():
        return []
    try:
        html = REPORT_FILE.read_text(encoding="utf-8")
    except OSError:
        return []
    match = re.search(
        r'(<script[^>]+id=["\']versions-data["\'][^>]*>)([\s\S]*?)(</script>)',
        html,
    )
    if not match:
        return []
    try:
        buckets = json.loads(match.group(2).strip())
    except (json.JSONDecodeError, ValueError):
        return []
    data = _read_versions()
    assigned_names = {
        (v.get("name") or "").strip()
        for v in data.get("versions", [])
        if v.get("params")
    }
    assigned_names.discard("")
    dropped = []
    kept = []
    for b in buckets:
        nm = (b.get("name") or "").strip()
        if nm in assigned_names:
            kept.append(b)
        else:
            dropped.append((nm, len(b.get("runs") or [])))
    if not dropped:
        return []
    # Use _delete_bucket_from_report so sidecars also get swept. It rereads
    # report.html each call which is wasteful for many drops, but cleaner
    # than duplicating the regex + write logic here.
    for nm, _count in dropped:
        _delete_bucket_from_report(nm)
    return dropped


def _build_renumber_map(versions):
    """Given a versions list in creation order (the natural order in the
    versions.json array), return {old_id: new_id} that renumbers every
    version to v1..vN sequentially. Only entries whose id actually changes
    appear in the returned map."""
    rename = {}
    for i, v in enumerate(versions):
        new_id = "v" + str(i + 1)
        old_id = v.get("id")
        if old_id and old_id != new_id:
            rename[old_id] = new_id
    return rename


def _delete_version(version_id):
    """Remove a version, drop its report.html bucket, then auto-renumber
    the remaining versions to a contiguous v1..vN sequence. Refuses if
    it's the last remaining one. If the deleted version was active, falls
    back to the first remaining (which becomes v1 after renumber).

    Returns (ok, error_or_rename_map). When ok=True the second slot is
    the rename map applied during the renumber step — the caller hands
    this to the client so it can rewrite its localStorage
    discovery_trial_assignments without a page reload."""
    data = _read_versions()
    versions = data.get("versions", [])
    if len(versions) <= 1:
        return False, "Cannot delete the last remaining version"
    target = next((v for v in versions if v.get("id") == version_id), None)
    if target is None:
        return False, "Version not found"
    target_name = target.get("name") or target.get("id") or ""
    new_versions = [v for v in versions if v.get("id") != version_id]
    data["versions"] = new_versions
    if data.get("active_version_id") == version_id:
        data["active_version_id"] = new_versions[0]["id"]
    _write_versions(data)
    # Clean up the deleted version's run bucket BEFORE renumber so the
    # rename pass doesn't try to renumber a bucket that's about to be
    # removed anyway.
    _delete_bucket_from_report(target_name)
    # Auto-renumber the survivors to v1..vN.
    rename_map = _build_renumber_map(new_versions)
    _apply_version_rename(rename_map)
    # Sweep any orphan buckets that survived the rename — buckets whose
    # name now matches an unassigned version (or no version at all). These
    # accumulate across the codebase's history (versions deleted before
    # auto-cleanup existed, or names that happened to clash with a
    # renumbered id). Pruning here keeps the report.html data consistent
    # with versions.json after every delete.
    _prune_orphan_buckets()
    return True, rename_map


def _write_active_regime_state(allowed_macro, allowed_micro):
    """Save the regime-toggle state to the currently-active version. Used
    by /run_regime_analysis on every successful run so the RA page's
    toggles persist per-version."""
    data = _read_versions()
    active_id = data.get("active_version_id")
    found = False
    for v in data.get("versions", []):
        if v.get("id") == active_id:
            v["regime_state"] = {
                "allowed_macro_regimes": list(allowed_macro or []),
                "allowed_micro_regimes": list(allowed_micro or []),
                "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
            found = True
            break
    if found:
        _write_versions(data)


_KNOWN_STRATEGY_MODULES = {"v1", "v2"}


def _resolve_version_for_run(payload):
    """Pick which version's params/regime drive this backtest run.

    Bugfix (May 2026): the dropdown selection is now decoupled from the
    global active version. The BD dropdown is a 'what am I looking at /
    testing right now' selector; the global active version is a separate
    concept ('what's promoted as current production'). The two can differ:
    running a test on v4 while v3 stays globally active.

    Resolution order:
      1. payload `version_id` (explicit override from the BD run-bar).
      2. payload `strategy_version` if it's a version id (legacy path —
         the BD's run payload still sends the dropdown value under this
         key for backward compat).
      3. The globally-active version (fallback for direct API callers
         that don't pass a version_id).

    Returns the resolved version dict, or None if nothing matches."""
    if not isinstance(payload, dict):
        return _get_active_version()

    candidate = (payload.get("version_id") or "").strip()
    if not candidate:
        sv = (payload.get("strategy_version") or "").strip()
        if sv and sv not in _KNOWN_STRATEGY_MODULES:
            candidate = sv

    if candidate:
        for v in _read_versions().get("versions", []):
            if v.get("id") == candidate:
                return v
    return _get_active_version()


def _apply_active_version_to_env(env_overrides, payload):
    """Layer the SELECTED version's strategy_version + regime allow-lists
    + params into env_overrides. Selection is whatever
    _resolve_version_for_run picks (payload.version_id wins; otherwise
    falls back to global active). Precedence within the resolved version:

      1. Explicit values in the request payload win (with one exception
         below for STRATEGY_VERSION).
      2. Resolved version supplies STRATEGY_VERSION + ALLOWED_*_REGIMES
         + params (ema_long, fractal_stop_pips, etc.).
      3. Strategy module hardcoded defaults are the final fallback.

    STRATEGY_VERSION quirk: the BD dropdown sends the option value as
    `strategy_version` in the run payload. For unassigned slots this
    would be a version id — but unassigned versions don't appear in the
    dropdown, so this case is filtered upstream. Assigned versions
    resolve through versions.json to the underlying 'v1' / 'v2' module.

    The strategy module distinguishes 'unset' from 'empty string': empty
    means 'gate disabled', unset means 'use default'. We always set
    ALLOWED_*_REGIMES when a version exists (empty list → empty string),
    preserving that distinction."""
    av = _resolve_version_for_run(payload)
    if av is None:
        return

    # ── STRATEGY_VERSION resolution ─────────────────────────────────────
    current_sv = (env_overrides.get("STRATEGY_VERSION") or "").strip()
    if current_sv and current_sv not in _KNOWN_STRATEGY_MODULES:
        # Treat the payload value as a version id and resolve to the
        # underlying strategy module via versions.json.
        env_overrides.pop("STRATEGY_VERSION", None)
        for v in _read_versions().get("versions", []):
            if v.get("id") == current_sv:
                resolved = (v.get("strategy_version") or "").strip()
                if resolved in _KNOWN_STRATEGY_MODULES:
                    env_overrides["STRATEGY_VERSION"] = resolved
                break
    if "STRATEGY_VERSION" not in env_overrides:
        av_sv = (av.get("strategy_version") or "").strip()
        if av_sv in _KNOWN_STRATEGY_MODULES:
            env_overrides["STRATEGY_VERSION"] = av_sv

    # ── Regime allow-lists ──────────────────────────────────────────────
    payload_macro = payload.get("allowed_macro_regimes") if isinstance(payload, dict) else None
    payload_micro = payload.get("allowed_micro_regimes") if isinstance(payload, dict) else None
    rs = av.get("regime_state") or {}

    if payload_macro is not None:
        env_overrides["ALLOWED_MACRO_REGIMES"] = ",".join(payload_macro)
    elif "ALLOWED_MACRO_REGIMES" not in env_overrides:
        env_overrides["ALLOWED_MACRO_REGIMES"] = ",".join(rs.get("allowed_macro_regimes", []) or [])

    if payload_micro is not None:
        env_overrides["ALLOWED_MICRO_REGIMES"] = ",".join(payload_micro)
    elif "ALLOWED_MICRO_REGIMES" not in env_overrides:
        env_overrides["ALLOWED_MICRO_REGIMES"] = ",".join(rs.get("allowed_micro_regimes", []) or [])

    # ── params fallback ─────────────────────────────────────────────────────
    # When the active version was assigned by Discovery it carries a slim
    # `params` snapshot (the new schema; superseded the old `backtest_params`
    # field name). The BD always sends its input values in the payload, so
    # for BD-originated runs the payload wins and this fallback never fires.
    # But direct API callers and tooling that POST a minimal payload still
    # get the version's parameters applied.
    #
    # Schema is intentionally slim: rrr_risk is implicitly 1 (Discovery
    # holds it fixed), and blocked_hours is NOT stored on the version —
    # the user's BD-level BLOCKED_HOURS_UTC default carries through.
    bp = av.get("params") or {}
    if bp:
        _BP_TO_ENV = {
            "ema_long":          "EMA_LONG",
            "fractal_stop_pips": "FRACTAL_STOP_PIPS",
            "rrr_reward":        "RRR_REWARD",
            "max_daily_losses":  "MAX_DAILY_LOSSES",
            # Bug fix (May 2026): trade_direction and blocked_hours are
            # now stored on the version too. Without these in the layering
            # the BD payload's user-tunable values silently override the
            # Discovery-recorded settings (this was the residual symptom
            # after the use_ema_filter fix — direction stayed "both" so
            # backtest counts diverged from Discovery's short-only profile).
            "trade_direction":   "TRADE_DIRECTION",
            "blocked_hours":     "BLOCKED_HOURS_UTC",
            # Structural change (May 2026): instrument + interval are now
            # per-version too. The BD toolbar instrument dropdown is gone;
            # the BD payload omits these so this fallback supplies them
            # from the active version's params (defaults GBPUSD / 5m
            # applied below if the params block is silent).
            "instrument":        "INSTRUMENT",
            "interval":          "INTERVAL",
        }
        for bp_key, env_key in _BP_TO_ENV.items():
            if env_key not in env_overrides or env_overrides[env_key] == "":
                if bp_key in bp and bp[bp_key] is not None:
                    env_overrides[env_key] = str(bp[bp_key])
        # USE_EMA_FILTER is a bool in params; the strategy reads
        # the env as the string 'true' / 'false'.
        if "USE_EMA_FILTER" not in env_overrides and "use_ema_filter" in bp:
            env_overrides["USE_EMA_FILTER"] = "true" if bp["use_ema_filter"] else "false"

    # Defaults for the structural per-version fields (May 2026 restructure).
    # Apply even when av_params is empty so callers get a working backtest
    # without the version explicitly carrying instrument/interval. Matches
    # the documented per-spec defaults of GBPUSD / 5m.
    if not env_overrides.get("INSTRUMENT"):
        env_overrides["INSTRUMENT"] = "GBPUSD"
    if not env_overrides.get("INTERVAL"):
        env_overrides["INTERVAL"] = "5m"

# ── Backtest state (shared between the Flask thread and the worker thread) ─────
_bt_lock  = threading.Lock()
_bt_state = {"running": False, "ok": None, "error": None, "no_data": False, "stage": "", "progress": 0}

# ── Run-bar HTML (injected into every page response) ──────────────────────────

INJECT_HTML = """
<nav class="top-nav" id="top-nav">
  <ul class="top-nav-items">
    <li><a class="top-nav-link top-nav-link-active" href="/">Backtesting</a></li>
    <li><a class="top-nav-link" href="/results/regime_analysis.html">Regimes</a></li>
    <li><a class="top-nav-link" href="/discovery">Discovery</a></li>
    <li><a class="top-nav-link" href="/versions">Versions</a></li>
  </ul>
  <span class="top-nav-active-version" id="top-nav-active-version"></span>
</nav>

<div id="run-bar" style="
  position: fixed; top: 0; left: 0; right: 0; height: 52px;
  z-index: 9999; display: flex; align-items: center; gap: 12px;
  padding: 0 20px;
  background: #0c0c18; border-bottom: 1px solid #1e1e32;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
">
  <!-- Version + instrument dropdowns both removed (May 2026). The active
       version is now set exclusively on the /versions page via per-card
       radio buttons; the BD reflects it read-only via the top-nav
       "Active: vN" indicator. window.__activeVersionId is injected by the
       / route handler so the strategy-template JS knows which version's
       runs to show in the sidebar before /api/versions resolves. -->
  <!-- Instrument is now driven by the active version's params.instrument
       so a single version always runs against the same instrument.
       getSelectedInstrument() in this file falls back to "" when no
       dropdown is present, and _apply_active_version_to_env layers the
       instrument in from the version. -->

  <span class="rb-sep"></span>

  <button id="run-new-btn" class="rb-btn rb-btn-green" onclick="runNewVersion()">&#9654;&nbsp; Add Year</button>

  <span class="rb-sep"></span>

  <div id="rb-range-group" style="display: flex; align-items: center; gap: 12px;">
    <!-- Task 3: native date input. The previous design hid the input's
         text with `color: transparent` and overlaid a custom "Mon-DD-YY"
         span — that broke whenever the value was set programmatically
         (no input/change event → overlay stayed stale). The native
         input is reliable and accessible. -->
    <label class="rb-label" for="rb-start">From</label>
    <input type="date" id="rb-start" class="rb-date">
    <label class="rb-label" for="rb-end">To</label>
    <input type="date" id="rb-end" class="rb-date">
    <button id="run-range-btn" class="rb-btn rb-btn-blue" onclick="runDateRange()">&#9654;&nbsp; Add Date Range</button>
  </div>

  <span id="run-status" style="font-size: 13px; color: #666690; margin-left: 8px;"></span>

  <div id="rb-action-group" style="margin-left: auto; display: flex; align-items: center; gap: 12px;"></div>
</div>

<style>
  body { padding-top: 92px !important; }

  .rb-btn {
    color: #fff; border: none; border-radius: 6px;
    padding: 7px 16px; font-size: 12px; cursor: pointer;
    letter-spacing: 0.02em; flex-shrink: 0; transition: background 0.15s;
    white-space: nowrap;
  }
  .rb-btn-green { background: green; }
  .rb-btn-green:hover:not(:disabled) { background: #02bc02; }
  .rb-btn-blue { background: steelblue; }
  .rb-btn-blue:hover:not(:disabled) { background: #55a0dd; }
  .rb-btn:disabled { background: #1e1e38 !important; color: #404060; cursor: not-allowed; }

  .rb-sep {
    width: 1px; height: 28px; background: #1e1e32; flex-shrink: 0;
  }

  .rb-label {
    font-size: 11px; color: #505070; flex-shrink: 0;
  }

  .rb-date {
    background: #14142a; color: #c0c0e0; border: 1px solid #2a2a44;
    border-radius: 5px; padding: 5px 8px; font-size: 12px;
    font-family: inherit; width: 130px; flex-shrink: 0;
    color-scheme: dark;
  }
  .rb-date:focus { border-color: #4cc9f0; outline: none; }

  .rb-select {
    background: #14142a; color: #c0c0e0; border: 1px solid #2a2a44;
    border-radius: 5px; padding: 5px 8px; font-size: 12px;
    font-family: inherit; flex-shrink: 0; cursor: pointer;
    color-scheme: dark; outline: none;
  }
  .rb-select:focus { border-color: #4cc9f0; }

  .rb-devlog-btn {
    display: flex; align-items: center; justify-content: center;
    background: transparent; color: #888; border: none; border-radius: 5px;
    padding: 4px; cursor: pointer; transition: color 0.15s, background 0.15s;
    flex-shrink: 0;
  }
  .rb-devlog-btn .material-symbols-outlined { font-size: 22px; }
  .rb-devlog-btn:hover { color: #fff; background: rgba(255,255,255,0.08); }
  .rb-devlog-btn.active { color: #ffd700; background: rgba(255,215,0,0.1); }

  .rb-btn-copy { background: green; }
  .rb-btn-copy:hover:not(:disabled) { background: #02bc02; }
  .rb-btn-copy.copied { background: transparent !important; color: #6bcb77; border: 1px solid #6bcb77; }
  .rb-btn-cbot { background: teal; }
  .rb-btn-cbot:hover:not(:disabled) { background: #02b5b5; }
  .rb-btn-cbot.downloaded { background: transparent !important; color: #4cc9f0; border: 1px solid #4cc9f0; }
  /* .rb-btn-delete now lives in style.css so /versions, /discovery, RA,
     and trial-detail pages get the same crimson + hover + disabled
     styling. The inline duplicates that used to sit here have been
     removed (May 2026). */

  @keyframes rb-spin { to { transform: rotate(360deg); } }
  .rb-spin {
    display: inline-block; width: 13px; height: 13px; margin-right: 7px;
    border: 2px solid #303050; border-top-color: #4cc9f0;
    border-radius: 50%; animation: rb-spin 0.75s linear infinite;
    vertical-align: middle;
  }

  .rb-progress-wrap {
    display: inline-flex; align-items: center; gap: 8px; vertical-align: middle;
  }
  .rb-progress-bar {
    width: 120px; height: 8px; background: #1e1e38; border-radius: 4px;
    overflow: hidden; position: relative;
  }
  .rb-progress-fill {
    height: 100%; background: #4cc9f0; border-radius: 4px;
    transition: width 0.4s ease;
  }
  .rb-progress-text {
    font-size: 12px; color: #9090c0; white-space: nowrap;
  }
</style>

<script>
/* INJECT_HTML is now inserted at the START of <body> (see the / route in
   this file) so its DOM elements — version-select, instrument-select, the
   run-bar inputs — are parsed before report.html's inline scripts run.
   But this IIFE accesses elements that live LATER in the body (copy-btn,
   cbot-btn, devlog-btn, hidden in report.html and moved into the run-bar
   here), so we defer it until DOMContentLoaded — by which point the
   entire page is parsed and every element is available. Top-level
   functions below (runNewVersion, runDateRange, setRunning, etc.) stay
   at script-global scope so onclick="…" handlers can still call them. */
document.addEventListener("DOMContentLoaded", function () {
  /* ── Move action buttons into the run bar (preserve visibility set by strategy.py) ── */
  var _actGroup  = document.getElementById("rb-action-group");
  var _devlogBtn = document.getElementById("devlog-btn");
  var _actSep    = document.getElementById("rb-act-sep");
  var _copyBtn   = document.getElementById("copy-btn");
  var _cbotBtn   = document.getElementById("cbot-btn");
  if (_actGroup) {
    if (_copyBtn)   { _copyBtn.className = "rb-btn rb-btn-copy"; _actGroup.appendChild(_copyBtn); }
    if (_cbotBtn)   { _cbotBtn.className = "rb-btn rb-btn-cbot"; _actGroup.appendChild(_cbotBtn); }
    if (_actSep)    { _actSep.className = "rb-sep";  _actGroup.appendChild(_actSep); }
    if (_devlogBtn) { _devlogBtn.className = "rb-devlog-btn"; _devlogBtn.style.display = ""; _actGroup.appendChild(_devlogBtn); }
  }

  /* ── Persist date pickers via localStorage ─────────────────────── */
  /* Native date inputs now (Task 3) — no overlay sync needed. */
  var startEl = document.getElementById("rb-start");
  var endEl   = document.getElementById("rb-end");

  var savedStart = localStorage.getItem("rb_start_date");
  var savedEnd   = localStorage.getItem("rb_end_date");
  if (savedStart) startEl.value = savedStart;
  if (savedEnd)   endEl.value   = savedEnd;

  startEl.addEventListener("change", function () {
    localStorage.setItem("rb_start_date", startEl.value);
  });
  endEl.addEventListener("change", function () {
    localStorage.setItem("rb_end_date", endEl.value);
  });

  /* ── On load: resume polling if a backtest is already running ───── */
  fetch("/status")
    .then(function (r) { return r.json(); })
    .then(function (data) {
      if (data.running) { setRunning(); pollStatus(); }
    })
    .catch(function () {});

  /* ── Versions integration ───────────────────────────────────────────────
     Source of truth for "versions" is now /api/versions (data/versions.json
     server-side). We:
       1. Populate the top-nav active-version indicator.
       2. Make sure the BD's #version-select contains an option for every
          version in versions.json (the dropdown is otherwise driven by
          report.html's embedded versions-data; we additively top it up so
          newly-created profiles are selectable even before a backtest run).
       3. Sync the dropdown's selection to the active version on load.
       4. Intercept the dropdown's change event so picking a version also
          posts /api/active_version, switching the global context (drives
          /run, /run_range, /run_batch defaults + the RA toggle scope).
     ───────────────────────────────────────────────────────────────────── */
  function _setActiveIndicator(name) {
    var el = document.getElementById("top-nav-active-version");
    if (el) el.textContent = name ? "Active: " + name : "";
  }

  /* ── BD pre-populate from active version's params ──────────────────────
     Discovery-assigned versions carry a slim `params` snapshot — the new
     schema replaces the old `backtest_params` field. Stamp these onto
     the BD's settings-panel inputs in the DOM so the run-bar reflects
     what the next backtest will actually use.

     Fields pre-filled (per spec): EMA Long, fractal stop pips, RRR
     reward, max daily losses. rrr_risk is implicitly 1 (Discovery holds
     it fixed). blocked_hours is NOT in the version's params block — it
     stays under the user's own BD control.

     We deliberately do NOT write to localStorage — switching to an
     unassigned version (no params) restores from localStorage so the
     user's hand-tuned BD prefs survive a discovery detour.

     use_ema_filter has no visible BD checkbox today, but the strategy
     subprocess still picks it up via _apply_active_version_to_env. */
  function applyBacktestParamsFromVersion(version) {
    var p = version && version.params;
    function setVal(id, v) {
      var el = document.getElementById(id);
      if (el && v !== undefined && v !== null && v !== "") el.value = String(v);
    }
    function setCheckbox(id, v) {
      var el = document.getElementById(id);
      if (el && v !== undefined && v !== null) el.checked = !!v;
    }
    /* Bug fix (May 2026): blocked_hours is a CSV string like
       "4,5,6,8,10,11,14,17". Parse and apply to the 24 bs-bh-* checkboxes.
       Hours present in the CSV → checked (blocked); absent → unchecked
       (allowed). Done with an explicit reset so a switch from a version
       with many blocked hours to one with few clears the extras. */
    function setBlockedHours(csv) {
      var hSet = {};
      String(csv || "").split(",").forEach(function (s) {
        var t = s.trim(); if (t !== "") hSet[parseInt(t, 10)] = true;
      });
      for (var h = 0; h <= 23; h++) {
        var cb = document.getElementById("bs-bh-" + h);
        if (cb) cb.checked = !!hSet[h];
      }
    }
    if (p) {
      setVal("bs-ema-long",   p.ema_long);
      setVal("bs-stop-pips",  p.fractal_stop_pips);
      setVal("bs-rrr-reward", p.rrr_reward);
      setVal("bs-max-dd",     p.max_daily_losses);
      /* Bug fix (May 2026): pre-fill the EMA filter checkbox from the
         version's params.use_ema_filter so the user sees the state
         that will be used for the next backtest. Previously the BD had
         no visible toggle for this — the strategy subprocess silently
         read USE_EMA_FILTER from env, and discrepancies between the
         version's setting and the BD's default-on state went unnoticed. */
      setCheckbox("bs-use-ema-filter", p.use_ema_filter);
      /* Bug fix (May 2026): pre-fill Direction and Blocked Hours from
         the version's params. Discovery holds these fixed in Phase 1
         (short_only / 4,5,6,8,10,11,14,17) — without applying them here
         the BD's user-tunable Direction dropdown and blocked-hours
         checkboxes silently override the Discovery setting on the next
         run, producing materially different results. */
      if (p.trade_direction) setVal("bs-direction-select", p.trade_direction);
      if (p.blocked_hours !== undefined) setBlockedHours(p.blocked_hours);
    } else {
      /* Unassigned version: restore inputs from localStorage so a
         previous stamp from an assigned version doesn't bleed through. */
      var ls = window.localStorage;
      setVal("bs-ema-long",   ls.getItem("bs_ema_long"));
      setVal("bs-stop-pips",  ls.getItem("bs_stop_pips"));
      setVal("bs-rrr-reward", ls.getItem("bs_rrr_reward"));
      setVal("bs-max-dd",     ls.getItem("bs_max_dd"));
      var storedFilter = ls.getItem("bs_use_ema_filter");
      if (storedFilter !== null) setCheckbox("bs-use-ema-filter", storedFilter === "true");
      var storedDir = ls.getItem("bs_direction");
      if (storedDir) setVal("bs-direction-select", storedDir);
      var storedBH = ls.getItem("bs_blocked_hours");
      if (storedBH !== null) setBlockedHours(storedBH);
    }
  }

  /* Active-version sync (May 2026 redesign). The version dropdown is gone;
     the active version is set exclusively on /versions and the BD reads
     it from /api/versions on every load. We:
       1. Update the "Active: vN" indicator in the top nav.
       2. Stash the active version on window.__activeVersion so other code
          (sidebar render, title row, RA title sync) can read it.
       3. Re-drive the sidebar + content if report.html's IIFE has already
          mounted with the wrong currentVersion (e.g. on first load before
          window.__activeVersionId was set, or after the /versions page
          changed it and the user navigated back here). The
          window._rbSetActiveVersion helper is exposed by the strategy
          template's IIFE for exactly this purpose.
       4. Apply the active version's params to the bs-* inputs so the
          settings panel reflects what the next run will use. */
  fetch("/api/versions").then(function (r) { return r.json(); }).then(function (store) {
    var allVersions = (store && store.versions) || [];
    var versions = allVersions.filter(function (v) { return v && v.params; });
    var activeId = store && store.active_version_id;
    var active = null;
    for (var i = 0; i < versions.length; i++) {
      if (versions[i].id === activeId) { active = versions[i]; break; }
    }
    if (!active && versions.length) active = versions[0];

    _setActiveIndicator(active ? active.name : null);
    window.__activeVersion = active || null;
    window.__activeVersionId = active ? active.id : "";

    if (active) {
      /* Sync the strategy template's private `currentVersion`. If the
         template already mounted with the right id (window.__activeVersionId
         was injected server-side before the IIFE ran) this is a no-op.
         Otherwise it reroutes the sidebar to the right version's runs. */
      if (typeof window._rbSetActiveVersion === "function") {
        window._rbSetActiveVersion(active.id);
      }
      window._currentVersionName        = active.name || "";
      window._currentVersionDisplayName = active.name || active.strategy_version || "";
      if (typeof updateRangeButtonLabel === "function") updateRangeButtonLabel();
      /* Apply params AFTER _rbSetActiveVersion because that helper can
         re-render the settings panel and wipe the bs-* values. */
      applyBacktestParamsFromVersion(active);
    }
  }).catch(function () {});

  /* ── Update Add Date Range button label on load and on version tab clicks ── */
  setTimeout(function () {
    updateRangeButtonLabel();
    /* Listen for version/run clicks to update the label dynamically */
    document.addEventListener("click", function (e) {
      var item = e.target.closest(".v-item");
      if (item) {
        setTimeout(updateRangeButtonLabel, 100);
      }
    });
  }, 100);
});

function setRunning() {
  var btns = [document.getElementById("run-new-btn"), document.getElementById("run-range-btn"),
              document.getElementById("copy-btn"), document.getElementById("cbot-btn")];
  btns.forEach(function (b) { if (b) b.disabled = true; });
  document.getElementById("run-status").innerHTML =
    '<span class="rb-spin"></span><span id="rb-progress-text">Starting\u2026</span>';
  document.getElementById("run-status").style.color = "#9090c0";
}

function resetButtons() {
  var newBtn   = document.getElementById("run-new-btn");
  var rangeBtn = document.getElementById("run-range-btn");
  newBtn.disabled   = false;
  newBtn.innerHTML   = "&#9654;&nbsp; Add Year";
  rangeBtn.disabled = false;
  /* Uncheck all monthly checkboxes and re-enable date inputs */
  document.querySelectorAll(".mo-check:checked").forEach(function (cb) { cb.checked = false; });
  var startEl = document.getElementById("rb-start");
  var endEl   = document.getElementById("rb-end");
  if (startEl) startEl.disabled = false;
  if (endEl)   endEl.disabled   = false;
  updateRangeButtonLabel();
}

function createCbot() {
  var btn = document.getElementById("cbot-btn");
  if (!btn || btn.disabled) return;

  /* Read current version + run data from globals set by renderContent */
  var ver = window._cbotVersion;
  var run = window._cbotRun;
  if (!ver || !run) { return; }

  var stratVer = ver.strategy_version || ver.name || "v1";
  var payload = {
    strategy_version: stratVer,
    ema_short:        String(run.ema_short   || (ver.params && ver.params.ema_short) || 8),
    ema_mid:          String(run.ema_mid     || (ver.params && ver.params.ema_mid)   || 20),
    ema_long:         String(run.ema_long    || (ver.params && ver.params.ema_long)  || 40),
    stop_loss_pips:   String(run.stop_loss_pips || (ver.params && ver.params.stop_loss_pips) || 15),
    rrr_risk:         String(run.rrr_risk    || (ver.params && ver.params.rrr_risk)  || 1),
    rrr_reward:       String(run.rrr_reward  || (ver.params && ver.params.rrr_reward) || 2),
    max_daily_losses: String(run.max_daily_losses || (ver.params && ver.params.max_daily_losses) || 2),
    trade_direction:  run.trade_direction || (ver.params && ver.params.trade_direction) || "both",
    blocked_hours:    Array.isArray(run.blocked_hours) ? run.blocked_hours.join(",") : String(run.blocked_hours || ""),
    instrument:       run.instrument || (ver.params && ver.params.ticker ? ver.params.ticker.replace(/=X$/i, "") : "EURUSD")
  };

  btn.disabled = true;
  btn.textContent = "Generating\u2026";

  fetch("/generate_cbot", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  })
  .then(function (r) {
    if (!r.ok) return r.json().then(function (d) { throw new Error(d.error || "Server error"); });
    var disp = r.headers.get("Content-Disposition") || "";
    var match = disp.match(/filename=(.+)/);
    var fname = match ? match[1] : "FractalBot_" + stratVer + ".cs";
    return r.blob().then(function (blob) { return { blob: blob, fname: fname }; });
  })
  .then(function (data) {
    /* Trigger browser download */
    var url = URL.createObjectURL(data.blob);
    var a = document.createElement("a");
    a.href = url; a.download = data.fname;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    btn.textContent = "\u2713 Downloaded!";
    btn.classList.add("downloaded");
    if (typeof showToast === "function") showToast("cBot downloaded: " + data.fname);
    setTimeout(function () {
      btn.textContent = "Create cBot";
      btn.classList.remove("downloaded");
      btn.disabled = false;
    }, 2200);
  })
  .catch(function (err) {
    btn.textContent = "Failed";
    if (typeof showToast === "function") showToast("cBot generation failed: " + err.message);
    setTimeout(function () { btn.textContent = "Create cBot"; btn.disabled = false; }, 2500);
  });
}

function getCurrentVersionName() {
  /* Dropdown removed (May 2026). The active version id is the canonical
     source of "which version's params should this run use", injected
     server-side via window.__activeVersionId. Fall back to legacy hints
     for robustness. */
  if (window.__activeVersionId) return window.__activeVersionId;
  if (window._currentVersionName) return window._currentVersionName;
  var sel = document.getElementById("version-select");
  if (sel && sel.value) return sel.value;
  /* Fallback: parse the versions-data JSON for the last version */
  var script = document.getElementById("versions-data");
  if (script) {
    try {
      var versions = JSON.parse(script.textContent);
      if (versions.length > 0) return versions[versions.length - 1].name;
    } catch(e) {}
  }
  return "";
}

function updateRangeButtonLabel() {
  /* Dropdown removed (May 2026). Read the active version's display name
     from window.__activeVersion.name (set by the run-bar IIFE after
     /api/versions resolves) so the button label always tracks the
     globally-active version. */
  var av = window.__activeVersion;
  var displayName = (av && (av.name || av.strategy_version)) ||
                    window.__activeVersionId ||
                    window._currentVersionDisplayName ||
                    getCurrentVersionName();
  var rangeBtn = document.getElementById("run-range-btn");
  if (!rangeBtn) return;
  if (displayName) {
    rangeBtn.innerHTML = "&#9654;&nbsp; Add Date Range (" + displayName + ")";
  } else {
    rangeBtn.innerHTML = "&#9654;&nbsp; Add Date Range";
  }
}

function getSelectedVersion() {
  /* Dropdown removed (May 2026). The active version is set on /versions
     and surfaced server-side via window.__activeVersionId (injected by the
     / route handler) — read from there. Fall back to the legacy dropdown
     for backwards compat if anything still renders it, then to "v1". */
  if (window.__activeVersionId) return window.__activeVersionId;
  var el = document.getElementById("version-select");
  if (el && el.value) return el.value;
  return "v1";
}

function getSelectedDirection() {
  var el = document.getElementById("bs-direction-select");
  if (el) return el.value;
  var stored = localStorage.getItem("bs_direction");
  return stored || "short_only";
}

function getSelectedInstrument() {
  /* Instrument toolbar dropdown was removed (May 2026); instrument is
     now per-version. Returning "" tells the run handlers in server.py
     to skip the INSTRUMENT env override, which lets
     _apply_active_version_to_env layer the active version's
     params.instrument in. The legacy localStorage `rb_instrument` value
     (if present from earlier sessions) is intentionally ignored. */
  var el = document.getElementById("instrument-select");
  if (el) return el.value;
  return "";
}

function getSelectedInterval() {
  var el = document.getElementById("bs-interval-select");
  if (el) return el.value;
  var stored = localStorage.getItem("bs_interval");
  return stored || "5m";
}

function getSelectedEmaShort() {
  var el = document.getElementById("bs-ema-short");
  if (el) return el.value;
  var stored = localStorage.getItem("bs_ema_short");
  return stored || "8";
}

function getSelectedEmaMid() {
  var el = document.getElementById("bs-ema-mid");
  if (el) return el.value;
  var stored = localStorage.getItem("bs_ema_mid");
  return stored || "20";
}

function getSelectedEmaLong() {
  var el = document.getElementById("bs-ema-long");
  if (el) return el.value;
  var stored = localStorage.getItem("bs_ema_long");
  return stored || "40";
}

function getSelectedStopPips() {
  var el = document.getElementById("bs-stop-pips");
  if (el) return el.value;
  var stored = localStorage.getItem("bs_stop_pips");
  return stored || "15";
}

function getSelectedRrrRisk() {
  var el = document.getElementById("bs-rrr-risk");
  if (el) return el.value;
  var stored = localStorage.getItem("bs_rrr_risk");
  return stored || "1";
}

function getSelectedRrrReward() {
  var el = document.getElementById("bs-rrr-reward");
  if (el) return el.value;
  var stored = localStorage.getItem("bs_rrr_reward");
  return stored || "2";
}

function getSelectedMaxDD() {
  var el = document.getElementById("bs-max-dd");
  if (el) return el.value;
  var stored = localStorage.getItem("bs_max_dd");
  return stored || "2";
}

function getSelectedBlockedHours() {
  var checked = [];
  for (var h = 0; h <= 23; h++) {
    var cb = document.getElementById("bs-bh-" + h);
    if (cb && cb.checked) checked.push(h);
  }
  if (checked.length > 0) return checked.join(",");
  var stored = localStorage.getItem("bs_blocked_hours");
  return stored || "";
}

function getSelectedApplySlippage() {
  var el = document.getElementById("bs-apply-slippage");
  if (el) return el.checked ? "true" : "false";
  var stored = localStorage.getItem("bs_apply_slippage");
  return stored === null ? "true" : stored;
}

function getSelectedSpreadPips() {
  var el = document.getElementById("bs-spread-pips");
  if (el) return el.value;
  var stored = localStorage.getItem("bs_spread_pips");
  return stored || "1.0";
}

function getSelectedSlSlippagePips() {
  var el = document.getElementById("bs-sl-slippage-pips");
  if (el) return el.value;
  var stored = localStorage.getItem("bs_sl_slippage_pips");
  return stored || "1.0";
}

/* Bug fix (May 2026): expose the EMA filter checkbox state to the run
   payload. The checkbox `bs-use-ema-filter` lives in the BD settings
   panel and is pre-filled from the selected version's params. If the
   checkbox isn't in the DOM yet (older report.html without the new row,
   or empty state), we fall back to localStorage, then to true (the
   default). Returns a boolean. */
function getSelectedUseEmaFilter() {
  var el = document.getElementById("bs-use-ema-filter");
  if (el) return !!el.checked;
  var stored = localStorage.getItem("bs_use_ema_filter");
  if (stored === "true")  return true;
  if (stored === "false") return false;
  return true;
}

function runNewVersion() {
  var instrument = getSelectedInstrument();
  var direction  = getSelectedDirection();
  var interval   = getSelectedInterval();
  var version    = getSelectedVersion();
  localStorage.setItem("rb_pending_run_type", "new_version_auto");
  localStorage.setItem("rb_strategy_version", version);
  setRunning();
  fetch("/run", { method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode: "new_version", instrument: instrument, direction: direction, interval: interval, strategy_version: version, version_id: version, ema_short: getSelectedEmaShort(), ema_mid: getSelectedEmaMid(), ema_long: getSelectedEmaLong(), stop_loss_pips: getSelectedStopPips(), rrr_risk: getSelectedRrrRisk(), rrr_reward: getSelectedRrrReward(), blocked_hours: getSelectedBlockedHours(), max_daily_losses: getSelectedMaxDD(), use_ema_filter: getSelectedUseEmaFilter(), apply_slippage: getSelectedApplySlippage(), spread_pips: getSelectedSpreadPips(), sl_slippage_pips: getSelectedSlSlippagePips() })
  })
  .then(function (r) { return r.json(); })
  .then(function (data) {
    if (data.started) { pollStatus(); }
    else { localStorage.removeItem("rb_pending_run_type"); resetButtons(); showError(data.error); }
  })
  .catch(function () { localStorage.removeItem("rb_pending_run_type"); resetButtons(); showError("Request failed"); });
}

function runDateRange() {
  /* ── Check for multi-select mode (monthly checkboxes) ──── */
  var selectedRanges = (typeof window.getSelectedMonthRanges === "function") ? window.getSelectedMonthRanges() : [];
  if (selectedRanges.length > 0) {
    /* Batch mode: run all selected date ranges sequentially */
    var instrument     = getSelectedInstrument();
    var targetVersion  = getCurrentVersionName();
    var version        = getSelectedVersion();
    localStorage.setItem("rb_pending_run_type", "date_range_batch");
    localStorage.setItem("rb_pending_run_version", targetVersion);
    setRunning();
    fetch("/run_batch", { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ranges: selectedRanges, instrument: instrument, target_version: targetVersion, strategy_version: version, version_id: version, direction: getSelectedDirection(), interval: getSelectedInterval(), ema_short: getSelectedEmaShort(), ema_mid: getSelectedEmaMid(), ema_long: getSelectedEmaLong(), stop_loss_pips: getSelectedStopPips(), rrr_risk: getSelectedRrrRisk(), rrr_reward: getSelectedRrrReward(), blocked_hours: getSelectedBlockedHours(), max_daily_losses: getSelectedMaxDD(), use_ema_filter: getSelectedUseEmaFilter(), apply_slippage: getSelectedApplySlippage(), spread_pips: getSelectedSpreadPips(), sl_slippage_pips: getSelectedSlSlippagePips() })
    })
    .then(function (r) { return r.json(); })
    .then(function (data) {
      if (data.started) { pollStatus(); }
      else { localStorage.removeItem("rb_pending_run_type"); localStorage.removeItem("rb_pending_run_version"); resetButtons(); showError(data.error); }
    })
    .catch(function () { localStorage.removeItem("rb_pending_run_type"); localStorage.removeItem("rb_pending_run_version"); resetButtons(); showError("Request failed"); });
    return;
  }
  /* ── Single date range mode ──── */
  var startDate = document.getElementById("rb-start").value;
  var endDate   = document.getElementById("rb-end").value;
  if (!startDate || !endDate) {
    showError("Select both start and end dates");
    return;
  }
  var instrument     = getSelectedInstrument();
  var targetVersion  = getCurrentVersionName();
  var version        = getSelectedVersion();
  localStorage.setItem("rb_pending_run_type", "date_range");
  localStorage.setItem("rb_pending_run_version", targetVersion);
  setRunning();
  fetch("/run_range", { method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ start_date: startDate, end_date: endDate, instrument: instrument, target_version: targetVersion, strategy_version: version, version_id: version, direction: getSelectedDirection(), interval: getSelectedInterval(), ema_short: getSelectedEmaShort(), ema_mid: getSelectedEmaMid(), ema_long: getSelectedEmaLong(), stop_loss_pips: getSelectedStopPips(), rrr_risk: getSelectedRrrRisk(), rrr_reward: getSelectedRrrReward(), blocked_hours: getSelectedBlockedHours(), max_daily_losses: getSelectedMaxDD(), use_ema_filter: getSelectedUseEmaFilter(), apply_slippage: getSelectedApplySlippage(), spread_pips: getSelectedSpreadPips(), sl_slippage_pips: getSelectedSlSlippagePips() })
  })
  .then(function (r) { return r.json(); })
  .then(function (data) {
    if (data.started) { pollStatus(); }
    else { localStorage.removeItem("rb_pending_run_type"); localStorage.removeItem("rb_pending_run_version"); resetButtons(); showError(data.error); }
  })
  .catch(function () { localStorage.removeItem("rb_pending_run_type"); localStorage.removeItem("rb_pending_run_version"); resetButtons(); showError("Request failed"); });
}

function showError(msg) {
  var status = document.getElementById("run-status");
  status.innerHTML   = "\\u2717\\u2009" + (msg || "Unknown error");
  status.style.color = "#ff6b6b";
}

function pollStatus() {
  fetch("/status")
    .then(function (r) { return r.json(); })
    .then(function (data) {
      if (data.running) {
        var pct = data.progress || 0;
        var txt  = document.getElementById("rb-progress-text");
        if (txt)  txt.textContent  = (data.stage || "Running\u2026") + " " + pct + "%";
        setTimeout(pollStatus, 1500);
      } else if (data.ok && data.no_data) {
        resetButtons();
        document.getElementById("run-status").innerHTML = "";
        showNoDataNotification();
      } else if (data.ok) {
        var status = document.getElementById("run-status");
        status.innerHTML   = "\\u2713\\u2009Complete \\u2014 refreshing\\u2026";
        status.style.color = "#6bcb77";
        setTimeout(function () { window.location.href = window.location.pathname + "?t=" + Date.now(); }, 900);
      } else {
        resetButtons();
        showError(data.error);
      }
    })
    .catch(function () { setTimeout(pollStatus, 2000); });
}

function showNoDataNotification() {
  var existing = document.getElementById("no-data-toast");
  if (existing) existing.remove();
  var toast = document.createElement("div");
  toast.id = "no-data-toast";
  toast.textContent = "No Data Available";
  document.body.appendChild(toast);
  /* Trigger reflow so the initial opacity:0 state is rendered before adding .show */
  toast.offsetHeight;
  toast.classList.add("show");
  setTimeout(function () {
    toast.classList.remove("show");
    setTimeout(function () { toast.remove(); }, 600);
  }, 3000);
}
</script>
"""

# ── Empty-state page (shown before the first backtest is run) ─────────────────

EMPTY_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Trading Bot</title>
</head>
<body style="
  margin: 0;
  background: #0c0c14; color: #9090b8;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  display: flex; align-items: center; justify-content: center;
  height: 100vh; flex-direction: column; gap: 12px; text-align: center;
">
  __INJECT__
  <span style="font-size: 36px; line-height: 1;">&#128202;</span>
  <span style="font-size: 15px; color: #c0c0e0;">No report yet</span>
  <span style="font-size: 13px; color: #505070; max-width: 340px; line-height: 1.6;">
    Click <strong style="color: #d0d0ee;">&#9654;&nbsp;Run New Version</strong>
    to run <code style="color:#4cc9f0">strategy.py</code>
    and generate the first report.
  </span>
</body>
</html>""".replace("__INJECT__", INJECT_HTML)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve report.html with the Run bar injected."""
    if not REPORT_FILE.exists():
        return Response(EMPTY_PAGE, mimetype="text/html")

    html = REPORT_FILE.read_text(encoding="utf-8")
    # Inject the run-bar at the START of <body>. Even though the version
    # and instrument dropdowns are gone (May 2026), the run-bar still owns
    # the date pickers, run buttons, and active-version indicator, all of
    # which need to exist before report.html's inline scripts read them.
    #
    # We also inject window.__activeVersionId / window.__activeVersion
    # SYNCHRONOUSLY (as a leading <script>) so the strategy template's
    # IIFE sees the right id when it calls populateVersionSelector and
    # picks the right sidebar bucket on first paint — no flash of the
    # wrong version's runs.
    av = _get_active_version() or {}
    av_id   = (av.get("id")   or "").replace("\\", "\\\\").replace('"', '\\"')
    av_name = (av.get("name") or "").replace("\\", "\\\\").replace('"', '\\"')
    sync_script = (
        '<script>'
        f'window.__activeVersionId = "{av_id}";'
        f'window.__activeVersionName = "{av_name}";'
        '</script>'
    )
    html = html.replace("<body>", "<body>\n" + sync_script + "\n" + INJECT_HTML, 1)
    return Response(html, mimetype="text/html")


@app.route("/style.css")
def serve_css():
    """Serve the dashboard stylesheet."""
    css_path = BASE_DIR / "style.css"
    if not css_path.exists():
        return Response("", mimetype="text/css")
    return Response(css_path.read_text(encoding="utf-8"), mimetype="text/css")


# ── /versions — strategy-profile manager ─────────────────────────────────────
# Renders a standalone page listing all versions with add / delete / make-active
# controls. Data is loaded client-side from /api/versions so the page stays in
# sync with the BD selector and the RA toggle persistence.

_VERSIONS_PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Versions — Fractal Bot</title>
  <link rel="stylesheet" href="/style.css">
</head>
<body class="versions-page">
  <nav class="top-nav" id="top-nav">
      <ul class="top-nav-items">
      <li><a class="top-nav-link" href="/">Backtesting</a></li>
      <li><a class="top-nav-link" href="/results/regime_analysis.html">Regimes</a></li>
      <li><a class="top-nav-link" href="/discovery">Discovery</a></li>
      <li><a class="top-nav-link top-nav-link-active" href="/versions">Versions</a></li>
    </ul>
    <span class="top-nav-active-version" id="top-nav-active-version"></span>
  </nav>

  <main class="versions-container">
    <header class="versions-header">
      <h1>Versions</h1>
      <!-- Add Version moved into the page header (May 2026): the standalone
           .versions-add-section + subtitle were removed; this button is the
           single control for creating a new unassigned slot. Errors surface
           in #versions-form-error below the list. -->
      <button type="button" id="versions-add-btn" class="rb-btn rb-btn-green">Add Version</button>
    </header>

    <section class="versions-list-section">
      <h2>Existing versions</h2>
      <ul id="versions-list" class="versions-list"></ul>
      <span id="versions-form-error" class="versions-form-error"></span>
    </section>
  </main>

  <script>
    (function () {
      var listEl    = document.getElementById("versions-list");
      var addBtnEl  = document.getElementById("versions-add-btn");
      var errEl     = document.getElementById("versions-form-error");
      var navAvEl   = document.getElementById("top-nav-active-version");

      function showError(msg) { errEl.textContent = msg || ""; }

      /* short_only / long_only / both → "Short Only" / "Long Only" / "Both"
         (May 2026). Parallel to the helper on the Discovery page so each
         card's title can render direction in the same human-friendly form. */
      function humanizeDirection(d) {
        if (!d) return "";
        var s = String(d).toLowerCase();
        if (s === "short_only") return "Short Only";
        if (s === "long_only")  return "Long Only";
        if (s === "both")       return "Both";
        return s.replace(/_/g, " ").replace(/\\b\\w/g, function (c) { return c.toUpperCase(); });
      }

      function renderList(store) {
        var active = store.active_version_id;
        /* Render newest-first (May 2026): versions.json keeps creation
           order (v1 oldest, vN newest) so the renumber-on-delete logic
           and _next_version_name stay correct. We just reverse a copy at
           render time so the top of the visual stack is the most
           recently added version. */
        var versions = (store.versions || []).slice().reverse();
        listEl.innerHTML = "";
        versions.forEach(function (v) {
          var isActive   = (v.id === active);
          var isAssigned = !!v.params;
          var runCount   = (typeof v.run_count === "number") ? v.run_count : 0;

          var li = document.createElement("li");
          li.className = "versions-row" + (isActive ? " versions-row-active" : "");
          if (!isAssigned) li.classList.add("versions-row-unassigned");

          /* Radio button (May 2026): the sole control for promoting a
             version to globally active. Only assigned versions can be
             made active — unassigned slots have no params/regime_state
             for the BD/RA to read, so we disable the radio for them. */
          var radioLabel = document.createElement("label");
          radioLabel.className = "versions-row-radio";
          radioLabel.title = isAssigned ? "Set as active version"
                                        : "Assign params from Discovery first";
          var radio = document.createElement("input");
          radio.type = "radio";
          radio.name = "active-version";
          radio.value = v.id;
          radio.checked = isActive;
          radio.disabled = !isAssigned;
          radio.addEventListener("change", function () {
            if (!radio.checked) return;
            fetch("/api/active_version", {
              method: "POST",
              headers: {"Content-Type": "application/json"},
              body: JSON.stringify({id: v.id}),
            })
              .then(function (r) { return r.json(); })
              .then(function (resp) {
                if (!resp.ok) { showError(resp.error || "Switch failed"); return; }
                /* Re-render to update card styling + indicator. */
                refresh();
              })
              .catch(function (e) { showError("Switch failed: " + e.message); });
          });
          radioLabel.appendChild(radio);

          var nameSpan = document.createElement("span");
          nameSpan.className = "versions-row-name";
          /* Primary identifier — the version id (v7, v8, …). */
          var idEl = document.createElement("span");
          idEl.className = "versions-row-name-id";
          idEl.textContent = v.name;
          nameSpan.appendChild(idEl);
          /* May 2026: surface the version's instrument / interval /
             direction inline after the version id so the card title
             reads "v7 GBPUSD 5m Short Only". Skip for unassigned slots
             — those have no params yet and surface a placeholder strip
             in the row below. */
          if (isAssigned) {
            var p = v.params || {};
            if (p.instrument) {
              var instEl = document.createElement("span");
              instEl.className = "versions-row-name-meta";
              instEl.textContent = String(p.instrument).toUpperCase();
              nameSpan.appendChild(instEl);
            }
            if (p.interval) {
              var ivEl = document.createElement("span");
              ivEl.className = "versions-row-name-meta";
              ivEl.textContent = String(p.interval);
              nameSpan.appendChild(ivEl);
            }
            if (p.trade_direction) {
              var dirEl = document.createElement("span");
              dirEl.className = "versions-row-name-meta";
              dirEl.textContent = humanizeDirection(p.trade_direction);
              nameSpan.appendChild(dirEl);
            }
          }
          if (isActive) {
            var badge = document.createElement("span");
            badge.className = "versions-row-active-badge";
            badge.textContent = "ACTIVE";
            nameSpan.appendChild(badge);
          }

          var actionsSpan = document.createElement("span");
          actionsSpan.className = "versions-row-actions";
          var delBtn = document.createElement("button");
          delBtn.type = "button";
          delBtn.className = "rb-btn rb-btn-delete";
          delBtn.textContent = "Delete";
          if (versions.length <= 1) delBtn.disabled = true;
          delBtn.addEventListener("click", function () {
            deleteVersion(v, isAssigned, runCount);
          });
          actionsSpan.appendChild(delBtn);

          /* Free-form notes — unchanged from prior behaviour, auto-saves
             on blur to /api/versions/<id>/notes. */
          var notesWrap = document.createElement("div");
          notesWrap.className = "versions-row-notes";
          var notesLabel = document.createElement("label");
          notesLabel.className = "versions-row-notes-label";
          notesLabel.textContent = "Notes";
          var notesArea = document.createElement("textarea");
          notesArea.className = "versions-row-notes-textarea";
          notesArea.rows = 3;
          notesArea.placeholder = "Free-form notes for this version — strategy intent, notable runs, observations…";
          notesArea.value = v.notes || "";
          var notesStatus = document.createElement("span");
          notesStatus.className = "versions-row-notes-status";
          notesArea.addEventListener("blur", function () {
            var newVal = notesArea.value;
            if (newVal === (v.notes || "")) return;
            notesStatus.textContent = "Saving…";
            fetch("/api/versions/" + encodeURIComponent(v.id) + "/notes", {
              method: "POST",
              headers: {"Content-Type": "application/json"},
              body: JSON.stringify({notes: newVal}),
            })
              .then(function (r) { return r.json(); })
              .then(function (resp) {
                if (resp.ok) {
                  v.notes = newVal;
                  notesStatus.textContent = "Saved";
                  setTimeout(function () { notesStatus.textContent = ""; }, 2000);
                } else {
                  notesStatus.textContent = "Save failed: " + (resp.error || "?");
                }
              })
              .catch(function (e) { notesStatus.textContent = "Save failed: " + e.message; });
          });
          notesWrap.appendChild(notesLabel);
          notesWrap.appendChild(notesArea);
          notesWrap.appendChild(notesStatus);

          li.appendChild(radioLabel);
          li.appendChild(nameSpan);
          li.appendChild(actionsSpan);
          li.appendChild(notesWrap);

          /* Unassigned slots get a placeholder strip in place of any
             parameter display — the card itself stays intentionally
             empty of param info per spec. */
          if (!isAssigned) {
            var placeholder = document.createElement("span");
            placeholder.className = "versions-row-placeholder";
            placeholder.textContent = "Awaiting parameter assignment from Discovery";
            li.appendChild(placeholder);
          }

          listEl.appendChild(li);
        });
        var activeVersion = versions.find(function (v) { return v.id === active; }) || versions[0];
        if (activeVersion && navAvEl) {
          navAvEl.textContent = "Active: " + activeVersion.name;
        }
      }

      function refresh() {
        return fetch("/api/versions")
          .then(function (r) { return r.json(); })
          .then(renderList)
          .catch(function (e) { showError("Failed to load versions: " + e.message); });
      }

      /* Delete flow:
         - Unassigned, no runs: single plain confirm.
         - Assigned with runs:  warn that all associated BD run history
                                will also be deleted.
         - Assigned, no runs:   plain confirm.
         After versions.json delete succeeds, if the version had runs we
         also call /delete_version to purge them from report.html so the
         BD sidebar doesn't carry orphaned buckets. */
      function deleteVersion(v, isAssigned, runCount) {
        /* \\u201C / \\u201D = curly double quotes. Escaping these (and the
           \\n separators below) is required because this script lives inside
           a Python triple-quoted string — bare \\n in the .py source becomes
           a real newline, which is a syntax error inside a JS string literal
           and silently bricks the entire IIFE on the /versions page. */
        var msg;
        if (isAssigned && runCount > 0) {
          msg = "Delete version \\u201C" + v.name + "\\u201D?\\n\\n" +
                "WARNING: " + runCount + " associated BD run" +
                (runCount === 1 ? "" : "s") +
                " will also be deleted. This cannot be undone.";
        } else {
          msg = "Delete version \\u201C" + v.name + "\\u201D?";
        }
        if (!window.confirm(msg)) return;

        fetch("/api/versions/" + encodeURIComponent(v.id), {method: "DELETE"})
          .then(function (r) { return r.json(); })
          .then(function (resp) {
            if (!resp.ok) { showError(resp.error || "Delete failed"); return; }
            showError("");
            /* May 2026: the server-side DELETE handler now atomically
               drops the version, removes its bucket from report.html,
               renumbers the remaining versions to v1..vN, and returns
               the rename map. We just need to rewrite localStorage's
               discovery_trial_assignments so any "Assigned → vN"
               buttons on the Discovery page reflect the new ids on
               their next render. */
            applyRenameToLocalAssignments(resp.rename_map || {});
            return refresh();
          });
      }

      /* Rewrite localStorage discovery_trial_assignments so trial buttons
         keep showing the correct "Assigned → vN" label after a renumber.
         No-op for empty maps. Mirrors the saveAssignment writer on the
         Discovery page (key: discovery_trial_assignments). */
      function applyRenameToLocalAssignments(renameMap) {
        if (!renameMap || !Object.keys(renameMap).length) return;
        try {
          var raw = window.localStorage.getItem("discovery_trial_assignments");
          if (!raw) return;
          var m = JSON.parse(raw);
          var changed = false;
          Object.keys(m).forEach(function (trialId) {
            var rec = m[trialId];
            if (rec && renameMap[rec.version]) {
              rec.version = renameMap[rec.version];
              changed = true;
            }
          });
          if (changed) {
            window.localStorage.setItem("discovery_trial_assignments", JSON.stringify(m));
          }
        } catch (e) { /* localStorage unavailable / parse error — best effort */ }
      }

      addBtnEl.addEventListener("click", function () {
        showError("");
        fetch("/api/versions", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: "{}"
        }).then(function (r) { return r.json(); }).then(function (resp) {
          if (!resp.ok) { showError(resp.error || "Add failed"); return; }
          refresh();
        });
      });

      refresh();
    })();
  </script>
</body>
</html>
"""

@app.route("/versions")
def versions_page():
    """Render the Versions management page."""
    return Response(_VERSIONS_PAGE_HTML, mimetype="text/html")


# ── /discovery — Phase 1 random parameter search UI ──────────────────────────
# Standalone page (no INJECT_HTML — Discovery has its own config bar). All
# styling lives in style.css under .discovery-*. The page polls
# /api/discovery/status while a run is in flight and hydrates the results
# table from /api/discovery/results on load so the last run's output is
# visible without re-running.

_DISCOVERY_PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Discovery — Fractal Bot</title>
  <link rel="stylesheet" href="/style.css">
</head>
<body class="discovery-page">
  <nav class="top-nav" id="top-nav">
    <ul class="top-nav-items">
      <li><a class="top-nav-link" href="/">Backtesting</a></li>
      <li><a class="top-nav-link" href="/results/regime_analysis.html">Regimes</a></li>
      <li><a class="top-nav-link top-nav-link-active" href="/discovery">Discovery</a></li>
      <li><a class="top-nav-link" href="/versions">Versions</a></li>
    </ul>
    <span class="top-nav-active-version" id="top-nav-active-version"></span>
  </nav>

  <main class="discovery-container">
    <header class="discovery-header">
      <h1>Discovery</h1>
      <!-- Settings gear moved into the page header (May 2026), parallel to
           the BD + RA + Versions pattern: title left, control on the
           right. The standalone .discovery-settings-header div + its h2
           "Discovery Settings" + the outer .discovery-config card were
           retired; the collapsible body below drops down inline when the
           gear is clicked. The button id (#d-settings-toggle) and chevron
           id (#d-settings-chevron) are preserved so the existing toggle
           JS still finds them. -->
      <button type="button" class="bs-toggle-btn" id="d-settings-toggle"
              title="Toggle Discovery Settings" aria-label="Toggle Discovery Settings">
        <svg id="d-settings-chevron" width="16" height="16" viewBox="0 0 16 16" fill="none">
          <circle cx="8" cy="8" r="7" stroke="currentColor" stroke-width="1.5"/>
          <path d="M5.5 7L8 9.5L10.5 7" stroke="currentColor" stroke-width="1.5"
                stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </button>
    </header>

    <!-- ── Discovery Settings body (collapsible) ──────────────────────────
         Plain [hidden] attribute pattern instead of .bs-collapsible's
         max-height transition (cascade quirk on this page; [hidden] is
         bulletproof). Surfaces what's fixed, what's being searched, and
         what counts as passing. -->
    <section class="discovery-settings-panel">
      <div class="discovery-settings-body" id="d-settings-collapsible" hidden>
        <div class="discovery-settings-group">
          <div class="section-title">Fixed Constants</div>
          <table><tbody>
            <tr><td class="lbl">Instrument</td>
                <td><select id="ds-instrument" class="discovery-form-input ds-select">
                      <option value="GBPUSD">GBPUSD</option>
                      <option value="EURUSD">EURUSD</option>
                    </select></td></tr>
            <tr><td class="lbl">Interval</td>
                <td><select id="ds-interval" class="discovery-form-input ds-select">
                      <option value="1m">1m</option>
                      <option value="5m">5m</option>
                      <option value="15m">15m</option>
                      <option value="1h">1h</option>
                    </select></td></tr>
            <tr><td class="lbl">Direction</td>
                <td><select id="ds-direction" class="discovery-form-input ds-select">
                      <option value="short_only">Short only</option>
                      <option value="long_only">Long only</option>
                      <option value="both">Both</option>
                    </select></td></tr>
            <tr><td class="lbl">Blocked Hours</td>
                <td>
                  <div class="bs-hours-grid" id="ds-blocked-hours-grid">
                    <!-- 24 pills (0–23) wired in JS below from localStorage / default
                         set. Reuses BD's .bs-hour-cb + .bs-hour-label styling so the
                         red-when-blocked treatment matches the rest of the platform. -->
                  </div>
                  <div class="text-dim ds-hint">UTC</div>
                </td></tr>
            <tr><td class="lbl">Slippage</td>
                <td><span class="val-highlight">On</span> <span class="text-dim">— 1 pip SL slippage + 1 pip spread (read-only)</span></td></tr>
          </tbody></table>
        </div>
        <div class="discovery-settings-group">
          <div class="section-title">Search Bounds</div>
          <table><tbody>
            <tr><td class="lbl">EMA Long</td>      <td><span class="val-highlight">10 – 200</span></td></tr>
            <tr><td class="lbl">Stop Loss</td>     <td><span class="val-highlight">5 – 50 pips</span></td></tr>
            <tr><td class="lbl">RRR Reward</td>    <td><span class="val-highlight">1 – 5</span> <span class="text-dim">(risk fixed at 1)</span></td></tr>
            <tr><td class="lbl">Max DD</td>        <td><span class="val-highlight">1 – 5</span></td></tr>
            <tr><td class="lbl">EMA Filter</td>    <td><span class="val-highlight">on / off</span></td></tr>
            <tr><td class="lbl">Macro Regimes</td> <td><span class="val-highlight">any combination of 5</span></td></tr>
            <tr><td class="lbl">Micro Regimes</td> <td><span class="val-highlight">any combination of 10</span></td></tr>
          </tbody></table>
        </div>
        <div class="discovery-settings-group">
          <div class="section-title">Passing Criteria</div>
          <table><tbody>
            <tr><td class="lbl">Profit Factor</td>
                <td>
                  <span class="ds-op">&ge;</span>
                  <input type="number" id="ds-min-pf" class="discovery-form-input ds-num-input"
                         step="0.05" min="0.1" value="1.5">
                </td></tr>
            <tr><td class="lbl">Min Trades</td>
                <td>
                  <span class="ds-op">&ge;</span>
                  <input type="number" id="ds-min-trades" class="discovery-form-input ds-num-input"
                         step="1" min="0" value="50">
                </td></tr>
            <tr><td class="lbl">Max Drawdown</td>
                <td>
                  <span class="ds-op">&le;</span>
                  <input type="number" id="ds-max-dd-pct" class="discovery-form-input ds-num-input"
                         step="0.5" min="0" max="100" value="15">
                  <span class="text-dim">%</span>
                </td></tr>
          </tbody></table>
        </div>
      </div>
    </section>

    <section class="discovery-config">
      <h2>Run configuration</h2>
      <form id="discovery-config-form" class="discovery-config-form">
        <div class="discovery-config-field">
          <label class="discovery-form-label" for="d-start">Start date</label>
          <input type="date" id="d-start" class="rb-date" value="2025-07-01">
        </div>
        <div class="discovery-config-field">
          <label class="discovery-form-label" for="d-end">End date</label>
          <input type="date" id="d-end" class="rb-date" value="2025-12-31">
        </div>
        <div class="discovery-config-field">
          <label class="discovery-form-label" for="d-trials">Trials</label>
          <input type="number" id="d-trials" class="discovery-form-input" value="200" min="1" max="10000">
        </div>
        <div class="discovery-config-field">
          <label class="discovery-form-label" for="d-seed">Seed</label>
          <input type="number" id="d-seed" class="discovery-form-input" placeholder="random">
        </div>
        <button type="submit" id="d-run-btn" class="rb-btn rb-btn-green discovery-run-btn">&#9654; Run Discovery</button>
      </form>
      <div class="discovery-progress-row">
        <span id="d-status-line" class="discovery-status-line">Idle.</span>
        <div class="discovery-progress-bar">
          <div id="d-progress-fill" class="discovery-progress-fill"></div>
        </div>
      </div>
    </section>

    <section class="discovery-results-section">
      <div class="discovery-results-header">
        <h2>Results</h2>
        <span id="d-stack-count" class="discovery-result-count"></span>
      </div>
      <!-- Stack of per-run blocks — populated client-side from
           /api/discovery/results. Each block renders the existing
           sortable trials table with its own show-passing-only toggle
           and a Delete button that removes that run from the array. -->
      <div id="d-runs-stack" class="discovery-runs-stack"></div>
      <div id="d-empty-state" class="discovery-empty-state" hidden>
        No discovery runs yet. Configure above and click <strong>Run Discovery</strong>.
      </div>
    </section>
  </main>

  <div id="d-assign-modal" class="discovery-modal" hidden>
    <div class="discovery-modal-backdrop"></div>
    <div class="discovery-modal-content">
      <h3>Add to version</h3>
      <p class="discovery-modal-trial" id="d-assign-trial-summary"></p>
      <label class="discovery-form-label" for="d-assign-version">Version slot</label>
      <select id="d-assign-version" class="discovery-form-input discovery-modal-input"></select>
      <p id="d-assign-empty-hint" class="discovery-modal-empty-hint" hidden>
        No unassigned versions available — create one on the
        <a href="/versions">Versions page</a>.
      </p>
      <div class="discovery-modal-actions">
        <button type="button" id="d-assign-cancel" class="rb-btn rb-btn-delete">Cancel</button>
        <button type="button" id="d-assign-confirm" class="rb-btn rb-btn-green">Add to version</button>
      </div>
      <span id="d-assign-error" class="discovery-modal-error"></span>
    </div>
  </div>

  <script>
    (function () {
      /* ── State ────────────────────────────────────────────────────────── */
      var STATE = {
        runs:           [],     /* array of run dicts, newest first */
        blockState:     {},     /* per-run-id: {sortKey, sortDir, showPassingOnly} */
        polling:        false,
        assignTrialId:  null,
      };

      function getBlockState(runId) {
        if (!STATE.blockState[runId]) {
          STATE.blockState[runId] = {
            sortKey: "profit_factor",
            sortDir: -1,           /* -1 desc, +1 asc */
            showPassingOnly: true, /* matches the default checked checkbox */
          };
        }
        return STATE.blockState[runId];
      }

      var POLL_MS = 1500;

      /* ── DOM refs ──────────────────────────────────────────────────────── */
      var formEl     = document.getElementById("discovery-config-form");
      var runBtn     = document.getElementById("d-run-btn");
      var startEl    = document.getElementById("d-start");
      var endEl      = document.getElementById("d-end");
      var trialsEl   = document.getElementById("d-trials");
      var seedEl     = document.getElementById("d-seed");
      var statusEl   = document.getElementById("d-status-line");
      var fillEl     = document.getElementById("d-progress-fill");
      var stackEl    = document.getElementById("d-runs-stack");
      var emptyEl    = document.getElementById("d-empty-state");
      var countEl    = document.getElementById("d-stack-count");
      var modalEl    = document.getElementById("d-assign-modal");
      var modalCfm   = document.getElementById("d-assign-confirm");
      var modalCnc   = document.getElementById("d-assign-cancel");
      var modalSum   = document.getElementById("d-assign-trial-summary");
      var modalSel   = document.getElementById("d-assign-version");
      var modalEmpty = document.getElementById("d-assign-empty-hint");
      var modalErr   = document.getElementById("d-assign-error");
      var navAvEl    = document.getElementById("top-nav-active-version");

      /* ── Active-version indicator ────────────────────────────────────── */
      fetch("/api/active_version")
        .then(function (r) { return r.json(); })
        .then(function (resp) {
          if (resp && resp.ok && resp.active && navAvEl) {
            navAvEl.textContent = "Active: " + resp.active.name;
          }
        })
        .catch(function () {});

      /* ── Formatting helpers ────────────────────────────────────────────── */
      function fmtPF(v) {
        if (v === null || v === undefined) return "∞";
        return Number(v).toFixed(2);
      }
      function fmtPct(v) { return (v === null || v === undefined) ? "—" : Number(v).toFixed(1) + "%"; }
      function fmtUSD(v) {
        if (v === null || v === undefined) return "—";
        var n = Number(v);
        var sign = n < 0 ? "-$" : "$";
        return sign + Math.abs(n).toLocaleString(undefined, {minimumFractionDigits: 0, maximumFractionDigits: 0});
      }
      function describeParams(p) {
        if (!p) return "";
        var emaFilter = p.use_ema_filter ? "on" : "off";
        return "EMA " + p.ema_long + " (" + emaFilter + ") | SL " + p.stop_loss_pips +
               "p | RRR 1:" + p.rrr_reward + " | DLL " + p.max_daily_losses +
               " | macro[" + (p.allowed_macro_regimes || []).length +
               "] micro[" + (p.allowed_micro_regimes || []).length + "]";
      }
      /* Integer formatter for the Trades / Wins / Losses cells. Renders an
         em-dash for missing values so older trial records (saved before
         strategy_v2 wrote the full metrics dict) degrade gracefully. */
      function fmtInt(v) {
        if (v === null || v === undefined) return "—";
        return String(Math.round(Number(v)));
      }
      /* max_daily_drawdown is stored as {dollar, pct} by strategy_v2's
         compute_metrics. We display the .pct (matches DD 1's percent
         format). Returns null if missing → fmtPct renders an em-dash. */
      function maxDailyDDPct(m) {
        var mdd = m && m.max_daily_drawdown;
        if (mdd && typeof mdd === "object" && mdd.pct !== undefined && mdd.pct !== null) {
          return Math.abs(Number(mdd.pct));
        }
        return null;
      }

      /* ── Assignment-state tracking (localStorage) ──────────────────────
         Bug fix (May 2026 — Issue 1): the row-level Assign button used to
         stay labelled "Assign" forever, with the only post-click feedback
         being a small status line inside the modal. Users couldn't tell
         from the table whether a trial had already been assigned, and
         could accidentally re-assign the same trial.
         Fix: persist trial_id → {version, at} in localStorage. The row
         button reads this map on render and shows "Assigned → vN" disabled
         if the trial was previously assigned. Survives reloads. */
      var ASSIGN_LS_KEY = "discovery_trial_assignments";
      function loadAssignments() {
        try {
          var raw = window.localStorage.getItem(ASSIGN_LS_KEY);
          return raw ? JSON.parse(raw) : {};
        } catch (e) { return {}; }
      }
      function saveAssignment(trialId, versionName) {
        var m = loadAssignments();
        m[trialId] = { version: versionName, at: new Date().toISOString() };
        try { window.localStorage.setItem(ASSIGN_LS_KEY, JSON.stringify(m)); } catch (e) {}
      }

      /* ── Sorting (per-block) ───────────────────────────────────────────── */
      function sortKeyOf(trial, key) {
        if (key === "pass") return trial.pass ? 1 : 0;
        if (key === "trial") return trial.trial;
        var m = trial.metrics || {};
        var v = m[key];
        /* Profit factor null = infinity = best */
        if (key === "profit_factor" && (v === null || v === undefined)) {
          return Number.POSITIVE_INFINITY;
        }
        /* max_daily_drawdown is an object {dollar, pct} — sort by abs(pct) */
        if (key === "max_daily_drawdown") {
          if (v && typeof v === "object" && v.pct !== undefined && v.pct !== null) {
            return Math.abs(Number(v.pct));
          }
          return 0;
        }
        return v == null ? 0 : v;
      }
      function makeComparator(bs) {
        return function (a, b) {
          var av = sortKeyOf(a, bs.sortKey);
          var bv = sortKeyOf(b, bs.sortKey);
          if (av < bv) return -1 * bs.sortDir;
          if (av > bv) return  1 * bs.sortDir;
          return a.trial - b.trial;
        };
      }

      function td(text, cls) {
        var el = document.createElement("td");
        if (cls) el.className = cls;
        el.textContent = text;
        return el;
      }

      /* ── Render: stack + per-block ─────────────────────────────────────── */
      function fmtRunDate(iso) {
        if (!iso) return "—";
        /* "2026-05-19T06:33:05Z" → "2026-05-19 06:33" */
        return iso.replace("T", " ").replace(/:\d{2}Z?$/, "");
      }

      /* short_only / long_only / both → "Short Only" / "Long Only" / "Both"
         (May 2026 redesign — surfaces the run's direction in the header). */
      function humanizeDirection(d) {
        if (!d) return "—";
        var s = String(d).toLowerCase();
        if (s === "short_only") return "Short Only";
        if (s === "long_only")  return "Long Only";
        if (s === "both")       return "Both";
        /* Generic fallback: snake_case → Title Case */
        return s.replace(/_/g, " ").replace(/\b\w/g, function (c) { return c.toUpperCase(); });
      }

      function renderStack() {
        var runs = STATE.runs || [];
        emptyEl.hidden = runs.length > 0;
        countEl.textContent = runs.length ? (runs.length + " run" + (runs.length === 1 ? "" : "s")) : "";
        stackEl.innerHTML = "";
        runs.forEach(function (run) {
          stackEl.appendChild(renderBlock(run));
        });
      }

      function renderBlock(run) {
        var bs = getBlockState(run.run_id);
        var trials = run.trials || [];
        var passCount = trials.filter(function (t) { return t.pass; }).length;

        var block = document.createElement("div");
        block.className = "discovery-run-block";
        block.setAttribute("data-run-id", run.run_id || "");

        /* ── Header ────────────────────────────────────────────────────── */
        var header = document.createElement("div");
        header.className = "discovery-run-header";

        var meta = document.createElement("div");
        meta.className = "discovery-run-meta";
        var cfg = run.config || {};

        /* May 2026 redesign: lead the header with the run's instrument,
           interval, and direction so the user can see at a glance what
           the run was actually testing — before the date range / counts.
           Order: instrument · interval · direction · date_range · trials ·
           passing · started_at · STATUS. */
        if (cfg.instrument) {
          meta.appendChild(spanCls("discovery-run-prop", String(cfg.instrument).toUpperCase()));
        }
        if (cfg.interval) {
          meta.appendChild(spanCls("discovery-run-prop", String(cfg.interval)));
        }
        if (cfg.direction) {
          meta.appendChild(spanCls("discovery-run-prop", humanizeDirection(cfg.direction)));
        }
        if (cfg.start && cfg.end) {
          meta.appendChild(spanCls("discovery-run-range", cfg.start + " → " + cfg.end));
        }
        meta.appendChild(spanCls("discovery-run-stat", trials.length + " / " + (run.trials_total || trials.length) + " trials"));
        meta.appendChild(spanCls("discovery-run-stat", passCount + " passing"));
        meta.appendChild(spanCls("discovery-run-date", fmtRunDate(run.started_at)));
        var statusClass = "discovery-run-status discovery-run-status-" + (run.status || "idle");
        meta.appendChild(spanCls(statusClass, (run.status || "—").toUpperCase()));
        header.appendChild(meta);

        var controls = document.createElement("div");
        controls.className = "discovery-run-controls";
        /* The per-block "Show passing only" checkbox was removed (May 2026)
           — passing-only is now the unconditional default. The Delete
           button remains the sole control in the header. */

        var delBtn = document.createElement("button");
        delBtn.type = "button";
        delBtn.className = "rb-btn rb-btn-delete discovery-run-delete";
        delBtn.textContent = "Delete";
        delBtn.addEventListener("click", function () {
          deleteRun(run);
        });
        /* Refuse to delete the in-progress run — server-side check is
           definitive, but disable the button locally as a UX hint. */
        if (run.status === "running" && STATE.polling) {
          delBtn.disabled = true;
          delBtn.title = "Wait for the run to finish before deleting it.";
        }
        controls.appendChild(delBtn);

        header.appendChild(controls);
        block.appendChild(header);

        /* ── In-progress indicator ────────────────────────────────────── */
        if (run.status === "running") {
          var progress = document.createElement("div");
          progress.className = "discovery-run-progress";
          var pct = (run.trials_total || 0) > 0 ? (trials.length / run.trials_total) * 100 : 0;
          var bar = document.createElement("div");
          bar.className = "discovery-progress-bar";
          var fill = document.createElement("div");
          fill.className = "discovery-progress-fill";
          fill.style.width = Math.min(100, Math.max(0, pct)) + "%";
          bar.appendChild(fill);
          progress.appendChild(bar);
          block.appendChild(progress);
        }

        /* ── Table ────────────────────────────────────────────────────── */
        var wrap = document.createElement("div");
        wrap.className = "discovery-table-wrap";
        var table = document.createElement("table");
        table.className = "discovery-table";
        var thead = document.createElement("thead");
        var theadTr = document.createElement("tr");
        /* Columns (updated May 2026): #, Pass, PF, P&L, Trades, Wins,
           Losses, Win %, DD 1 (max), DD 2 (max daily), Assign. Params
           column was dropped — full params are visible on the trial
           detail page (click any row). DD 1 = max_drawdown (run-level).
           DD 2 = max_daily_drawdown.pct (worst single-day drawdown).
           Both shown as percent; DD 2 is stored as {dollar, pct} so we
           pull .pct in renderBlockBody. */
        var cols = [
          ["trial",              "#"],
          ["pass",               "Pass"],
          ["profit_factor",      "PF"],
          ["net_profit",         "P&L"],
          ["total_trades",       "Trades"],
          ["winning_trades",     "Wins"],
          ["losing_trades",      "Losses"],
          ["win_rate",           "Win %"],
          ["max_drawdown",       "DD 1"],
          ["max_daily_drawdown", "DD 2"],
          [null,                 "Assign"],
        ];
        cols.forEach(function (c) {
          var th = document.createElement("th");
          var key = c[0];
          var label = c[1];
          if (key) {
            th.setAttribute("data-sort", key);
            var indicator = "";
            if (bs.sortKey === key) {
              indicator = " " + (bs.sortDir < 0 ? "↓" : "↑");
              th.classList.add("discovery-th-active");
            }
            th.innerHTML = label + indicator;
            th.addEventListener("click", function () {
              if (bs.sortKey === key) {
                bs.sortDir = -bs.sortDir;
              } else {
                bs.sortKey = key;
                bs.sortDir = -1;
              }
              renderBlockBody(block, run);
            });
          } else {
            th.innerHTML = label;
          }
          theadTr.appendChild(th);
        });
        thead.appendChild(theadTr);
        table.appendChild(thead);
        var tbody = document.createElement("tbody");
        tbody.className = "discovery-block-tbody";
        table.appendChild(tbody);
        wrap.appendChild(table);
        block.appendChild(wrap);

        renderBlockBody(block, run);
        return block;
      }

      function renderBlockBody(block, run) {
        var bs = getBlockState(run.run_id);
        var tbody = block.querySelector(".discovery-block-tbody");
        var trials = run.trials || [];
        /* Always filter to passing trials — the per-block "Show passing only"
           checkbox was removed (May 2026). When the run has zero passing
           trials we render a single colspanned "No passing results." row
           below instead of leaving an empty tbody. */
        var visible = trials.filter(function (t) { return t.pass === true; });
        visible.sort(makeComparator(bs));

        /* Sync the thead indicators for the active sort column without
           rebuilding the table. */
        var ths = block.querySelectorAll("thead th[data-sort]");
        ths.forEach(function (th) {
          var key = th.getAttribute("data-sort");
          th.classList.remove("discovery-th-active");
          var base = th.textContent.replace(/ [↑↓]$/, "");
          if (key === bs.sortKey) {
            th.classList.add("discovery-th-active");
            th.textContent = base + " " + (bs.sortDir < 0 ? "↓" : "↑");
          } else {
            th.textContent = base;
          }
        });

        tbody.innerHTML = "";
        if (visible.length === 0) {
          /* No passing trials — render a single colspanned message row
             instead of an empty tbody. colspan matches the 11-column header. */
          var emptyTr = document.createElement("tr");
          emptyTr.className = "discovery-run-empty";
          var emptyTd = document.createElement("td");
          emptyTd.setAttribute("colspan", "11");
          emptyTd.textContent = "No passing results.";
          emptyTr.appendChild(emptyTd);
          tbody.appendChild(emptyTr);
          return;
        }
        visible.forEach(function (t) {
          var m = t.metrics || {};
          var tr = document.createElement("tr");
          tr.className = "discovery-row" + (t.pass ? " discovery-row-pass" : " discovery-row-fail");
          /* Navigate by trial.id (globally unique) not trial.trial
             (number — only unique within a single run). Previously two
             runs each had a "trial 1" and clicking either row routed to
             the same /discovery/trial/1, returning the wrong report. */
          tr.addEventListener("click", function () {
            window.location.href = "/discovery/trial/" + encodeURIComponent(t.id);
          });

          tr.appendChild(td(String(t.trial)));

          var passTd = document.createElement("td");
          var badge = document.createElement("span");
          badge.className = "discovery-badge " + (t.pass ? "discovery-badge-pass" : "discovery-badge-fail");
          badge.textContent = t.pass ? "PASS" : (t.error ? "ERR" : "FAIL");
          if (t.error) {
            badge.title = t.error;
          } else if (!t.pass && t.fail_reasons && t.fail_reasons.length) {
            badge.title = t.fail_reasons.join("; ");
          }
          passTd.appendChild(badge);
          tr.appendChild(passTd);

          /* Data cells — order MUST match the cols[] definition in
             renderBlock above: PF, P&L, Trades, Wins, Losses, Win %,
             DD 1 (max_drawdown), DD 2 (max_daily_drawdown.pct). */
          tr.appendChild(td(fmtPF(m.profit_factor),            "discovery-cell-num"));
          tr.appendChild(td(fmtUSD(m.net_profit),              "discovery-cell-num"));
          tr.appendChild(td(fmtInt(m.total_trades),            "discovery-cell-num"));
          tr.appendChild(td(fmtInt(m.winning_trades),          "discovery-cell-num"));
          tr.appendChild(td(fmtInt(m.losing_trades),           "discovery-cell-num"));
          tr.appendChild(td(fmtPct(m.win_rate),                "discovery-cell-num"));
          tr.appendChild(td(fmtPct(m.max_drawdown),            "discovery-cell-num"));
          tr.appendChild(td(fmtPct(maxDailyDDPct(m)),          "discovery-cell-num"));

          var actionTd = document.createElement("td");
          if (t.pass) {
            var btn = document.createElement("button");
            btn.type = "button";
            var existing = loadAssignments()[t.id];
            if (existing && existing.version) {
              /* Trial was previously assigned — show as a disabled
                 "Assigned" indicator so the user gets persistent feedback
                 across reloads. */
              btn.className = "rb-btn rb-btn-blue discovery-assign-btn discovery-assign-btn-done";
              btn.textContent = "✓ Assigned → " + existing.version;
              btn.disabled = true;
              btn.title = "Assigned at " + (existing.at || "—") +
                          ". Clear localStorage to re-enable.";
            } else {
              btn.className = "rb-btn rb-btn-blue discovery-assign-btn";
              btn.textContent = "Assign";
              btn.addEventListener("click", function (e) {
                e.preventDefault();
                e.stopPropagation();
                openAssignModal(t);
              });
            }
            actionTd.appendChild(btn);
          } else {
            actionTd.textContent = "—";
          }
          tr.appendChild(actionTd);

          tbody.appendChild(tr);
        });
      }

      function spanCls(cls, text) {
        var s = document.createElement("span");
        s.className = cls;
        s.textContent = text;
        return s;
      }

      function deleteRun(run) {
        if (!confirm("Delete this discovery run? This permanently removes its results from disk.")) return;
        fetch("/api/discovery/results/" + encodeURIComponent(run.run_id), {method: "DELETE"})
          .then(function (r) { return r.json().then(function (j) { return {ok: r.ok, body: j}; }); })
          .then(function (resp) {
            if (!resp.ok || !resp.body.ok) {
              alert("Delete failed: " + ((resp.body && resp.body.error) || "unknown error"));
              return;
            }
            /* Drop the block state too so the run id is fully forgotten. */
            delete STATE.blockState[run.run_id];
            loadResults();
          })
          .catch(function () { alert("Delete request failed."); });
      }

      /* ── Assign modal ──────────────────────────────────────────────────── */
      /* The flow is now: pick an EXISTING unassigned version slot from a
         select (populated from /api/versions filtered to params == null)
         and write the trial's params into it. There is no "create new"
         path here — users create empty slots on /versions first, then
         come here to assign. If no unassigned slots exist, the confirm
         button is disabled and a hint points at /versions. */
      function openAssignModal(trial) {
        STATE.assignTrialId = trial.id;
        var m = trial.metrics || {};
        modalSum.textContent = "Trial #" + trial.trial + " — PF " + fmtPF(m.profit_factor) +
                               ", " + (m.total_trades || 0) + " trades, " +
                               fmtPct(m.max_drawdown) + " max DD";
        modalErr.textContent = "";
        modalSel.innerHTML = "";
        modalCfm.disabled = true;
        modalCfm.textContent = "Add to version";  /* reset in case a previous open left it as "✓ Assigned ..." */
        modalCfm.classList.remove("discovery-assign-btn-done");
        modalEmpty.hidden = true;
        modalEl.hidden = false;

        fetch("/api/versions")
          .then(function (r) { return r.json(); })
          .then(function (store) {
            var all = (store && store.versions) || [];
            var unassigned = all.filter(function (v) { return !v.params; });
            if (unassigned.length === 0) {
              modalEmpty.hidden = false;
              modalSel.hidden = true;
              modalCfm.disabled = true;
              return;
            }
            modalEmpty.hidden = true;
            modalSel.hidden = false;
            unassigned.forEach(function (v) {
              var opt = document.createElement("option");
              opt.value = v.id;
              opt.textContent = v.name;
              modalSel.appendChild(opt);
            });
            modalCfm.disabled = false;
          })
          .catch(function () {
            modalErr.textContent = "Failed to load versions.";
          });
      }
      function closeAssignModal() {
        STATE.assignTrialId = null;
        modalEl.hidden = true;
      }
      modalCnc.addEventListener("click", closeAssignModal);
      modalEl.querySelector(".discovery-modal-backdrop").addEventListener("click", closeAssignModal);
      modalCfm.addEventListener("click", function () {
        if (!STATE.assignTrialId) return;
        var versionId = (modalSel.value || "").trim();
        if (!versionId) {
          modalErr.textContent = "Pick a version slot first.";
          return;
        }
        var trialIdAtClick = STATE.assignTrialId;
        modalCfm.disabled = true;
        modalErr.textContent = "";
        fetch("/api/discovery/assign", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({result_id: trialIdAtClick, version_id: versionId})
        })
          .then(function (r) { return r.json().then(function (j) { return {ok: r.ok, body: j}; }); })
          .then(function (resp) {
            if (!resp.ok || !resp.body.ok) {
              modalErr.textContent = (resp.body && resp.body.error) || "Failed to assign trial.";
              modalCfm.disabled = false;
              return;
            }
            var assignedName = resp.body.version.name;
            /* Bug fix Issue 1: in-modal button feedback + persistent
               row-button state via localStorage. */
            saveAssignment(trialIdAtClick, assignedName);
            modalCfm.textContent = "✓ Assigned to " + assignedName;
            modalCfm.classList.add("discovery-assign-btn-done");
            modalErr.textContent = "Now available in the BD dropdown.";
            setTimeout(function () {
              closeAssignModal();
              modalCfm.disabled = false;
              /* Re-render the stack so the row's Assign button picks up
                 its new "Assigned → vN" state from localStorage. */
              renderStack();
            }, 1200);
          })
          .catch(function () {
            modalErr.textContent = "Request failed.";
            modalCfm.disabled = false;
          });
      });

      /* ── Discovery Settings toggle (collapsed by default) ─────────────
         Uses the [hidden] attribute for show/hide and an inline transform
         on the SVG chevron for the rotation indicator. Avoids the
         .bs-collapsible / .bs-toggle-btn.open class rules whose inline-
         override behaviour was unreliable on this page in earlier
         testing (max-height stuck at 0 even with !important inline). */
      (function () {
        var btn     = document.getElementById("d-settings-toggle");
        var panel   = document.getElementById("d-settings-collapsible");
        var chevron = document.getElementById("d-settings-chevron");
        if (!btn || !panel) return;
        btn.addEventListener("click", function () {
          var isOpen = panel.hidden;  /* about to become open */
          panel.hidden = !isOpen;
          if (chevron) {
            chevron.style.transform = isOpen ? "rotate(180deg)" : "";
            chevron.style.transition = "transform 0.2s";
          }
        });
      })();

      /* ── Run + poll ────────────────────────────────────────────────────── */
      function setStatus(text) { statusEl.textContent = text; }
      function setProgress(pct) {
        var p = Math.max(0, Math.min(100, pct));
        fillEl.style.width = p + "%";
      }

      function pollStatus() {
        if (!STATE.polling) return;
        /* Each tick: refresh the global status line/progress bar from the
           lightweight /status endpoint, then refetch the full /results
           array so the in-progress block's tbody updates as trials land.
           For a 200-trial run the file grows to ~200KB at most; the cost
           is bounded and the simpler "always render canonical state"
           model avoids per-run-id stub bookkeeping. */
        fetch("/api/discovery/status")
          .then(function (r) { return r.json(); })
          .then(function (s) {
            var total = s.trials_total || 0;
            var done  = s.trials_complete || 0;
            var pct = total > 0 ? (done / total) * 100 : 0;
            setProgress(pct);

            return loadResults().then(function () { return s; });
          })
          .then(function (s) {
            if (s.running) {
              var bestPF = s.best && s.best.metrics ? fmtPF(s.best.metrics.profit_factor) : "—";
              var done = s.trials_complete || 0;
              var total = s.trials_total || 0;
              setStatus("Running: trial " + done + " / " + total + " — best PF so far: " + bestPF);
              setTimeout(pollStatus, POLL_MS);
            } else {
              STATE.polling = false;
              runBtn.disabled = false;
              if (s.status === "complete") {
                setStatus("Complete: " + (s.trials_complete || 0) + " / " + (s.trials_total || 0) + " trials.");
                setProgress(100);
              } else if (s.status === "error") {
                setStatus("Errored: " + (s.error || "unknown error"));
              } else if (s.status === "cancelled") {
                setStatus("Cancelled.");
              } else {
                setStatus("Idle.");
              }
              /* Final refetch + re-render guarantees the just-completed
                 block reflects its finalized state (status=complete,
                 finished_at populated, delete button enabled). */
              loadResults();
            }
          })
          .catch(function () { setTimeout(pollStatus, POLL_MS * 2); });
      }

      function loadResults() {
        return fetch("/api/discovery/results")
          .then(function (r) { return r.json(); })
          .then(function (data) {
            STATE.runs = (data && data.runs) || [];
            renderStack();
          })
          .catch(function () {});
      }

      formEl.addEventListener("submit", function (e) {
        e.preventDefault();
        var trials = parseInt(trialsEl.value || "200", 10);
        var start  = startEl.value;
        var end    = endEl.value;
        var seed   = seedEl.value ? parseInt(seedEl.value, 10) : null;
        if (!start || !end) {
          setStatus("Pick both start and end dates.");
          return;
        }
        runBtn.disabled = true;
        setStatus("Starting discovery run…");
        setProgress(0);
        var settings = getDiscoverySettings();
        var body = {
          trials: trials, start: start, end: end,
          instrument:    settings.instrument,
          interval:      settings.interval,
          direction:     settings.direction,
          blocked_hours: settings.blocked_hours,
          min_pf:        settings.min_pf,
          min_trades:    settings.min_trades,
          max_dd_pct:    settings.max_dd_pct,
        };
        if (seed !== null && !isNaN(seed)) body.seed = seed;
        fetch("/api/discovery/run", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(body)
        })
          .then(function (r) { return r.json().then(function (j) { return {ok: r.ok, body: j}; }); })
          .then(function (resp) {
            if (!resp.ok || !resp.body.ok) {
              setStatus("Failed: " + ((resp.body && resp.body.error) || "unknown error"));
              runBtn.disabled = false;
              return;
            }
            STATE.polling = true;
            pollStatus();
          })
          .catch(function () {
            setStatus("Request failed.");
            runBtn.disabled = false;
          });
      });

      /* ── Discovery Settings: editable fields + localStorage persistence ──
         The fields' values are persisted to localStorage so they survive
         reload, and getDiscoverySettings() is called inside the form
         submit handler above so the values land in every /api/discovery/run
         payload — each trial then honors them via discovery.py's config. */
      var DISCO_DEFAULTS = {
        instrument:    "GBPUSD",
        interval:      "5m",
        direction:     "short_only",
        blocked_hours: "4,5,6,8,10,11,14,17",
        min_pf:        "1.5",
        min_trades:    "50",
        max_dd_pct:    "15",
      };
      function loadDS(key) {
        try {
          var s = window.localStorage.getItem("disco_" + key);
          return s !== null ? s : DISCO_DEFAULTS[key];
        } catch (e) { return DISCO_DEFAULTS[key]; }
      }
      function saveDS(key, val) {
        try { window.localStorage.setItem("disco_" + key, val); } catch (e) {}
      }
      function saveBlockedHours() {
        var blocked = [];
        for (var h = 0; h <= 23; h++) {
          var cb = document.getElementById("ds-bh-" + h);
          if (cb && cb.checked) blocked.push(h);
        }
        saveDS("blocked_hours", blocked.join(","));
      }
      function initDiscoverySettings() {
        ["instrument", "interval", "direction"].forEach(function (k) {
          var el = document.getElementById("ds-" + k);
          if (!el) return;
          el.value = loadDS(k);
          el.addEventListener("change", function () { saveDS(k, el.value); });
        });
        [["min-pf", "min_pf"], ["min-trades", "min_trades"], ["max-dd-pct", "max_dd_pct"]].forEach(function (p) {
          var el = document.getElementById("ds-" + p[0]);
          if (!el) return;
          el.value = loadDS(p[1]);
          el.addEventListener("change", function () { saveDS(p[1], el.value); });
        });
        var grid = document.getElementById("ds-blocked-hours-grid");
        if (grid) {
          var csv = loadDS("blocked_hours");
          var blockedSet = {};
          csv.split(",").forEach(function (h) {
            var n = parseInt((h || "").trim(), 10);
            if (!isNaN(n)) blockedSet[n] = true;
          });
          grid.innerHTML = "";
          for (var h = 0; h <= 23; h++) {
            var cb = document.createElement("input");
            cb.type = "checkbox";
            cb.className = "bs-hour-cb";
            cb.id = "ds-bh-" + h;
            cb.value = String(h);
            cb.checked = !!blockedSet[h];
            cb.addEventListener("change", saveBlockedHours);
            grid.appendChild(cb);
            var label = document.createElement("label");
            label.className = "bs-hour-label";
            label.setAttribute("for", "ds-bh-" + h);
            label.textContent = String(h);
            grid.appendChild(label);
          }
        }
      }
      /* Called from the form submit handler to fold current values into
         the /api/discovery/run payload. Reads live from the DOM so any
         in-flight unsaved typing is captured too. */
      function getDiscoverySettings() {
        var blocked = [];
        for (var h = 0; h <= 23; h++) {
          var cb = document.getElementById("ds-bh-" + h);
          if (cb && cb.checked) blocked.push(h);
        }
        function v(id, dflt) {
          var el = document.getElementById(id);
          return (el && el.value !== "") ? el.value : dflt;
        }
        return {
          instrument:    v("ds-instrument", DISCO_DEFAULTS.instrument),
          interval:      v("ds-interval",   DISCO_DEFAULTS.interval),
          direction:     v("ds-direction",  DISCO_DEFAULTS.direction),
          blocked_hours: blocked.join(","),
          min_pf:        parseFloat(v("ds-min-pf",     DISCO_DEFAULTS.min_pf)),
          min_trades:    parseInt(v("ds-min-trades",   DISCO_DEFAULTS.min_trades), 10),
          max_dd_pct:    parseFloat(v("ds-max-dd-pct", DISCO_DEFAULTS.max_dd_pct)),
        };
      }

      /* ── On load: hydrate from existing results + resume polling if running ── */
      initDiscoverySettings();
      loadResults();
      fetch("/api/discovery/status")
        .then(function (r) { return r.json(); })
        .then(function (s) {
          if (s.running) {
            runBtn.disabled = true;
            STATE.polling = true;
            pollStatus();
          }
        })
        .catch(function () {});
    })();
  </script>
</body>
</html>"""


@app.route("/discovery")
def discovery_page():
    """Render the Discovery page (Phase 1 random search UI)."""
    return Response(_DISCOVERY_PAGE_HTML, mimetype="text/html")


@app.route("/discovery/trial/<trial_id>")
def discovery_trial_detail(trial_id):
    """Per-trial detail page. URL uses the trial's globally-unique id
    (e.g. 't1_bad89744') NOT its trial number — trial numbers are scoped
    to a single run, so two runs each have a 'trial 1' and routing by
    number returned the wrong report when multiple runs existed. The id
    is uuid-suffixed so collisions are vanishingly unlikely.

    Verifies the trial exists in any of the persisted runs before serving
    discovery_trial.html — that way a typo'd URL gets a real 404 instead
    of a working page that fails to render. The page itself fetches the
    trial data client-side via /api/discovery/trial/<id>."""
    found = False
    for run in _read_discovery_runs():
        if any(t.get("id") == trial_id for t in (run.get("trials") or [])):
            found = True
            break
    if not found:
        abort(404)
    template = BASE_DIR / "discovery_trial.html"
    if not template.exists():
        abort(500)
    return Response(template.read_text(encoding="utf-8"), mimetype="text/html")


# ── /results — file server + directory listing ───────────────────────────────
# Exposes everything under the project's `results/` folder. Supports nested
# paths (e.g. /results/regime_charts/2026-01-15.png) so the regime labeler
# report, generated charts, versioned PNG snapshots, etc. are all reachable
# from a single base URL.

def _fmt_bytes(n):
    """Compact human-readable file size."""
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


@app.route("/results/<path:filename>")
def serve_results_file(filename):
    """Serve any file beneath results/, including nested subdirectories.

    send_from_directory rejects paths that escape the directory root, so this
    is safe against path-traversal attacks (e.g. /results/../server.py) even
    when `filename` comes straight from the URL.

    For HTML files (e.g. regime_analysis.html, regime_discovery.html) we
    explicitly disable browser caching. Those files embed inline JS that we
    edit frequently during development; without no-store the browser will
    happily serve a stale cached copy and the user sees "the bug is back"
    even after we've fixed it. PNG/CSV/etc. assets keep their normal cache
    behavior since they don't carry behaviour-changing code.
    """
    if not RESULTS_DIR.exists():
        abort(404)
    try:
        resp = send_from_directory(str(RESULTS_DIR), filename, conditional=True)
        if filename.endswith(".html"):
            resp.headers["Cache-Control"] = "no-store, must-revalidate"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
        return resp
    except FileNotFoundError:
        abort(404)


@app.route("/results")
@app.route("/results/")
def list_results():
    """Plain HTML directory listing of every file under results/.

    Walks the directory recursively so files in subfolders
    (e.g. results/regime_charts/*.png) appear alongside top-level files,
    each linked to its /results/<rel> URL.
    """
    if not RESULTS_DIR.exists():
        body = (
            "<div class='results-listing-container'>"
            "<h1>results/ — directory not found</h1>"
            "<p>The <code>results/</code> folder does not exist yet. "
            "Run a backtest or the regime labeler to populate it.</p>"
            "</div>"
        )
        return Response(_results_page(body), mimetype="text/html")

    entries = []
    for root, _dirs, files in os.walk(RESULTS_DIR):
        for f in files:
            full = Path(root) / f
            try:
                rel = full.relative_to(RESULTS_DIR).as_posix()
                st  = full.stat()
            except (OSError, ValueError):
                continue
            entries.append({
                "rel":   rel,
                "size":  st.st_size,
                "mtime": datetime.fromtimestamp(st.st_mtime),
            })
    entries.sort(key=lambda e: e["rel"])

    rows = []
    for e in entries:
        rows.append(
            "<tr>"
            f"<td><a href='/results/{e['rel']}'>{e['rel']}</a></td>"
            f"<td>{_fmt_bytes(e['size'])}</td>"
            f"<td>{e['mtime'].strftime('%Y-%m-%d %H:%M')}</td>"
            "</tr>"
        )
    table_body = "".join(rows) or (
        "<tr><td colspan='3' class='regime-dim'>No files in results/.</td></tr>"
    )
    body = f"""
      <div class='results-listing-container'>
        <h1>results/ — {len(entries)} file{'s' if len(entries) != 1 else ''}</h1>
        <p>Files under <code>{RESULTS_DIR}</code>. Click any path to view.</p>
        <table class='results-listing-table'>
          <thead><tr><th>Path</th><th>Size</th><th>Modified</th></tr></thead>
          <tbody>{table_body}</tbody>
        </table>
      </div>
    """
    return Response(_results_page(body), mimetype="text/html")


def _results_page(body_html):
    """Wrap a body fragment in a minimal page that pulls in style.css."""
    return f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <title>Results — directory listing</title>
  <link rel='stylesheet' href='/style.css'>
</head>
<body class='results-listing'>{body_html}</body>
</html>"""


# ── /run_regime_analysis — interactive endpoint for the regime labeler page ──
#
# Accepts a JSON POST with start_date, end_date, allowed_macro_regimes (display
# names or internal keys), and allowed_micro_regimes (internal keys). Filters
# trades from the cached data via strategy_v2's existing entry-time gates,
# rebuilds the four dynamic sections the regime_analysis.html page swaps in
# place (stats bar, macro perf, regime perf, timeline, daily breakdown), and
# returns them as JSON HTML fragments.
#
# This endpoint does NOT re-run the full regime labeler — it relies on the
# already-computed data/regime_labels.parquet for the per-fractal labels and
# the schema-metadata macro_by_date mapping.

# ── FUTURE WORK — RA backtest caching ─────────────────────────────────────
# The /run_regime_analysis endpoint runs the full backtest (~100s for a
# 15-month range) on every call. An earlier iteration cached the
# unfiltered backtest result (run with empty regime allow-lists) and
# re-filtered per request, which made toggle changes near-instant. BUT
# the daily-loss-limit (2 losses/day) interacts with regime gating in a
# way that produces materially different trade counts when the gate is
# inactive vs active: signals on locked-regime days consume the daily
# budget before would-be allowed signals can fire. Result: post-filtered
# counts were 4× smaller than the gate-active equivalent.
#
# To re-enable caching correctly, run_backtest needs to expose its raw
# signal stream BEFORE the daily-loss-limit is applied. The RA endpoint
# can then re-simulate the DLL per requested toggle state on the cached
# signals (fast). That refactor is non-trivial and out of scope for now;
# tracked as a future engineering task. Until then, every Run Analysis
# click triggers the full backtest with the user's toggle state as
# active gates — exact numbers, ~100s per click.

@app.route("/run_regime_analysis", methods=["POST"])
def run_regime_analysis():
    import time as _time
    t0 = _time.time()
    try:
        payload = request.get_json(force=True) or {}
        start_date    = payload.get("start_date")
        end_date      = payload.get("end_date")
        allowed_macro = list(payload.get("allowed_macro_regimes", []))
        allowed_micro = list(payload.get("allowed_micro_regimes", []))
        # Instrument is sent by the RA page from its run-bar selection
        # (sourced from localStorage `rb_instrument`). Default to GBPUSD,
        # which is the historical labeler default and what the parquet on
        # disk most likely contains.
        instrument = str(payload.get("instrument") or "GBPUSD").strip().upper() or "GBPUSD"
        use_ema_filter = bool(payload.get("use_ema_filter", True))
        if not start_date or not end_date:
            return jsonify({"error": "start_date and end_date are required"}), 400

        # Lazy-import so server.py can boot without pyarrow/pandas if the
        # regime feature isn't used. Importing strategy_v2 also triggers its
        # module-level regime-labels load. Set INSTRUMENT in the env BEFORE
        # importing strategy_v2 so any module-init paths that read it (e.g.
        # MASSIVE_TICKER resolution) see the right value.
        os.environ["INSTRUMENT"] = instrument
        import pandas as pd  # noqa: F401  — re-export used below
        import strategy_v2 as strat
        import regime_analysis as rl
        # Re-pin the strategy module's instrument globals so a backtest
        # called from a previous request with a different INSTRUMENT
        # doesn't bleed through (strategy_v2 caches these at import time).
        strat.TICKER          = instrument
        strat._INSTRUMENT     = instrument
        strat.MASSIVE_TICKER  = strat._INSTRUMENT_MAP.get(instrument, strat.MASSIVE_TICKER)

        # (regime_labels.parquet reload moved into the cache-miss branch —
        # only needed when we're about to call run_backtest, since the in-
        # backtest macro/micro gates read those strategy module globals.
        # On cache hit we don't run a backtest at all.)

        # Force a re-read of data/regime_labels.parquet. strategy_v2 normally
        # loads labels once at import time and caches them; if the parquet
        # didn't exist (or pyarrow/fastparquet wasn't importable) at server
        # boot, the in-backtest macro/micro gates would silently pass-through
        # forever and toggle state on the RA page would have no effect on the
        # rerun. Resetting the cache flag here lets a freshly-generated
        # parquet take effect on the very next RA request.
        strat._REGIME_LABELS_LOADED = False
        strat._load_regime_labels()

        # ── Override strategy_v2 module globals with this request's filters ──
        # Active gates ON: signals on locked-regime days are rejected at
        # decision time. This makes trade counts + daily-loss-limit
        # consumption faithful to the toggle state at the cost of re-running
        # the full backtest on every click. See the "FUTURE WORK" block at
        # the top of this section for the caching path that would make
        # toggle changes near-instant once run_backtest exposes pre-DLL
        # signals.
        strat.ALLOWED_MACRO_KEYS = {strat._macro_key(n) for n in allowed_macro}
        strat.ALLOWED_MICRO_KEYS = {strat._micro_key(n) for n in allowed_micro}

        # Persist the toggle state into the ACTIVE version's regime_state so
        # subsequent BD backtests on this version pick up the same allow-lists,
        # and switching versions never overwrites another version's settings.
        _write_active_regime_state(
            sorted(strat.ALLOWED_MACRO_KEYS),
            sorted(strat.ALLOWED_MICRO_KEYS),
        )
        strat.USE_EMA_FILTER = use_ema_filter

        # ── Apply the active version's full backtest params ──────────────
        # Bug fix (May 2026 — Task 1): previously this endpoint only
        # overrode the regime gates + USE_EMA_FILTER, so the RA backtest
        # ran with strategy_v2's hardcoded defaults (EMA_LONG=40, SL=15,
        # RRR 1:2, MAX_DLL=2, BLOCKED_HOURS=4,5,...) — diverging from
        # BD/Discovery numbers for any Discovery-assigned version with
        # different params (v3, v4, v5, v6, ...).
        # Fix: layer the active version's params block onto the strategy
        # module's globals before run_backtest. Mirrors what
        # _apply_active_version_to_env does for the BD subprocess path —
        # except this is in-process so we mutate strat directly.
        av = _get_active_version()
        av_params = (av or {}).get("params") or {}
        if av_params:
            if av_params.get("ema_long") is not None:
                strat.EMA_LONG = int(av_params["ema_long"])
            if av_params.get("fractal_stop_pips") is not None:
                # strategy_v2 stores FRACTAL_STOP_PIPS as a price (pips/10000).
                strat.FRACTAL_STOP_PIPS = float(av_params["fractal_stop_pips"]) / 10000
            # rrr_risk is implicitly 1 for Phase 1 Discovery; set explicitly
            # so a previous request with a non-1 rrr_risk doesn't bleed.
            strat.RRR_RISK = int(av_params.get("rrr_risk") or 1)
            if av_params.get("rrr_reward") is not None:
                strat.RRR_REWARD = int(av_params["rrr_reward"])
            # Bug fix (May 2026 — Task 1, part 3): strategy_v2 derives `RRR`
            # ONCE at module import as `float(RRR_REWARD) / float(RRR_RISK)`
            # and run_backtest reads `RRR` (not RRR_REWARD/RRR_RISK) when
            # computing take-profits. Mutating RRR_REWARD/RRR_RISK alone
            # leaves the cached RRR pointing at the import-time value (e.g.
            # 2.0 when server.py booted with no env var), so RA's TPs land
            # at 2× stop distance even when the version says RRR_REWARD=1
            # — materially different trade outcomes (Discovery 75 trades
            # vs RA 45 trades for v6's RRR 1:1). Recompute RRR here so it
            # tracks the freshly-overridden numerator/denominator.
            strat.RRR = float(strat.RRR_REWARD) / float(strat.RRR_RISK)
            if av_params.get("max_daily_losses") is not None:
                strat.MAX_DAILY_LOSSES = int(av_params["max_daily_losses"])
            if av_params.get("trade_direction"):
                strat.TRADE_DIRECTION = str(av_params["trade_direction"])
            bh_csv = av_params.get("blocked_hours")
            if bh_csv:
                try:
                    strat.BLOCKED_HOURS_UTC = [int(h.strip()) for h in str(bh_csv).split(",") if h.strip()]
                except (TypeError, ValueError):
                    pass
            # use_ema_filter from the version overrides the payload's
            # default unless the payload explicitly sent one. The RA page
            # currently doesn't expose this toggle, so the version's
            # stored value should win.
            if av_params.get("use_ema_filter") is not None and "use_ema_filter" not in payload:
                strat.USE_EMA_FILTER = bool(av_params["use_ema_filter"])

        # ── Run the backtest ──
        df = strat.fetch_data(strat.TICKER, strat.INTERVAL, strat.DAYS_BACK,
                              start_date=start_date, end_date=end_date)
        df = strat.add_indicators(df)

        # Bug fix (May 2026 — Task 1, part 2): match strategy_v2's __main__
        # date-range trim BEFORE run_backtest. fetch_data returns a wide
        # window ([start-30d, end+7d]) so indicators warm up; __main__ then
        # trims to [start-1d, end+7d] before run_backtest so only ~1 day of
        # pre-start fractal history feeds the "prior fractal" comparisons.
        # Without this trim, RA's backtest used 30 extra days of prior
        # fractals → different higher-low / lower-high reference points →
        # materially different signals (45 trades vs Discovery's 75 for the
        # same params). Replicating the trim here makes RA produce the same
        # trade list as BD + Discovery.
        start_ts = pd.Timestamp(start_date, tz="UTC")
        end_ts   = pd.Timestamp(end_date,   tz="UTC") + pd.Timedelta(days=1)
        _bt_start = start_ts - pd.Timedelta(days=1)
        _bt_end   = end_ts   + pd.Timedelta(days=7)
        try:
            _dts = pd.to_datetime(df["Datetime"])
            _dts_utc = (_dts.dt.tz_convert("UTC")
                        if _dts.dt.tz is not None
                        else _dts.dt.tz_localize("UTC"))
            df = df[(_dts_utc >= _bt_start) & (_dts_utc < _bt_end)].reset_index(drop=True)
        except KeyError:
            # df has no "Datetime" column — older fetch_data versions used the
            # index. Fall back to index-based trim.
            try:
                df = df[(df.index >= _bt_start) & (df.index < _bt_end)].copy()
            except Exception:
                pass  # leave df as-is rather than break the request

        trades, equity, raw_blocked = strat.run_backtest(df)

        # Trim trades to the requested date range (entries only)
        if not trades.empty:
            _t = pd.to_datetime(trades["entry_ts"])
            _t = _t.dt.tz_convert("UTC") if _t.dt.tz is not None else _t.dt.tz_localize("UTC")
            trades = trades[(_t >= start_ts) & (_t < end_ts)].reset_index(drop=True)

        # ── Load fractal labels + macro from parquet ──
        labels_path = BASE_DIR / "data" / "regime_labels.parquet"
        macro = {}
        if labels_path.exists():
            try:
                import pyarrow.parquet as _pq
                tbl = _pq.read_table(str(labels_path))
                meta = tbl.schema.metadata or {}
                blob = meta.get(b"regime_analysis") or meta.get(b"regime_labeler")
                if blob:
                    payload_meta = json.loads(blob.decode("utf-8"))
                    for d, lbl in (payload_meta.get("macro_by_date") or {}).items():
                        macro[d] = {"label": lbl, "details": {}}
                fractal_df = tbl.to_pandas()
            except ImportError:
                from fastparquet import ParquetFile as _PF
                _pf = _PF(str(labels_path))
                kv = _pf.key_value_metadata or {}
                blob = kv.get("regime_analysis") or kv.get("regime_labeler")
                if isinstance(blob, bytes): blob = blob.decode("utf-8")
                if blob:
                    payload_meta = json.loads(blob)
                    for d, lbl in (payload_meta.get("macro_by_date") or {}).items():
                        macro[d] = {"label": lbl, "details": {}}
                fractal_df = _pf.to_pandas()
        else:
            return jsonify({"error": "data/regime_labels.parquet not found — "
                                    "run regime_analysis.py first"}), 400

        # Normalise fractal timestamps + filter to range
        _fts = pd.to_datetime(fractal_df["timestamp"])
        _fts = _fts.dt.tz_convert("UTC") if _fts.dt.tz is not None else _fts.dt.tz_localize("UTC")
        fractal_df = fractal_df.copy()
        fractal_df["timestamp"] = _fts
        in_range = fractal_df[(fractal_df["timestamp"] >= start_ts)
                              & (fractal_df["timestamp"] < end_ts)].copy()

        # ── Reconstruct periods (consecutive same-regime fractals) ──
        periods = []
        cur = None
        cl = in_range["regime"].values
        ts = in_range["timestamp"].tolist()
        fb = in_range["fractal_bar"].values
        idx = in_range.index.values
        for i in range(len(in_range)):
            label_i = cl[i]
            if cur is None or label_i != cur["label"]:
                if cur is not None:
                    periods.append(cur)
                cur = {
                    "label":   label_i,
                    "regime":  label_i,
                    "start_idx": idx[i],
                    "end_idx":   idx[i],
                    "start_ts":  ts[i],
                    "end_ts":    ts[i],
                    "start_bar": int(fb[i]),
                    "end_bar":   int(fb[i]),
                    "fractal_idxs": [idx[i]],
                }
            else:
                cur["end_idx"]    = idx[i]
                cur["end_ts"]     = ts[i]
                cur["end_bar"]    = int(fb[i])
                cur["fractal_idxs"].append(idx[i])
        if cur is not None:
            periods.append(cur)

        # Regime-period counts (used by the perf table)
        regime_count = {}
        for p in periods:
            regime_count[p["regime"]] = regime_count.get(p["regime"], 0) + 1

        # Derive blocked keys (complement of allowed in the universe)
        allowed_macro_keys = strat.ALLOWED_MACRO_KEYS
        allowed_micro_keys = strat.ALLOWED_MICRO_KEYS
        all_macro_keys = set(rl.MACRO_REGIME_ORDER)
        all_micro_keys = set(rl.REGIME_ORDER)
        blocked_macro_keys = all_macro_keys - allowed_macro_keys
        blocked_micro_keys = all_micro_keys - allowed_micro_keys

        # Per-fractal micro asof series for attribution
        if not in_range.empty:
            _frac_ts = pd.to_datetime(in_range["timestamp"])
            _frac_ts = _frac_ts.dt.tz_convert("UTC") if _frac_ts.dt.tz is not None else _frac_ts.dt.tz_localize("UTC")
            micro_asof = (
                pd.Series(in_range["regime"].values, index=_frac_ts).dropna().sort_index()
            )
        else:
            micro_asof = pd.Series([], dtype="object")

        def _attribute(df_):
            if df_.empty:
                df_ = df_.copy()
                df_["regime"] = pd.Series([], dtype="object")
                df_["macro_label"] = pd.Series([], dtype="object")
                return df_
            ts = pd.to_datetime(df_["entry_ts"])
            ts = ts.dt.tz_convert("UTC") if ts.dt.tz is not None else ts.dt.tz_localize("UTC")
            df_ = df_.copy()
            if not micro_asof.empty:
                df_["regime"] = [micro_asof.asof(t) for t in ts]
            else:
                df_["regime"] = None
            df_["macro_label"] = ts.dt.strftime("%Y-%m-%d").map(
                lambda d: (macro.get(d) or {}).get("label")
            ).values
            return df_

        # Attribute trades + blocked signals
        trades = _attribute(trades)
        if raw_blocked:
            blocked_df = pd.DataFrame(raw_blocked).rename(columns={"timestamp": "entry_ts"})
            if "entry_ts" in blocked_df.columns:
                _bts = pd.to_datetime(blocked_df["entry_ts"])
                _bts = _bts.dt.tz_convert("UTC") if _bts.dt.tz is not None else _bts.dt.tz_localize("UTC")
                blocked_df = blocked_df[(_bts >= start_ts) & (_bts < end_ts)].reset_index(drop=True)
        else:
            blocked_df = pd.DataFrame(columns=["entry_ts", "win", "pnl", "reason", "direction"])
        blocked_df = _attribute(blocked_df)

        # Daily-breakdown helpers
        fractals_per_day = {}
        if not in_range.empty:
            for d in in_range["timestamp"].dt.strftime("%Y-%m-%d"):
                fractals_per_day[d] = fractals_per_day.get(d, 0) + 1
        df_dts = pd.to_datetime(df["Datetime"])
        df_dts = df_dts.dt.tz_convert("UTC") if df_dts.dt.tz is not None else df_dts.dt.tz_localize("UTC")
        df_in_rng = df[(df_dts >= start_ts) & (df_dts < end_ts)]
        df_in_rng_dts = pd.to_datetime(df_in_rng["Datetime"])
        df_in_rng_dts = df_in_rng_dts.dt.tz_convert("UTC") if df_in_rng_dts.dt.tz is not None else df_in_rng_dts.dt.tz_localize("UTC")
        trading_days_all = sorted(set(df_in_rng_dts.dt.strftime("%Y-%m-%d")))
        low_activity_days = {
            d for d in trading_days_all
            if fractals_per_day.get(d, 0) < rl.LOW_ACTIVITY_FRACTAL_THRESHOLD
        }

        # ── Server-side authoritative filter ──────────────────────────────
        # Apply the toggle state as a post-filter on the trades coming out of
        # run_backtest. The in-backtest macro/micro gates *should* already
        # have rejected these signals (so they'd be in raw_blocked instead of
        # trades), but we don't rely on that for two reasons:
        #
        #   1. If strategy_v2 was imported before data/regime_labels.parquet
        #      existed, its label dicts are empty and the in-backtest gates
        #      silently pass-through. We reload labels above, but a single
        #      defensive post-filter ensures the displayed numbers honor the
        #      toggle state even if anything upstream silently no-ops.
        #   2. It localises the source-of-truth: every section of the report
        #      below sees the same filtered trade set, so the stats bar,
        #      per-regime tables, and daily timeline can never disagree about
        #      which trades count as "active".
        #
        # We filter on the attributed `macro_label` / `regime` columns set by
        # `_attribute` above (re-keyed off this request's freshly-loaded
        # parquet), so attribution and filtering are guaranteed consistent.
        #
        # Empty allow-list = filter inactive (pass everything through). This
        # mirrors strategy_v2's in-backtest gates — `_check_macro_regime` /
        # `_check_micro_regime` both `return True, ""` when their allowed-set
        # is empty — and is critical for v1, which has no regime gates at
        # all (both allow-lists empty by default). Treating empty as
        # "block everything" instead would make v1 always return zero
        # trades from the RA, which is what just happened.
        def _filter_trades(td):
            """Return (filtered, n_macro_excluded, n_micro_excluded)."""
            if td.empty:
                return td.copy(), 0, 0
            keep = pd.Series([True] * len(td), index=td.index)
            n_macro = 0
            n_micro = 0
            if allowed_macro_keys and blocked_macro_keys and "macro_label" in td.columns:
                macro_mask = td["macro_label"].isin(blocked_macro_keys)
                keep &= ~macro_mask
                n_macro = int(macro_mask.sum())
            if allowed_micro_keys and blocked_micro_keys and "regime" in td.columns:
                micro_mask = td["regime"].isin(blocked_micro_keys)
                keep &= ~micro_mask
                n_micro = int(micro_mask.sum())
            return td[keep].reset_index(drop=True), n_macro, n_micro

        filtered_trades, n_macro_excluded, n_micro_excluded = _filter_trades(trades)
        # n_excluded is macro-only for the macro-filter note (compute_filter_label
        # is built around the macro filter); the micro count is for diagnostics.
        n_excluded = n_macro_excluded
        perf_df = rl._compute_perf_df(filtered_trades)
        agg_stats = rl._compute_aggregate_stats(filtered_trades)
        total_in_range_trades = int(len(trades))

        filter_state_label, filter_state_class, macro_filter_note, macro_table_filter_note = \
            rl.compute_filter_label(blocked_macro_keys, total_in_range_trades, n_excluded)

        # ── Build HTML chunks ──
        # Every section below is fed `filtered_trades` so the stats bar,
        # per-regime perf tables, daily timeline, and breakdown are all
        # derived from the same authoritative trade set. The per-regime
        # tables still get `blocked_df` separately for their locked-row
        # counterfactual stats — that path is unchanged.
        stats_bar_inner = rl.build_stats_bar_html(
            agg_stats, filter_state_label, filter_state_class)

        macro_perf_table = rl.build_macro_perf_table(
            macro, filtered_trades, blocked_macro_keys=blocked_macro_keys,
            blocked_signals_df=blocked_df)

        perf_table = rl.build_perf_table_html(
            perf_df, regime_count,
            blocked_micro_keys=blocked_micro_keys,
            trades_df=filtered_trades,
            blocked_signals_df=blocked_df,
            allowed_macro_keys=allowed_macro_keys,
        )

        trades_per_day = rl.compute_trades_per_day(filtered_trades)
        timeline_inner = rl.build_timeline_section_html(
            periods, macro, trades_per_day, start_date, end_date, regime_count)

        # Daily performance — needs `df_in_rng` and the precomputed
        # `trading_days_all` / `low_activity_days` / `fractals_per_day`
        # from the cache (or just-computed at miss time above).
        # `available_chart_days` is cheap (glob a directory) so we always
        # compute it per-request — keeps the cache from being invalidated
        # when the user generates new per-day charts between requests.
        available_chart_days = set()
        chart_dir = BASE_DIR / "results" / "regime_charts"
        if chart_dir.exists():
            for png in chart_dir.glob("*.png"):
                available_chart_days.add(png.stem)

        # Temporarily override START_DATE/END_DATE on the rl module so its
        # _trading_days_in_range helper (called inside build_daily_breakdown)
        # uses this request's range.
        _orig_start, _orig_end = rl.START_DATE, rl.END_DATE
        rl.START_DATE = start_date
        rl.END_DATE   = end_date
        try:
            daily_table_html = rl.build_daily_breakdown(
                periods, filtered_trades, df, available_chart_days,
                in_range, low_activity_days,
                macro=macro, blocked_macro_keys=blocked_macro_keys,
                trading_days=trading_days_all,
            )
        finally:
            rl.START_DATE = _orig_start
            rl.END_DATE   = _orig_end

        # ── Daily performance section wrap (header + table + note) ──
        daily_section_inner = f"""
          <h2>Daily performance</h2>
          {daily_table_html}
          <p class="regime-dim regime-small regime-breakdown-note">
            <span class="regime-hour-chip regime-color-inactive regime-hour-chip--inline"></span>
            Dark chips indicate hours where <strong>no fractal was detected</strong>.
            &nbsp;&nbsp;
            <span class="regime-low-activity-dot regime-low-activity-dot--inline"></span>
            Indicates a low-activity day (fewer than {rl.LOW_ACTIVITY_FRACTAL_THRESHOLD}
            fractals across the whole 24-hour period).
          </p>
        """

        # Section wrappers (full inner HTML each section needs)
        macro_perf_inner = f"""
          <h2>Macro regime performance <span class="regime-dim regime-small">(daily context)</span></h2>
          <p class="regime-dim regime-small">
            Day-level performance by overall daily character — answers whether
            the strategy should be trading on certain types of days at all.
          </p>
          {macro_perf_table}
        """
        # Header suffix — reflects the active version so a v1 page doesn't
        # advertise "v2 short-only". For v2 (which is short-only with regime
        # gates), keep the direction qualifier; for v1 (no regime gates,
        # no narrow direction default) just show the version name.
        _av = _get_active_version() or {}
        _av_name = _av.get("name") or "v2"
        _av_strat = (_av.get("strategy_version") or "v2").strip()
        if _av_strat == "v2":
            _perf_suffix = f"({_av_name} short-only)"
        else:
            _perf_suffix = f"({_av_name})"
        regime_perf_inner = f"""
          <h2>Micro regime performance <span class="regime-dim regime-small">{_perf_suffix}</span></h2>
          {perf_table}
        """

        elapsed_ms = (_time.time() - t0) * 1000
        summary = (
            f"{agg_stats['total']} trades · WR "
            f"{(agg_stats['win_rate']):.1f}% · "
            f"PF " + ("∞" if agg_stats['pf'] == float('inf') else f"{agg_stats['pf']:.2f}")
        )
        return jsonify({
            "stats_bar":   stats_bar_inner,
            "macro_perf":  macro_perf_inner,
            "regime_perf": regime_perf_inner,
            "timeline":    timeline_inner,
            "daily":       daily_section_inner,
            "summary":     summary,
            "elapsed_ms":  elapsed_ms,
        })
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


def _run_backtest_sync(env_overrides=None):
    """Run strategy.py synchronously. Returns dict with ok, no_data, error."""
    import time as _time
    stdout_lines = []
    try:
        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)
        proc = subprocess.Popen(
            [sys.executable, "-u", str(STRATEGY_FILE)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(BASE_DIR),
            env=env,
        )
        _deadline = _time.time() + 300   # 5-minute safety timeout
        while True:
            line = proc.stdout.readline()
            if line:
                print(line, end="", flush=True)
                stdout_lines.append(line)
                if line.startswith("PROGRESS:"):
                    parts = line.strip().split(":", 2)
                    if len(parts) >= 3:
                        try:
                            pct = int(parts[1])
                            stage = parts[2]
                            with _bt_lock:
                                _bt_state["progress"] = pct
                                _bt_state["stage"] = stage
                        except ValueError:
                            pass
            elif proc.poll() is not None:
                break
            if _time.time() > _deadline:
                proc.kill()
                proc.wait()
                return {"ok": False, "no_data": False, "error": "Timed out after 5 minutes"}
        full_output = "".join(stdout_lines)
        if proc.returncode == 0:
            if "NO_DATA" in full_output:
                return {"ok": True, "no_data": True, "error": None}
            return {"ok": True, "no_data": False, "error": None}
        else:
            err = full_output.strip()
            return {"ok": False, "no_data": False, "error": err[-800:] if err else "Non-zero exit code"}
    except Exception as exc:
        return {"ok": False, "no_data": False, "error": str(exc)}


def _backtest_worker(env_overrides=None):
    """Run strategy.py in a background thread and update _bt_state when done."""
    result = _run_backtest_sync(env_overrides)
    with _bt_lock:
        _bt_state["ok"]      = result["ok"]
        _bt_state["no_data"] = result.get("no_data", False)
        _bt_state["error"]   = result.get("error")
        _bt_state["running"] = False
        _bt_state["stage"]   = ""


def _get_best_worst_months(version_name):
    """Read report.html, find the monthly data for version_name, return best & worst months."""
    import calendar
    from datetime import date
    if not REPORT_FILE.exists():
        return None, None
    html = REPORT_FILE.read_text(encoding="utf-8")
    match = re.search(
        r'(<script[^>]+id=["\']versions-data["\'][^>]*>)([\s\S]*?)(</script>)',
        html
    )
    if not match:
        return None, None
    try:
        versions = json.loads(match.group(2).strip())
    except (json.JSONDecodeError, ValueError):
        return None, None
    # Find the version
    target = None
    for v in versions:
        if v.get("name") == version_name:
            target = v
            break
    if not target:
        return None, None
    # Get monthly data from the first run (the full version run)
    runs = target.get("runs", [])
    if not runs:
        return None, None
    monthly = runs[0].get("metrics", {}).get("monthly", [])
    if len(monthly) < 2:
        return None, None
    # Find best and worst by net_pnl
    best  = max(monthly, key=lambda m: m.get("net_pnl", 0))
    worst = min(monthly, key=lambda m: m.get("net_pnl", 0))
    # Convert period string "2025-03" to date range
    def month_to_range(period_str):
        parts = period_str.split("-")
        y, m = int(parts[0]), int(parts[1])
        first = date(y, m, 1)
        last_day = calendar.monthrange(y, m)[1]
        last = date(y, m, last_day)
        return first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d")
    return month_to_range(best["month"]), month_to_range(worst["month"])


def _version_with_auto_ranges(env_overrides):
    """Run new version backtest, then auto-add best & worst month date ranges."""
    # Step 1: Run the version backtest
    with _bt_lock:
        _bt_state["stage"] = "Running version backtest\u2026"
    result = _run_backtest_sync(env_overrides)
    if not result["ok"]:
        with _bt_lock:
            _bt_state.update(result)
            _bt_state["running"] = False
            _bt_state["stage"]   = ""
        return

    # Done
    with _bt_lock:
        _bt_state["ok"]      = True
        _bt_state["no_data"] = False
        _bt_state["error"]   = None
        _bt_state["running"] = False
        _bt_state["stage"]   = ""


@app.route("/run", methods=["POST"])
def run_backtest():
    """Start strategy.py as a new version run (730 days, version incremented)."""
    with _bt_lock:
        if _bt_state["running"]:
            return jsonify({"ok": False, "error": "A backtest is already running"})
        _bt_state["running"] = True
        _bt_state["ok"]      = None
        _bt_state["error"]   = None
        _bt_state["no_data"] = False
        _bt_state["stage"]   = ""
        _bt_state["progress"] = 0

    # RUN_MODE=new_version tells strategy.py to increment version
    data = request.get_json(force=True) or {}
    instrument = (data.get("instrument") or "").strip()
    direction  = (data.get("direction") or "").strip()
    interval   = (data.get("interval") or "").strip()
    ema_short   = (data.get("ema_short") or "").strip()
    ema_mid     = (data.get("ema_mid") or "").strip()
    ema_long    = (data.get("ema_long") or "").strip()
    stop_pips   = (data.get("stop_loss_pips") or "").strip()
    rrr_risk    = (data.get("rrr_risk") or "").strip()
    rrr_reward  = (data.get("rrr_reward") or "").strip()
    blocked_hours = (data.get("blocked_hours") or "").strip()
    max_daily_losses = (data.get("max_daily_losses") or "").strip()
    apply_slippage   = (data.get("apply_slippage") or "").strip()
    spread_pips      = (data.get("spread_pips") or "").strip()
    sl_slippage_pips = (data.get("sl_slippage_pips") or "").strip()
    use_ema_filter   = data.get("use_ema_filter")
    strategy_version = (data.get("strategy_version") or "").strip()
    env_overrides = {"RUN_MODE": "new_version"}
    if strategy_version:
        env_overrides["STRATEGY_VERSION"] = strategy_version
    if instrument:
        env_overrides["INSTRUMENT"] = instrument
    if direction:
        env_overrides["TRADE_DIRECTION"] = direction
    if interval:
        env_overrides["INTERVAL"] = interval
    if ema_short:
        env_overrides["EMA_SHORT"] = ema_short
    if ema_mid:
        env_overrides["EMA_MID"] = ema_mid
    if ema_long:
        env_overrides["EMA_LONG"] = ema_long
    if stop_pips:
        env_overrides["FRACTAL_STOP_PIPS"] = stop_pips
    if rrr_risk:
        env_overrides["RRR_RISK"] = rrr_risk
    if rrr_reward:
        env_overrides["RRR_REWARD"] = rrr_reward
    env_overrides["BLOCKED_HOURS_UTC"] = blocked_hours if blocked_hours else ""
    if max_daily_losses:
        env_overrides["MAX_DAILY_LOSSES"] = max_daily_losses
    if apply_slippage:
        env_overrides["APPLY_SLIPPAGE"] = apply_slippage
    if spread_pips:
        env_overrides["SPREAD_PIPS"] = spread_pips
    if sl_slippage_pips:
        env_overrides["SL_SLIPPAGE_PIPS"] = sl_slippage_pips
    # Bug fix (May 2026): explicit use_ema_filter in the BD payload now
    # wins over the version's params.use_ema_filter. Previously the BD
    # didn't send this field at all, so the version's value was always
    # used — which broke when the dropdown was decoupled from the global
    # active version (the strategy ran with the wrong version's filter
    # state). Now the BD always sends what its checkbox shows.
    if isinstance(use_ema_filter, bool):
        env_overrides["USE_EMA_FILTER"] = "true" if use_ema_filter else "false"
    elif isinstance(use_ema_filter, str) and use_ema_filter.strip():
        env_overrides["USE_EMA_FILTER"] = use_ema_filter.strip()

    # Layer in the SELECTED version's params + regime allow-lists.
    # Selection comes from payload.version_id (or payload.strategy_version
    # when it's a version id) — falls back to global active.
    _apply_active_version_to_env(env_overrides, data)

    t = threading.Thread(
        target=_version_with_auto_ranges,
        args=(env_overrides,),
        daemon=True,
    )
    t.start()
    return jsonify({"ok": True, "started": True})


@app.route("/run_range", methods=["POST"])
def run_date_range():
    """Start strategy.py as a date-range iteration on the current version."""
    data = request.get_json(force=True) or {}
    start_date = (data.get("start_date") or "").strip()
    end_date   = (data.get("end_date")   or "").strip()

    if not start_date or not end_date:
        return jsonify({"ok": False, "error": "Start and end dates are required"})

    with _bt_lock:
        if _bt_state["running"]:
            return jsonify({"ok": False, "error": "A backtest is already running"})
        _bt_state["running"] = True
        _bt_state["ok"]      = None
        _bt_state["error"]   = None
        _bt_state["no_data"] = False
        _bt_state["stage"]   = ""
        _bt_state["progress"] = 0

    instrument     = (data.get("instrument") or "").strip()
    target_version = (data.get("target_version") or "").strip()
    direction      = (data.get("direction") or "").strip()
    interval       = (data.get("interval") or "").strip()
    ema_short      = (data.get("ema_short") or "").strip()
    ema_mid        = (data.get("ema_mid") or "").strip()
    ema_long       = (data.get("ema_long") or "").strip()
    stop_pips      = (data.get("stop_loss_pips") or "").strip()
    rrr_risk       = (data.get("rrr_risk") or "").strip()
    rrr_reward     = (data.get("rrr_reward") or "").strip()
    blocked_hours  = (data.get("blocked_hours") or "").strip()
    max_daily_losses = (data.get("max_daily_losses") or "").strip()
    apply_slippage   = (data.get("apply_slippage") or "").strip()
    spread_pips      = (data.get("spread_pips") or "").strip()
    sl_slippage_pips = (data.get("sl_slippage_pips") or "").strip()
    use_ema_filter   = data.get("use_ema_filter")
    strategy_version = (data.get("strategy_version") or "").strip()
    env_overrides = {
        "RUN_MODE":       "date_range",
        "RUN_START_DATE": start_date,
        "RUN_END_DATE":   end_date,
    }
    if strategy_version:
        env_overrides["STRATEGY_VERSION"] = strategy_version
    if instrument:
        env_overrides["INSTRUMENT"] = instrument
    if target_version:
        env_overrides["TARGET_VERSION"] = target_version
    if direction:
        env_overrides["TRADE_DIRECTION"] = direction
    if interval:
        env_overrides["INTERVAL"] = interval
    if ema_short:
        env_overrides["EMA_SHORT"] = ema_short
    if ema_mid:
        env_overrides["EMA_MID"] = ema_mid
    if ema_long:
        env_overrides["EMA_LONG"] = ema_long
    if stop_pips:
        env_overrides["FRACTAL_STOP_PIPS"] = stop_pips
    if rrr_risk:
        env_overrides["RRR_RISK"] = rrr_risk
    if rrr_reward:
        env_overrides["RRR_REWARD"] = rrr_reward
    env_overrides["BLOCKED_HOURS_UTC"] = blocked_hours if blocked_hours else ""
    if max_daily_losses:
        env_overrides["MAX_DAILY_LOSSES"] = max_daily_losses
    if apply_slippage:
        env_overrides["APPLY_SLIPPAGE"] = apply_slippage
    if spread_pips:
        env_overrides["SPREAD_PIPS"] = spread_pips
    if sl_slippage_pips:
        env_overrides["SL_SLIPPAGE_PIPS"] = sl_slippage_pips
    # See /run for context. Explicit USE_EMA_FILTER from payload wins over
    # the version's params.use_ema_filter so the BD checkbox is the user-
    # visible source of truth.
    if isinstance(use_ema_filter, bool):
        env_overrides["USE_EMA_FILTER"] = "true" if use_ema_filter else "false"
    elif isinstance(use_ema_filter, str) and use_ema_filter.strip():
        env_overrides["USE_EMA_FILTER"] = use_ema_filter.strip()

    # Layer in the SELECTED version's params + regime allow-lists.
    # Selection comes from payload.version_id (or payload.strategy_version
    # when it's a version id) — falls back to global active.
    _apply_active_version_to_env(env_overrides, data)

    t = threading.Thread(
        target=_backtest_worker,
        args=(env_overrides,),
        daemon=True,
    )
    t.start()
    return jsonify({"ok": True, "started": True})


def _batch_worker(ranges, shared_params):
    """Run multiple date-range backtests sequentially in a single thread."""
    total = len(ranges)
    # Keys whose empty-string value is semantically meaningful \u2014 strategy_v2
    # distinguishes 'unset' (use default) from 'empty string' (disable gate)
    # for these, so they must pass through even when ''.
    _PASS_EMPTY = {"ALLOWED_MACRO_REGIMES", "ALLOWED_MICRO_REGIMES"}
    for idx, rng in enumerate(ranges):
        with _bt_lock:
            _bt_state["stage"] = "Running date range %d of %d\u2026" % (idx + 1, total)
            _bt_state["progress"] = 0
        env_overrides = {
            "RUN_MODE":       "date_range",
            "RUN_START_DATE": rng["start"],
            "RUN_END_DATE":   rng["end"],
        }
        for key, val in shared_params.items():
            if val or key in _PASS_EMPTY:
                env_overrides[key] = val
        result = _run_backtest_sync(env_overrides)
        if not result["ok"]:
            with _bt_lock:
                _bt_state["ok"]      = False
                _bt_state["error"]   = result.get("error", "Batch run failed on range %d" % (idx + 1))
                _bt_state["running"] = False
                _bt_state["stage"]   = ""
            return
    # All ranges completed successfully
    with _bt_lock:
        _bt_state["ok"]      = True
        _bt_state["no_data"] = False
        _bt_state["error"]   = None
        _bt_state["running"] = False
        _bt_state["stage"]   = ""


@app.route("/run_batch", methods=["POST"])
def run_batch():
    """Run multiple date-range backtests sequentially."""
    data = request.get_json(force=True) or {}
    ranges = data.get("ranges", [])
    if not ranges or not isinstance(ranges, list):
        return jsonify({"ok": False, "error": "No date ranges provided"})
    # Validate all ranges have start and end
    for rng in ranges:
        if not rng.get("start") or not rng.get("end"):
            return jsonify({"ok": False, "error": "Each range must have start and end dates"})

    with _bt_lock:
        if _bt_state["running"]:
            return jsonify({"ok": False, "error": "A backtest is already running"})
        _bt_state["running"] = True
        _bt_state["ok"]      = None
        _bt_state["error"]   = None
        _bt_state["no_data"] = False
        _bt_state["stage"]   = ""
        _bt_state["progress"] = 0

    # Build shared params dict (same for all ranges)
    shared_params = {}
    strategy_version = (data.get("strategy_version") or "").strip()
    instrument       = (data.get("instrument") or "").strip()
    target_version   = (data.get("target_version") or "").strip()
    direction        = (data.get("direction") or "").strip()
    interval         = (data.get("interval") or "").strip()
    ema_short        = (data.get("ema_short") or "").strip()
    ema_mid          = (data.get("ema_mid") or "").strip()
    ema_long         = (data.get("ema_long") or "").strip()
    stop_pips        = (data.get("stop_loss_pips") or "").strip()
    rrr_risk         = (data.get("rrr_risk") or "").strip()
    rrr_reward       = (data.get("rrr_reward") or "").strip()
    blocked_hours    = (data.get("blocked_hours") or "").strip()
    max_daily_losses = (data.get("max_daily_losses") or "").strip()
    apply_slippage   = (data.get("apply_slippage") or "").strip()
    spread_pips      = (data.get("spread_pips") or "").strip()
    sl_slippage_pips = (data.get("sl_slippage_pips") or "").strip()
    if strategy_version: shared_params["STRATEGY_VERSION"] = strategy_version
    if instrument:       shared_params["INSTRUMENT"]       = instrument
    if target_version:   shared_params["TARGET_VERSION"]   = target_version
    if direction:        shared_params["TRADE_DIRECTION"]  = direction
    if interval:         shared_params["INTERVAL"]         = interval
    if ema_short:        shared_params["EMA_SHORT"]        = ema_short
    if ema_mid:          shared_params["EMA_MID"]          = ema_mid
    if ema_long:         shared_params["EMA_LONG"]         = ema_long
    if stop_pips:        shared_params["FRACTAL_STOP_PIPS"] = stop_pips
    if rrr_risk:         shared_params["RRR_RISK"]         = rrr_risk
    if rrr_reward:       shared_params["RRR_REWARD"]       = rrr_reward
    shared_params["BLOCKED_HOURS_UTC"] = blocked_hours if blocked_hours else ""
    if max_daily_losses: shared_params["MAX_DAILY_LOSSES"] = max_daily_losses
    if apply_slippage:   shared_params["APPLY_SLIPPAGE"]   = apply_slippage
    if spread_pips:      shared_params["SPREAD_PIPS"]      = spread_pips
    if sl_slippage_pips: shared_params["SL_SLIPPAGE_PIPS"] = sl_slippage_pips
    use_ema_filter = data.get("use_ema_filter")
    if isinstance(use_ema_filter, bool):
        shared_params["USE_EMA_FILTER"] = "true" if use_ema_filter else "false"
    elif isinstance(use_ema_filter, str) and use_ema_filter.strip():
        shared_params["USE_EMA_FILTER"] = use_ema_filter.strip()

    # Layer in the SELECTED version's params + regime allow-lists.
    # Selection comes from payload.version_id (or payload.strategy_version
    # when it's a version id) — falls back to global active.
    # The batch worker propagates shared_params into each range's env, so
    # injecting here means every range in the batch inherits the same filters.
    _apply_active_version_to_env(shared_params, data)

    t = threading.Thread(
        target=_batch_worker,
        args=(ranges, shared_params),
        daemon=True,
    )
    t.start()
    return jsonify({"ok": True, "started": True})


@app.route("/status")
def backtest_status():
    """Return the current backtest state for the browser to poll."""
    with _bt_lock:
        return jsonify(dict(_bt_state))


# ── Versions API — strategy profiles for /versions page + BD selector ─────────

@app.route("/api/versions", methods=["GET"])
def api_versions_list():
    """Return the full versions store including active_version_id.
    Each version is stamped with a `run_count` field (count of BD runs
    in report.html for that version's name) so the Versions page can
    decide whether to surface the run-history-deletion warning before
    issuing a delete."""
    store = _read_versions()
    counts = _count_runs_per_version_name()
    for v in store.get("versions", []):
        v["run_count"] = counts.get(v.get("name", ""), 0)
    return jsonify(store)


@app.route("/api/versions", methods=["POST"])
def api_versions_add():
    """Create a blank unassigned version slot. No request body needed.
    Auto-named v<N+1>. params/regime_state/strategy_version all start as
    null — Discovery's assign flow fills them in a single one-time write.
    Until then the slot is invisible in BD + RA dropdowns."""
    new_version = _add_version()
    return jsonify({"ok": True, "version": new_version})


@app.route("/api/versions/<version_id>", methods=["DELETE"])
def api_versions_delete(version_id):
    """Delete a version. Refuses if it's the last one; auto-switches active
    if the deleted one was active.

    May 2026: also auto-renumbers the remaining versions to v1..vN in
    creation order (atomically rewrites versions.json + report.html
    buckets) and returns the rename map in `rename_map` so the client can
    rewrite localStorage discovery_trial_assignments accordingly."""
    ok, payload = _delete_version(version_id)
    if not ok:
        return jsonify({"ok": False, "error": payload}), 400
    return jsonify({
        "ok": True,
        "store": _read_versions(),
        "rename_map": payload or {},
    })


@app.route("/api/versions/<version_id>/notes", methods=["POST"])
def api_versions_set_notes(version_id):
    """Task 4: persist a per-version free-form notes field.

    Body: {notes: "..."}. Stored under the version's `notes` key in
    versions.json so it survives server restarts and is visible to any
    page that reads the versions store. Together with each version's
    own run-history bucket this replaces the manually-maintained
    RESULTS_LOG.md (per-version notable-result notes) and devlog.json
    (free-form dev log) — those files can now be retired.
    """
    body = request.get_json(force=True, silent=True) or {}
    notes = body.get("notes", "")
    if not isinstance(notes, str):
        return jsonify({"ok": False, "error": "notes must be a string"}), 400
    data = _read_versions()
    found = False
    for v in data.get("versions", []):
        if v.get("id") == version_id:
            v["notes"] = notes
            v["notes_updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            found = True
            break
    if not found:
        return jsonify({"ok": False, "error": "unknown version id"}), 404
    _write_versions(data)
    return jsonify({"ok": True, "store": data})


@app.route("/api/active_version", methods=["GET"])
def api_active_version_get():
    """Return the active version dict (id, name, strategy_version, regime_state)."""
    av = _get_active_version()
    if av is None:
        return jsonify({"ok": False, "error": "no versions configured"}), 500
    return jsonify({"ok": True, "active": av})


@app.route("/api/active_version", methods=["POST"])
def api_active_version_set():
    """Switch the active version. Body: {id}."""
    body = request.get_json(force=True, silent=True) or {}
    version_id = (body.get("id") or "").strip()
    if not version_id:
        return jsonify({"ok": False, "error": "id is required"}), 400
    if not _set_active_version(version_id):
        return jsonify({"ok": False, "error": "unknown version id"}), 404
    return jsonify({"ok": True, "active": _get_active_version()})


# ── Discovery — Phase 1 random parameter search ──────────────────────────────
# Distinct from the BD backtest flow:
#   - The BD spawns ONE strategy.py subprocess per "Run" click and writes a
#     full report.html. Discovery spawns N (default 200) subprocesses each
#     with DISCOVERY_MODE=1 — strategy_v2.py short-circuits its report-writing
#     side effects and dumps a slim metrics JSON instead.
#   - discovery.py owns the subprocess loop and writes incrementally to
#     data/discovery_results.json, so the UI can poll status while the run
#     is in flight without us having to thread per-trial progress back here.
#   - Subprocess concurrency: only one discovery run at a time; we keep the
#     Popen in _disco_state so /api/discovery/run can refuse concurrent
#     starts and so the page can show "still running" on reload.

DISCOVERY_SCRIPT       = BASE_DIR / "discovery.py"
DISCOVERY_RESULTS_FILE = DATA_DIR / "discovery_results.json"

_disco_lock  = threading.Lock()
_disco_state = {"process": None, "run_id": None, "started_at": None}


def _discovery_is_running():
    """True iff the discovery subprocess is still alive."""
    proc = _disco_state["process"]
    if proc is None:
        return False
    return proc.poll() is None


def _read_discovery_runs():
    """Load discovery_results.json as an array of runs (newest first).
    Swallows read errors because the file is written atomically by
    discovery.py — a transient error usually means the swap is mid-flight.

    Transparently migrates the legacy single-object schema (one run dict
    at the top level) to a single-element array. Doesn't persist the
    migration here — discovery.py's next init_results_file will rewrite
    the file in the new format. Both shapes return correct semantics in
    the meantime."""
    if not DISCOVERY_RESULTS_FILE.exists():
        return []
    try:
        with open(DISCOVERY_RESULTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and data.get("run_id"):
        return [data]
    return []


def _atomic_write_discovery_runs(runs):
    """Write the runs array atomically. Used by the DELETE endpoint;
    discovery.py owns its own writes (which also use atomic rename)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = DISCOVERY_RESULTS_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)
    tmp.replace(DISCOVERY_RESULTS_FILE)


def _latest_discovery_run():
    """Return the most-recent run dict (first in the array), or None."""
    runs = _read_discovery_runs()
    return runs[0] if runs else None


def _find_trial_across_runs(value, key="id"):
    """Search every run for a trial matching trial[key] == value. Returns
    (trial_dict, parent_run_dict) or (None, None). Trial IDs ('t1_<hex>')
    are uuid-suffixed so they're unique across runs; trial numbers are not
    — but we never search by trial number across runs (only within a run),
    so this helper is only used with key='id'."""
    for run in _read_discovery_runs():
        for t in (run.get("trials") or []):
            if t.get(key) == value:
                return t, run
    return None, None


@app.route("/api/discovery/run", methods=["POST"])
def api_discovery_run():
    """Launch discovery.py as a subprocess. Body accepts:
      trials, start, end, seed (existing)
      instrument, interval, direction, blocked_hours (editable fixed-constants)
      min_pf, min_trades, max_dd_pct (editable passing criteria)
    Anything missing falls back to discovery.py's hardcoded defaults.
    Refuses if a discovery run is already in flight."""
    body = request.get_json(force=True, silent=True) or {}
    trials = body.get("trials", 200)
    start  = (body.get("start") or "2025-07-01").strip()
    end    = (body.get("end")   or "2025-12-31").strip()
    seed   = body.get("seed")

    try:
        trials = int(trials)
        if trials < 1 or trials > 10000:
            return jsonify({"ok": False, "error": "trials must be between 1 and 10000"}), 400
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "trials must be an integer"}), 400

    # Basic date-string validation
    for label, val in (("start", start), ("end", end)):
        try:
            datetime.strptime(val, "%Y-%m-%d")
        except ValueError:
            return jsonify({"ok": False, "error": f"{label} must be YYYY-MM-DD"}), 400

    # ── Editable Discovery Settings ──────────────────────────────────────
    # Validation: enumerate allowed values for instrument/interval/direction;
    # blocked_hours is a CSV of integers 0–23; thresholds are numbers in
    # sane ranges. Any invalid input → 400 with a clear error message.
    _ALLOWED_INSTRUMENTS = {"GBPUSD", "EURUSD"}
    _ALLOWED_INTERVALS   = {"1m", "5m", "15m", "1h"}
    _ALLOWED_DIRECTIONS  = {"short_only", "long_only", "both"}

    instrument = (body.get("instrument") or "").strip().upper()
    if instrument and instrument not in _ALLOWED_INSTRUMENTS:
        return jsonify({"ok": False, "error": f"instrument must be one of {sorted(_ALLOWED_INSTRUMENTS)}"}), 400
    interval = (body.get("interval") or "").strip()
    if interval and interval not in _ALLOWED_INTERVALS:
        return jsonify({"ok": False, "error": f"interval must be one of {sorted(_ALLOWED_INTERVALS)}"}), 400
    direction = (body.get("direction") or "").strip()
    if direction and direction not in _ALLOWED_DIRECTIONS:
        return jsonify({"ok": False, "error": f"direction must be one of {sorted(_ALLOWED_DIRECTIONS)}"}), 400

    blocked_hours = (body.get("blocked_hours") or "").strip()
    if blocked_hours:
        try:
            _hours = [int(h.strip()) for h in blocked_hours.split(",") if h.strip()]
            if any(h < 0 or h > 23 for h in _hours):
                raise ValueError("hours must be 0–23")
            blocked_hours = ",".join(str(h) for h in _hours)
        except (TypeError, ValueError) as e:
            return jsonify({"ok": False, "error": f"blocked_hours must be a comma-separated list of integers 0–23 ({e})"}), 400

    def _parse_num(key, lo, hi, kind):
        raw = body.get(key)
        if raw is None or raw == "":
            return None
        try:
            v = kind(raw)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": f"{key} must be a number"}), 400
        if v < lo or v > hi:
            return jsonify({"ok": False, "error": f"{key} must be between {lo} and {hi}"}), 400
        return v

    min_pf      = _parse_num("min_pf",     0.0,  10.0,  float)
    min_trades  = _parse_num("min_trades", 0,    10000, int)
    max_dd_pct  = _parse_num("max_dd_pct", 0.0,  100.0, float)
    # _parse_num returns a (jsonify(...), 400) tuple on validation failure
    # (not a Response object). Detect the tuple and propagate it as the
    # endpoint's error response. Bug fix (May 2026): previously this used
    # hasattr(v, "status_code") which a tuple doesn't have — out-of-range
    # values fell through and triggered a 500 when json.dump tried to
    # serialise the tuple into the config file.
    for v in (min_pf, min_trades, max_dd_pct):
        if isinstance(v, tuple):
            return v

    with _disco_lock:
        if _discovery_is_running():
            return jsonify({"ok": False, "error": "A discovery run is already in progress"}), 409

        # Write a fresh config JSON the subprocess will pick up via --config-json
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        config_path = DATA_DIR / ".discovery_config.json"
        cfg = {"trials": trials, "start": start, "end": end,
               "results_file": str(DISCOVERY_RESULTS_FILE)}
        if seed is not None:
            try:
                cfg["seed"] = int(seed)
            except (TypeError, ValueError):
                pass
        # Layer in the editable settings — only set keys that were actually
        # provided in the body so discovery.py falls back to its defaults
        # when callers (e.g. CLI --once mode) omit them.
        if instrument:    cfg["instrument"]    = instrument
        if interval:      cfg["interval"]      = interval
        if direction:     cfg["direction"]     = direction
        if blocked_hours: cfg["blocked_hours"] = blocked_hours
        if min_pf is not None:     cfg["min_pf"]     = min_pf
        if min_trades is not None: cfg["min_trades"] = min_trades
        if max_dd_pct is not None: cfg["max_dd_pct"] = max_dd_pct
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)

        # Launch the subprocess detached enough that the Flask request returns
        # immediately. discovery.py owns its own progress writes to
        # discovery_results.json; we just hold the Popen for is-running checks.
        run_id = "discovery_" + datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        proc = subprocess.Popen(
            [sys.executable, str(DISCOVERY_SCRIPT), "--config-json", str(config_path)],
            cwd=str(BASE_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _disco_state["process"]    = proc
        _disco_state["run_id"]     = run_id
        _disco_state["started_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    return jsonify({"ok": True, "started": True, "run_id": run_id})


@app.route("/api/discovery/status", methods=["GET"])
def api_discovery_status():
    """Lightweight progress poll. Reads the LATEST run from the array for
    trial counts + best-so-far, plus the in-memory subprocess state for
    is-running. Never returns the full trials array — use
    /api/discovery/results for that."""
    running = _discovery_is_running()
    latest  = _latest_discovery_run() or {}

    # If the file says running but the process is gone, the subprocess died.
    # Don't mutate the file from here — let the next /run overwrite it; just
    # report a corrected status in the response.
    status = latest.get("status", "idle")
    if status == "running" and not running:
        status = "error"

    return jsonify({
        "ok":              True,
        "running":         running,
        "status":          status,
        "run_id":          latest.get("run_id"),
        "started_at":      latest.get("started_at"),
        "finished_at":     latest.get("finished_at"),
        "trials_complete": latest.get("trials_complete", 0),
        "trials_total":    latest.get("trials_total", 0),
        "best":            latest.get("best"),
        "config":          latest.get("config"),
        "error":           latest.get("error"),
    })


@app.route("/api/discovery/results", methods=["GET"])
def api_discovery_results():
    """Return the full discovery runs array (newest first). Each element
    carries its own metadata (run_id, started_at, finished_at, config,
    status, trials_complete, trials_total, best) plus the full trials array.
    Used by the Discovery page on load to render the stack of per-run blocks."""
    runs = _read_discovery_runs()
    return jsonify({"ok": True, "empty": len(runs) == 0, "runs": runs})


@app.route("/api/discovery/results/<run_id>", methods=["DELETE"])
def api_discovery_delete_run(run_id):
    """Remove a single run from the array by run_id. Refuses to delete
    the run that's currently in progress — it would leave the discovery.py
    subprocess writing into a phantom slot that the next progress poll
    would silently re-prepend. The user must wait for the run to finish
    (or kill the subprocess) before deleting it."""
    runs = _read_discovery_runs()
    if _discovery_is_running() and runs and runs[0].get("run_id") == run_id:
        return jsonify({"ok": False, "error": "Cannot delete the run that's currently in progress"}), 409
    remaining = [r for r in runs if r.get("run_id") != run_id]
    if len(remaining) == len(runs):
        return jsonify({"ok": False, "error": f"run_id '{run_id}' not found"}), 404
    _atomic_write_discovery_runs(remaining)
    return jsonify({"ok": True, "removed": run_id, "remaining": len(remaining)})


@app.route("/api/discovery/trial/<trial_id>", methods=["GET"])
def api_discovery_trial(trial_id):
    """Return a single discovery trial by its globally-unique id (e.g.
    't1_bad89744'), plus the run config it belongs to. Bug fix (May 2026):
    previously this routed by the trial number, which is only unique
    within a single run — so /api/discovery/trial/1 returned whichever
    run held the FIRST 'trial 1' (newest), even when the user clicked a
    different run's trial 1 in the stack. Routing by id eliminates the
    ambiguity. 404 if not found."""
    for run in _read_discovery_runs():
        for t in (run.get("trials") or []):
            if t.get("id") == trial_id:
                return jsonify({"ok": True, "trial": t, "config": run.get("config"), "run_id": run.get("run_id")})
    return jsonify({"ok": False, "error": f"trial id '{trial_id}' not found"}), 404


@app.route("/api/discovery/assign", methods=["POST"])
def api_discovery_assign():
    """Assign a discovery trial's params into an EXISTING unassigned
    version slot. Body: {result_id, version_id}.

    This is a one-time write — there is no update endpoint. To re-assign
    a different trial the user must create a new (unassigned) version on
    the Versions page first. Refuses if the target version already has
    params set.

    The trial's params dict is mapped to the new slim params schema:
      stop_loss_pips      → fractal_stop_pips    (rename)
      ema_long            → ema_long
      use_ema_filter      → use_ema_filter
      rrr_reward          → rrr_reward
      max_daily_losses    → max_daily_losses
      (rrr_risk + blocked_hours are NOT stored on the version)
    plus an assigned_at ISO timestamp.

    regime_state is populated from the trial's allowed_macro_regimes +
    allowed_micro_regimes. strategy_version is stamped 'v2' (Discovery's
    fixed base; Phase 1)."""
    body = request.get_json(force=True, silent=True) or {}
    result_id  = (body.get("result_id")  or "").strip()
    version_id = (body.get("version_id") or "").strip()
    if not result_id:
        return jsonify({"ok": False, "error": "result_id is required"}), 400
    if not version_id:
        return jsonify({"ok": False, "error": "version_id is required"}), 400

    # Trial IDs are uuid-suffixed ('t<n>_<hex>') so unique across all runs.
    trial, _run = _find_trial_across_runs(result_id, key="id")
    if trial is None:
        return jsonify({"ok": False, "error": f"trial id '{result_id}' not found"}), 404

    ok, payload = _assign_version_from_trial(
        version_id,
        trial.get("params") or {},
        run_config=(_run or {}).get("config") or {},
    )
    if not ok:
        return jsonify({"ok": False, "error": payload}), 400
    return jsonify({"ok": True, "version": payload})


@app.route("/delete_version", methods=["POST"])
def delete_version():
    """Remove a version from report.html and RESULTS_LOG.md."""
    try:
        data = request.get_json(force=True)
        name = (data.get("name") or "").strip()
        if not name:
            return jsonify({"ok": False, "error": "No version name provided"})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)})

    # ── Remove from report.html ────────────────────────────────────────────────
    if not REPORT_FILE.exists():
        return jsonify({"ok": False, "error": "report.html not found"})

    html = REPORT_FILE.read_text(encoding="utf-8")
    match = re.search(
        r'(<script[^>]+id=["\']versions-data["\'][^>]*>)([\s\S]*?)(</script>)',
        html
    )
    if not match:
        return jsonify({"ok": False, "error": "Could not parse versions data in report.html"})

    try:
        versions = json.loads(match.group(2).strip())
    except (json.JSONDecodeError, ValueError) as exc:
        return jsonify({"ok": False, "error": f"JSON parse error: {exc}"})

    original_count = len(versions)
    versions = [v for v in versions if v.get("name") != name]
    if len(versions) == original_count:
        return jsonify({"ok": False, "error": f"Version '{name}' not found"})

    new_json = json.dumps(versions, indent=2, ensure_ascii=False)
    new_html = html[:match.start(2)] + "\n" + new_json + "\n" + html[match.end(2):]
    REPORT_FILE.write_text(new_html, encoding="utf-8")

    # ── Remove from RESULTS_LOG.md ─────────────────────────────────────────────
    results_log = BASE_DIR / "RESULTS_LOG.md"
    if results_log.exists():
        lines = results_log.read_text(encoding="utf-8").splitlines(keepends=True)
        new_lines = [l for l in lines if not re.match(r'^\|\s*' + re.escape(name) + r'\s*\|', l)]
        results_log.write_text("".join(new_lines), encoding="utf-8")

    # ── Delete version files from results/ folder ─────────────────────────────
    results_dir = BASE_DIR / "results"
    if results_dir.is_dir():
        for f in results_dir.iterdir():
            if f.name.startswith(name + "_") or f.name.startswith(name + "."):
                try:
                    f.unlink()
                except OSError:
                    pass

    return jsonify({"ok": True})


@app.route("/delete_run", methods=["POST"])
def delete_run():
    """Remove a single run (date-range iteration) from a version in report.html."""
    try:
        data = request.get_json(force=True)
        name     = (data.get("name") or "").strip()
        run_idx  = data.get("run_idx")
        if not name or run_idx is None:
            return jsonify({"ok": False, "error": "Version name and run_idx are required"})
        run_idx = int(run_idx)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)})

    if not REPORT_FILE.exists():
        return jsonify({"ok": False, "error": "report.html not found"})

    html = REPORT_FILE.read_text(encoding="utf-8")
    match = re.search(
        r'(<script[^>]+id=["\']versions-data["\'][^>]*>)([\s\S]*?)(</script>)',
        html
    )
    if not match:
        return jsonify({"ok": False, "error": "Could not parse versions data in report.html"})

    try:
        versions = json.loads(match.group(2).strip())
    except (json.JSONDecodeError, ValueError) as exc:
        return jsonify({"ok": False, "error": f"JSON parse error: {exc}"})

    # Find the version
    target = None
    for v in versions:
        if v.get("name") == name:
            target = v
            break
    if target is None:
        return jsonify({"ok": False, "error": f"Version '{name}' not found"})

    runs = target.get("runs", [])
    if not runs or run_idx < 0 or run_idx >= len(runs):
        return jsonify({"ok": False, "error": f"Run index {run_idx} out of range"})

    # Don't allow deleting the last remaining run — that's a full version delete
    if len(runs) <= 1:
        return jsonify({"ok": False, "error": "Cannot delete the only run; use Delete Version instead"})

    runs.pop(run_idx)

    new_json = json.dumps(versions, indent=2, ensure_ascii=False)
    new_html = html[:match.start(2)] + "\n" + new_json + "\n" + html[match.end(2):]
    REPORT_FILE.write_text(new_html, encoding="utf-8")

    return jsonify({"ok": True})


@app.route("/reorder_runs", methods=["POST"])
def reorder_runs():
    """Reorder the runs array for a version in report.html."""
    try:
        data = request.get_json(force=True)
        name      = (data.get("name") or "").strip()
        new_order = data.get("order")  # list of old indices
        if not name or not isinstance(new_order, list):
            return jsonify({"ok": False, "error": "Version name and order array are required"})
        new_order = [int(i) for i in new_order]
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)})

    if not REPORT_FILE.exists():
        return jsonify({"ok": False, "error": "report.html not found"})

    html = REPORT_FILE.read_text(encoding="utf-8")
    match = re.search(
        r'(<script[^>]+id=["\']versions-data["\'][^>]*>)([\s\S]*?)(</script>)',
        html
    )
    if not match:
        return jsonify({"ok": False, "error": "Could not parse versions data in report.html"})

    try:
        versions = json.loads(match.group(2).strip())
    except (json.JSONDecodeError, ValueError) as exc:
        return jsonify({"ok": False, "error": f"JSON parse error: {exc}"})

    target = None
    for v in versions:
        if v.get("name") == name:
            target = v
            break
    if target is None:
        return jsonify({"ok": False, "error": f"Version '{name}' not found"})

    runs = target.get("runs", [])
    if sorted(new_order) != list(range(len(runs))):
        return jsonify({"ok": False, "error": "Invalid order — must be a permutation of run indices"})

    target["runs"] = [runs[i] for i in new_order]

    new_json = json.dumps(versions, indent=2, ensure_ascii=False)
    new_html = html[:match.start(2)] + "\n" + new_json + "\n" + html[match.end(2):]
    REPORT_FILE.write_text(new_html, encoding="utf-8")

    return jsonify({"ok": True})


# ── cBot Generator ────────────────────────────────────────────────────────────

@app.route("/generate_cbot", methods=["POST"])
def generate_cbot_endpoint():
    """Generate a C# cBot (.cs) file from the current version and parameters."""
    try:
        data = request.get_json(force=True) or {}
        strategy_version = (data.get("strategy_version") or "").strip()
        if not strategy_version:
            return jsonify({"ok": False, "error": "No strategy version provided"})

        params = {
            "ema_short":        data.get("ema_short", "8"),
            "ema_mid":          data.get("ema_mid", "20"),
            "ema_long":         data.get("ema_long", "40"),
            "stop_loss_pips":   data.get("stop_loss_pips", "15"),
            "rrr_risk":         data.get("rrr_risk", "1"),
            "rrr_reward":       data.get("rrr_reward", "2"),
            "max_daily_losses": data.get("max_daily_losses", "2"),
            "trade_direction":  data.get("trade_direction", "both"),
            "blocked_hours":    data.get("blocked_hours", ""),
            "instrument":       data.get("instrument", "EURUSD"),
        }

        filename, cs_code = generate_cbot(strategy_version, params)

        return Response(
            cs_code,
            mimetype="text/plain",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "text/plain; charset=utf-8",
            },
        )
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)})
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Generation failed: {exc}"})


# ── Dev Log API (devlog.json) ─────────────────────────────────────────────────

DEVLOG_FILE = BASE_DIR / "devlog.json"

def _load_devlog():
    if DEVLOG_FILE.exists():
        try:
            return json.loads(DEVLOG_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            return []
    return []

def _save_devlog(data):
    DEVLOG_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

@app.route("/devlog", methods=["GET"])
def devlog_get():
    return jsonify(_load_devlog())

@app.route("/devlog", methods=["POST"])
def devlog_save():
    try:
        data = request.get_json(force=True)
        if not isinstance(data, list):
            return jsonify({"ok": False, "error": "Expected a JSON array"})
        _save_devlog(data)
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("  Trading Bot Dashboard")
    print("  ────────────────────────────────────────")
    print("  Open  →  http://localhost:8080")
    print("  Stop  →  Ctrl+C")
    print()
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
