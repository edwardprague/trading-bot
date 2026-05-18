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
    """Return the next sequential 'vN' name based on existing entries.
    Scans the current versions for names matching 'v<digits>', takes
    the highest N, and returns 'v<N+1>'. Defaults to 'v1' if none match.
    Ignores any non-vN names (legacy or user-edited)."""
    max_n = 0
    for v in versions or []:
        m = re.match(r"^v(\d+)$", (v.get("name") or "").strip())
        if m:
            n = int(m.group(1))
            if n > max_n:
                max_n = n
    return "v" + str(max_n + 1)


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


def _add_version(strategy_version, base_id=None):
    """Append a new version. Name is auto-generated as 'v<N+1>' based on
    the highest existing vN. If base_id is given, copy that version's
    regime_state. Otherwise default to "all active" — every macro + every
    micro key allowed. This matches the convention that brand-new versions
    have no backtest params stored (run-bar comes up empty; strategy module
    defaults kick in on the first run) but DO have explicit regime state so
    the RA page renders toggles in a clearly "all enabled" state instead of
    the confusing empty allow-list."""
    data = _read_versions()
    versions = data.get("versions", [])
    existing_ids = {v.get("id") for v in versions}
    name = _next_version_name(versions)
    new_id = _make_version_id(name, existing_ids)

    if base_id:
        regime_state = {
            "allowed_macro_regimes": list(_ALL_MACRO_KEYS),
            "allowed_micro_regimes": list(_ALL_MICRO_KEYS),
        }
        for v in versions:
            if v.get("id") == base_id:
                src = v.get("regime_state") or {}
                regime_state = {
                    "allowed_macro_regimes": list(src.get("allowed_macro_regimes", []) or []),
                    "allowed_micro_regimes": list(src.get("allowed_micro_regimes", []) or []),
                }
                break
    else:
        regime_state = {
            "allowed_macro_regimes": list(_ALL_MACRO_KEYS),
            "allowed_micro_regimes": list(_ALL_MICRO_KEYS),
        }

    new_version = {
        "id": new_id,
        "name": name,
        "strategy_version": strategy_version,
        "regime_state": regime_state,
    }
    versions.append(new_version)
    data["versions"] = versions
    _write_versions(data)
    return new_version


def _delete_version(version_id):
    """Remove a version. Refuse if it's the last remaining one. If the
    deleted version was active, fall back to the first remaining."""
    data = _read_versions()
    versions = data.get("versions", [])
    if len(versions) <= 1:
        return False, "Cannot delete the last remaining version"
    new_versions = [v for v in versions if v.get("id") != version_id]
    if len(new_versions) == len(versions):
        return False, "Version not found"
    data["versions"] = new_versions
    if data.get("active_version_id") == version_id:
        data["active_version_id"] = new_versions[0]["id"]
    _write_versions(data)
    return True, None


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


def _apply_active_version_to_env(env_overrides, payload):
    """Layer the active version's strategy_version + regime allow-lists
    into env_overrides. Precedence:

      1. Explicit values in the request payload win (with one exception
         below for STRATEGY_VERSION).
      2. Active version supplies STRATEGY_VERSION + ALLOWED_*_REGIMES.
      3. Strategy module hardcoded defaults are the final fallback.

    STRATEGY_VERSION quirk: the BD dropdown sends the option value as
    `strategy_version` in the run payload. For seeded versions that value
    is 'v1' / 'v2' (a real strategy module). For user-added profiles, the
    dropdown option value is the version id (e.g. 'v3'), which is NOT a
    strategy module — strategy_v3.py doesn't exist. We detect this and
    resolve the id through versions.json to the underlying base strategy.

    The strategy module distinguishes 'unset' from 'empty string': empty
    means 'gate disabled', unset means 'use default'. We always set
    ALLOWED_*_REGIMES when an active version exists (empty list → empty
    string), preserving that distinction."""
    av = _get_active_version()
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

# ── Backtest state (shared between the Flask thread and the worker thread) ─────
_bt_lock  = threading.Lock()
_bt_state = {"running": False, "ok": None, "error": None, "no_data": False, "stage": "", "progress": 0}

# ── Run-bar HTML (injected into every page response) ──────────────────────────

INJECT_HTML = """
<nav class="top-nav" id="top-nav">
  <ul class="top-nav-items">
    <li><a class="top-nav-link top-nav-link-active" href="/">Backtesting</a></li>
    <li><a class="top-nav-link" href="/results/regime_analysis.html">Regimes</a></li>
    <li><span class="top-nav-link top-nav-link-disabled" aria-disabled="true">Discovery</span></li>
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
  <!-- Version + instrument selects live in the run bar (not the BD sidebar)
       so they're available on every page that injects the run bar — the BD
       and the RA both surface them and stay in sync via /api/active_version
       and localStorage rb_instrument. The IDs match the originals so the
       existing handlers in report.html (onVersionChange / onInstrumentChange)
       and in this file (line ~625, /api/active_version dropdown sync) keep
       working without modification. -->
  <select id="version-select" class="rb-select" title="Active version"></select>
  <select id="instrument-select" class="rb-select" title="Instrument">
    <option value="EURUSD">EURUSD</option>
    <option value="GBPUSD">GBPUSD</option>
  </select>

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
  .rb-btn-delete { background: crimson; }
  .rb-btn-delete:hover:not(:disabled) { background: #f4254e; }

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

  fetch("/api/versions").then(function (r) { return r.json(); }).then(function (store) {
    var versions = (store && store.versions) || [];
    var activeId = store && store.active_version_id;
    var active = null;
    for (var i = 0; i < versions.length; i++) {
      if (versions[i].id === activeId) { active = versions[i]; break; }
    }
    if (!active && versions.length) active = versions[0];

    _setActiveIndicator(active ? active.name : null);

    var sel = document.getElementById("version-select");
    if (!sel) return;

    /* Reconcile versions.json with report.html's existing dropdown options.
       report.html populates the dropdown from its embedded versions-data
       (option value = the version's short name, e.g. "v1"). versions.json
       holds the canonical display name for each version (e.g.
       "v1 — Fractal Only"). For each existing option whose value matches a
       versions.json id, rename the textContent in-place rather than
       appending a duplicate. Any versions.json entries with no matching
       option get appended at the end (e.g. user-added profiles that
       haven't been backtested yet). */
    var byId = {};
    versions.forEach(function (v) { byId[v.id] = v; });
    var reconciled = {};
    for (var j = 0; j < sel.options.length; j++) {
      var opt = sel.options[j];
      var match = byId[opt.value];
      if (match) {
        opt.textContent = match.name;
        reconciled[match.id] = true;
      }
    }
    versions.forEach(function (v) {
      if (reconciled[v.id]) return;
      var newOpt = document.createElement("option");
      newOpt.value = v.id;
      newOpt.textContent = v.name;
      sel.appendChild(newOpt);
    });

    if (active) {
      sel.value = active.id;
      /* Sync report.html's private `currentVersion` to the active version
         id. report.html attaches its onVersionChange handler inside an
         IIFE (so the function isn't a global) — but the handler is wired
         via addEventListener("change"), so dispatching a change event
         here triggers it. This makes getStrategyVersions re-filter the
         sidebar by name=active.id and re-render content for the right
         bucket. Also fires our OWN change listener above (the one that
         POSTs to /api/active_version), but that's a harmless no-op
         on initial load since the server is already on that version. */
      sel.dispatchEvent(new Event("change", {bubbles: true}));
    }

    /* Sync the active version when the user picks something. We use
       addEventListener so the page's existing onchange="selectVersion"
       (defined in report.html) keeps firing too. Lookup is by id, which
       matches both the seeded options (id == option value == strategy_version)
       and any user-added profiles (we set option value = id above). */
    sel.addEventListener("change", function () {
      var picked = byId[sel.value];
      if (!picked) return;
      fetch("/api/active_version", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({id: picked.id})
      }).then(function (r) { return r.json(); }).then(function (resp) {
        if (!resp || !resp.ok || !resp.active) return;
        _setActiveIndicator(resp.active.name);
        /* Mirror the sidebar pattern (renderSidebar in report.html, ~146481):
           keep the global version-name + display-name in sync so the run-bar
           "Add Date Range (vN)" label reflects this dropdown change without
           waiting for a separate sidebar click.
           Task 6b: use NAME first so v3 (strategy_version="v2") renders as
           "Add Date Range (v3)" — not "(v2)". */
        window._currentVersionName        = resp.active.name || "";
        window._currentVersionDisplayName = resp.active.name || resp.active.strategy_version || "";
        if (typeof updateRangeButtonLabel === "function") updateRangeButtonLabel();
      }).catch(function () {});
    });
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
  /* Use the globally-exposed name set by the dashboard's renderSidebar() */
  if (window._currentVersionName) return window._currentVersionName;
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
  /* Issue 2: the dropdown is the source of truth for the active version.
     Earlier this was reading window._currentVersionDisplayName, which
     renderSidebar overwrites with the CURRENTLY-DISPLAYED run's bucket
     name — so switching instruments (which re-runs renderSidebar with
     a different run focused) could replace the label with "v2" even
     when v3 was selected in the dropdown. Read the dropdown directly
     so the label tracks the active-version selection unconditionally. */
  var sel = document.getElementById("version-select");
  var displayName = (sel && sel.value) || window._currentVersionDisplayName || getCurrentVersionName();
  var rangeBtn = document.getElementById("run-range-btn");
  if (displayName) {
    rangeBtn.innerHTML = "&#9654;&nbsp; Add Date Range (" + displayName + ")";
  } else {
    rangeBtn.innerHTML = "&#9654;&nbsp; Add Date Range";
  }
}

function getSelectedVersion() {
  var el = document.getElementById("version-select");
  if (el) return el.value;
  return "v1";
}

function getSelectedDirection() {
  var el = document.getElementById("bs-direction-select");
  if (el) return el.value;
  var stored = localStorage.getItem("bs_direction");
  return stored || "short_only";
}

function getSelectedInstrument() {
  var el = document.getElementById("instrument-select");
  if (el) return el.value;
  var stored = localStorage.getItem("rb_instrument");
  return stored || "EURUSD";
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
    body: JSON.stringify({ mode: "new_version", instrument: instrument, direction: direction, interval: interval, strategy_version: version, ema_short: getSelectedEmaShort(), ema_mid: getSelectedEmaMid(), ema_long: getSelectedEmaLong(), stop_loss_pips: getSelectedStopPips(), rrr_risk: getSelectedRrrRisk(), rrr_reward: getSelectedRrrReward(), blocked_hours: getSelectedBlockedHours(), max_daily_losses: getSelectedMaxDD(), apply_slippage: getSelectedApplySlippage(), spread_pips: getSelectedSpreadPips(), sl_slippage_pips: getSelectedSlSlippagePips() })
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
      body: JSON.stringify({ ranges: selectedRanges, instrument: instrument, target_version: targetVersion, strategy_version: version, direction: getSelectedDirection(), interval: getSelectedInterval(), ema_short: getSelectedEmaShort(), ema_mid: getSelectedEmaMid(), ema_long: getSelectedEmaLong(), stop_loss_pips: getSelectedStopPips(), rrr_risk: getSelectedRrrRisk(), rrr_reward: getSelectedRrrReward(), blocked_hours: getSelectedBlockedHours(), max_daily_losses: getSelectedMaxDD(), apply_slippage: getSelectedApplySlippage(), spread_pips: getSelectedSpreadPips(), sl_slippage_pips: getSelectedSlSlippagePips() })
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
    body: JSON.stringify({ start_date: startDate, end_date: endDate, instrument: instrument, target_version: targetVersion, strategy_version: version, direction: getSelectedDirection(), interval: getSelectedInterval(), ema_short: getSelectedEmaShort(), ema_mid: getSelectedEmaMid(), ema_long: getSelectedEmaLong(), stop_loss_pips: getSelectedStopPips(), rrr_risk: getSelectedRrrRisk(), rrr_reward: getSelectedRrrReward(), blocked_hours: getSelectedBlockedHours(), max_daily_losses: getSelectedMaxDD(), apply_slippage: getSelectedApplySlippage(), spread_pips: getSelectedSpreadPips(), sl_slippage_pips: getSelectedSlSlippagePips() })
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
    # Inject the run-bar at the START of <body> so its elements — most
    # importantly #version-select and #instrument-select, which used to
    # live in report.html's sidebar but now live here — exist in the DOM
    # before report.html's inline scripts run during parsing. (The IIFE
    # inside INJECT_HTML that needs report.html's hidden action buttons
    # is wrapped in DOMContentLoaded for the same reason.) The run-bar is
    # fixed-positioned, so DOM order doesn't affect visual placement.
    html = html.replace("<body>", "<body>\n" + INJECT_HTML, 1)
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
      <li><span class="top-nav-link top-nav-link-disabled" aria-disabled="true">Discovery</span></li>
      <li><a class="top-nav-link top-nav-link-active" href="/versions">Versions</a></li>
    </ul>
    <span class="top-nav-active-version" id="top-nav-active-version"></span>
  </nav>

  <main class="versions-container">
    <header class="versions-header">
      <h1>Versions</h1>
      <p class="versions-subtitle">
        Strategy profiles — each version bundles a base strategy module
        and its own regime allow-list state. The active version drives
        every new backtest and the toggle state on the Regimes page.
      </p>
    </header>

    <section class="versions-add-section">
      <h2>Add a version</h2>
      <form id="versions-add-form" class="versions-add-form">
        <label class="versions-form-label" for="versions-add-strategy">Base strategy</label>
        <select id="versions-add-strategy" class="versions-form-select">
          <option value="v2">v2 (EMA + regime gates)</option>
          <option value="v1">v1 (fractal only)</option>
        </select>
        <label class="versions-form-label" for="versions-add-base">Copy regime state from</label>
        <select id="versions-add-base" class="versions-form-select">
          <option value="">— all regimes active —</option>
        </select>
        <button type="submit" class="rb-btn rb-btn-green">Add Version</button>
      </form>
      <p class="versions-form-hint">
        Name will be auto-assigned as the next available <code>v&lt;N&gt;</code>.
      </p>
      <span id="versions-form-error" class="versions-form-error"></span>
    </section>

    <section class="versions-list-section">
      <h2>Existing versions</h2>
      <ul id="versions-list" class="versions-list"></ul>
    </section>
  </main>

  <script>
    (function () {
      var listEl   = document.getElementById("versions-list");
      var formEl   = document.getElementById("versions-add-form");
      var stratEl  = document.getElementById("versions-add-strategy");
      var baseEl   = document.getElementById("versions-add-base");
      var errEl    = document.getElementById("versions-form-error");
      var navAvEl  = document.getElementById("top-nav-active-version");

      function showError(msg) { errEl.textContent = msg || ""; }

      function refreshBaseOptions(versions) {
        // Preserve the user's current selection if still valid
        var prev = baseEl.value;
        baseEl.innerHTML = "";
        var blank = document.createElement("option");
        blank.value = "";
        blank.textContent = "— blank (no regime gates) —";
        baseEl.appendChild(blank);
        versions.forEach(function (v) {
          var opt = document.createElement("option");
          opt.value = v.id;
          opt.textContent = v.name;
          baseEl.appendChild(opt);
        });
        if (prev) baseEl.value = prev;
      }

      function renderList(store) {
        var active = store.active_version_id;
        var versions = store.versions || [];
        refreshBaseOptions(versions);
        listEl.innerHTML = "";
        versions.forEach(function (v) {
          var isActive = (v.id === active);
          var li = document.createElement("li");
          li.className = "versions-row" + (isActive ? " versions-row-active" : "");

          var nameSpan = document.createElement("span");
          nameSpan.className = "versions-row-name";
          nameSpan.textContent = v.name;
          if (isActive) {
            var badge = document.createElement("span");
            badge.className = "versions-row-active-badge";
            badge.textContent = "ACTIVE";
            nameSpan.appendChild(badge);
          }

          var rs = v.regime_state || {};
          var macroCount = (rs.allowed_macro_regimes || []).length;
          var microCount = (rs.allowed_micro_regimes || []).length;
          var metaSpan = document.createElement("span");
          metaSpan.className = "versions-row-meta";
          metaSpan.textContent =
            "Strategy: " + (v.strategy_version || "—") + "   ·   " +
            "Macro allow-list: " + macroCount + "   ·   " +
            "Micro allow-list: " + microCount;

          var actionsSpan = document.createElement("span");
          actionsSpan.className = "versions-row-actions";
          var delBtn = document.createElement("button");
          delBtn.type = "button";
          delBtn.className = "rb-btn rb-btn-delete";
          delBtn.textContent = "Delete";
          if (versions.length <= 1) delBtn.disabled = true;
          delBtn.addEventListener("click", function () { deleteVersion(v); });
          actionsSpan.appendChild(delBtn);

          // Task 4: per-version free-form notes (replaces RESULTS_LOG.md
          // + devlog.json). Auto-saves on blur to /api/versions/<id>/notes.
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
            if (newVal === (v.notes || "")) return;  // unchanged
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

          li.appendChild(nameSpan);
          li.appendChild(actionsSpan);
          li.appendChild(metaSpan);
          li.appendChild(notesWrap);
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

      function deleteVersion(v) {
        if (!window.confirm("Delete version \\u201C" + v.name + "\\u201D?")) return;
        fetch("/api/versions/" + encodeURIComponent(v.id), {method: "DELETE"})
          .then(function (r) { return r.json(); })
          .then(function (resp) {
            if (!resp.ok) { showError(resp.error || "Delete failed"); return; }
            showError("");
            renderList(resp.store);
          });
      }

      formEl.addEventListener("submit", function (e) {
        e.preventDefault();
        showError("");
        fetch("/api/versions", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            strategy_version: stratEl.value,
            base_id: baseEl.value || null
          })
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

        # ── Run the backtest ──
        df = strat.fetch_data(strat.TICKER, strat.INTERVAL, strat.DAYS_BACK,
                              start_date=start_date, end_date=end_date)
        df = strat.add_indicators(df)
        trades, equity, raw_blocked = strat.run_backtest(df)

        # Trim trades to the requested date range
        start_ts = pd.Timestamp(start_date, tz="UTC")
        end_ts   = pd.Timestamp(end_date,   tz="UTC") + pd.Timedelta(days=1)
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

    # Layer in the active version's strategy + regime allow-lists. Payload
    # overrides win; otherwise the active version supplies defaults.
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

    # Layer in the active version's strategy + regime allow-lists. Payload
    # overrides win; otherwise the active version supplies defaults.
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

    # Layer in the regime allow-lists from the RA page (or explicit payload).
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
    """Return the full versions store including active_version_id."""
    return jsonify(_read_versions())


@app.route("/api/versions", methods=["POST"])
def api_versions_add():
    """Create a new version. Body: {strategy_version, base_id?}.
    Name is auto-generated as 'v<N+1>' from the current highest vN.
    base_id (optional) copies the regime_state from an existing version."""
    body = request.get_json(force=True, silent=True) or {}
    strategy_version = (body.get("strategy_version") or "").strip()
    base_id = (body.get("base_id") or "").strip() or None
    if strategy_version not in ("v1", "v2"):
        return jsonify({"ok": False, "error": "strategy_version must be 'v1' or 'v2'"}), 400
    new_version = _add_version(strategy_version, base_id=base_id)
    return jsonify({"ok": True, "version": new_version})


@app.route("/api/versions/<version_id>", methods=["DELETE"])
def api_versions_delete(version_id):
    """Delete a version. Refuses if it's the last one; auto-switches active
    if the deleted one was active."""
    ok, err = _delete_version(version_id)
    if not ok:
        return jsonify({"ok": False, "error": err}), 400
    return jsonify({"ok": True, "store": _read_versions()})


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
