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

# Shared regime-filter state — written by /run_regime_analysis whenever the
# RA page runs an analysis, read by /run, /run_range, /run_batch so the BD
# inherits whatever allow-lists the user last set on the Regimes page. This
# lets the user tune regime gates on the RA page and have those gates
# automatically apply to subsequent BD backtests without re-entering them.
REGIME_FILTER_STATE_FILE = DATA_DIR / "regime_filter_state.json"

app = Flask(__name__)


# ── Regime filter state — shared between RA (writer) and BD (reader) ──────────

def _read_regime_filter_state():
    """Return {'allowed_macro_regimes': [...], 'allowed_micro_regimes': [...]}
    from the shared state file, or None if it doesn't exist / is unreadable.

    The file is written by /run_regime_analysis and consumed by the BD
    /run, /run_range, /run_batch handlers. Internal-key format is used
    throughout (e.g. 'staircase_down', 'ranging_medium')."""
    try:
        if not REGIME_FILTER_STATE_FILE.exists():
            return None
        with open(REGIME_FILTER_STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        if not isinstance(state, dict):
            return None
        return {
            "allowed_macro_regimes": list(state.get("allowed_macro_regimes", []) or []),
            "allowed_micro_regimes": list(state.get("allowed_micro_regimes", []) or []),
        }
    except (OSError, ValueError):
        return None


def _write_regime_filter_state(allowed_macro, allowed_micro):
    """Persist the RA toggle state so subsequent BD backtests pick it up.
    Best-effort: a write failure is logged but doesn't break the response.
    Stores both lists plus an ISO timestamp for diagnostics."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "allowed_macro_regimes": list(allowed_macro or []),
            "allowed_micro_regimes": list(allowed_micro or []),
            "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        tmp = REGIME_FILTER_STATE_FILE.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        tmp.replace(REGIME_FILTER_STATE_FILE)
    except OSError as e:
        print(f"  [regime_filter_state] write failed: {e}", file=sys.stderr)


def _apply_regime_filter_state_to_env(env_overrides, payload):
    """Inject ALLOWED_MACRO_REGIMES / ALLOWED_MICRO_REGIMES into env_overrides
    using the following precedence:

      1. Explicit values in the request payload override everything.
      2. Otherwise, fall back to data/regime_filter_state.json (written by
         the RA page).
      3. Otherwise, omit — strategy_v2's hardcoded defaults take effect.

    The strategy module distinguishes 'unset' from 'empty string': empty
    means 'gate disabled', unset means 'use default'. Both are valid
    persisted states, so when the file is present we always set the env
    var (joining with commas; empty list → empty string)."""
    payload_macro = payload.get("allowed_macro_regimes") if isinstance(payload, dict) else None
    payload_micro = payload.get("allowed_micro_regimes") if isinstance(payload, dict) else None

    if payload_macro is not None:
        env_overrides["ALLOWED_MACRO_REGIMES"] = ",".join(payload_macro)
    if payload_micro is not None:
        env_overrides["ALLOWED_MICRO_REGIMES"] = ",".join(payload_micro)

    if "ALLOWED_MACRO_REGIMES" in env_overrides and "ALLOWED_MICRO_REGIMES" in env_overrides:
        return  # both explicitly set by caller

    state = _read_regime_filter_state()
    if state is None:
        return
    if "ALLOWED_MACRO_REGIMES" not in env_overrides:
        env_overrides["ALLOWED_MACRO_REGIMES"] = ",".join(state["allowed_macro_regimes"])
    if "ALLOWED_MICRO_REGIMES" not in env_overrides:
        env_overrides["ALLOWED_MICRO_REGIMES"] = ",".join(state["allowed_micro_regimes"])

# ── Backtest state (shared between the Flask thread and the worker thread) ─────
_bt_lock  = threading.Lock()
_bt_state = {"running": False, "ok": None, "error": None, "no_data": False, "stage": "", "progress": 0}

# ── Run-bar HTML (injected into every page response) ──────────────────────────

INJECT_HTML = """
<nav class="top-nav" id="top-nav">
  <span class="top-nav-brand">Fractal Bot</span>
  <ul class="top-nav-items">
    <li><a class="top-nav-link top-nav-link-active" href="/">Backtesting</a></li>
    <li><a class="top-nav-link" href="/results/regime_analysis.html">Regimes</a></li>
    <li><span class="top-nav-link top-nav-link-disabled" aria-disabled="true">Discovery</span></li>
    <li><span class="top-nav-link top-nav-link-disabled" aria-disabled="true">Versions</span></li>
  </ul>
</nav>

<div id="run-bar" style="
  position: fixed; top: 0; left: 0; right: 0; height: 52px;
  z-index: 9999; display: flex; align-items: center; gap: 12px;
  padding: 0 20px;
  background: #0c0c18; border-bottom: 1px solid #1e1e32;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
">
  <button id="run-new-btn" class="rb-btn rb-btn-green" onclick="runNewVersion()">&#9654;&nbsp; Add Year</button>

  <span class="rb-sep"></span>

  <div id="rb-range-group" style="display: flex; align-items: center; gap: 12px;">
    <label class="rb-label" for="rb-start">From</label>
    <span class="rb-date-wrap"><input type="date" id="rb-start" class="rb-date"><span class="rb-date-overlay" id="rb-start-overlay"></span></span>
    <label class="rb-label" for="rb-end">To</label>
    <span class="rb-date-wrap"><input type="date" id="rb-end" class="rb-date"><span class="rb-date-overlay" id="rb-end-overlay"></span></span>
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

  .rb-date-wrap {
    position: relative; display: inline-block; width: 130px; flex-shrink: 0;
  }
  .rb-date {
    background: #14142a; color: transparent; border: 1px solid #2a2a44;
    border-radius: 5px; padding: 5px 8px; font-size: 12px;
    font-family: inherit; width: 100%; flex-shrink: 0;
    color-scheme: dark; position: relative; z-index: 1;
  }
  .rb-date:focus { border-color: #4cc9f0; outline: none; }
  .rb-date-overlay {
    position: absolute; top: 0; left: 0; right: 22px; bottom: 0;
    display: flex; align-items: center;
    padding: 5px 8px; font-size: 12px; font-family: inherit;
    color: #c0c0e0; pointer-events: none; z-index: 2;
    white-space: nowrap;
  }

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
(function () {
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

  /* ── Date overlay helper: show Mon-DD-YY on top of native date input ── */
  var _ovMn = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  function updateOverlay(inputEl, overlayEl) {
    var v = inputEl.value;  /* native value is always YYYY-MM-DD */
    if (!v) { overlayEl.textContent = ""; return; }
    var p = v.split("-");
    if (p.length === 3) overlayEl.textContent = _ovMn[parseInt(p[1], 10) - 1] + "-" + p[2] + "-" + p[0].slice(2);
    else overlayEl.textContent = "";
  }

  /* ── Persist date pickers via localStorage ─────────────────────── */
  var startEl      = document.getElementById("rb-start");
  var endEl        = document.getElementById("rb-end");
  var startOverlay = document.getElementById("rb-start-overlay");
  var endOverlay   = document.getElementById("rb-end-overlay");

  var savedStart = localStorage.getItem("rb_start_date");
  var savedEnd   = localStorage.getItem("rb_end_date");
  if (savedStart) startEl.value = savedStart;
  if (savedEnd)   endEl.value   = savedEnd;

  updateOverlay(startEl, startOverlay);
  updateOverlay(endEl, endOverlay);

  startEl.addEventListener("change", function () {
    localStorage.setItem("rb_start_date", startEl.value);
    updateOverlay(startEl, startOverlay);
  });
  endEl.addEventListener("change", function () {
    localStorage.setItem("rb_end_date", endEl.value);
    updateOverlay(endEl, endOverlay);
  });

  /* ── On load: resume polling if a backtest is already running ───── */
  fetch("/status")
    .then(function (r) { return r.json(); })
    .then(function (data) {
      if (data.running) { setRunning(); pollStatus(); }
    })
    .catch(function () {});

  /* ── Ensure all strategy versions appear in the selector ───────────────── */
  (function () {
    var sel = document.getElementById("version-select");
    if (!sel) return;
    var required = ["v1", "v2"];
    var existing = {};
    for (var k = 0; k < sel.options.length; k++) existing[sel.options[k].value] = true;
    required.forEach(function (v) {
      if (!existing[v]) {
        var opt = document.createElement("option");
        opt.value = v; opt.textContent = v;
        sel.appendChild(opt);
      }
    });
  }());

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
})();

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
  var displayName = window._currentVersionDisplayName || getCurrentVersionName();
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
    # Inject the run-bar just before </body> so it sits on top of everything
    html = html.replace("</body>", INJECT_HTML + "\n</body>", 1)
    return Response(html, mimetype="text/html")


@app.route("/style.css")
def serve_css():
    """Serve the dashboard stylesheet."""
    css_path = BASE_DIR / "style.css"
    if not css_path.exists():
        return Response("", mimetype="text/css")
    return Response(css_path.read_text(encoding="utf-8"), mimetype="text/css")


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
    """
    if not RESULTS_DIR.exists():
        abort(404)
    try:
        return send_from_directory(str(RESULTS_DIR), filename, conditional=True)
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
        if not start_date or not end_date:
            return jsonify({"error": "start_date and end_date are required"}), 400

        # Lazy-import so server.py can boot without pyarrow/pandas if the
        # regime feature isn't used. Importing strategy_v2 also triggers its
        # module-level regime-labels load.
        # The regime labeler is GBPUSD-only, so pin TICKER here regardless of
        # what the strategy module imported at startup.
        os.environ.setdefault("INSTRUMENT", "GBPUSD")
        import pandas as pd  # noqa: F401  — re-export used below
        import strategy_v2 as strat
        import regime_analysis as rl
        # Force GBPUSD in case strategy_v2 was imported earlier with a
        # different instrument (e.g. EURUSD default).
        strat.TICKER          = "GBPUSD"
        strat._INSTRUMENT     = "GBPUSD"
        strat.MASSIVE_TICKER  = strat._INSTRUMENT_MAP.get("GBPUSD", strat.MASSIVE_TICKER)

        # ── Override strategy_v2 module globals with this request's filters ──
        strat.ALLOWED_MACRO_KEYS = {strat._macro_key(n) for n in allowed_macro}
        strat.ALLOWED_MICRO_KEYS = {strat._micro_key(n) for n in allowed_micro}

        # Persist the toggle state so subsequent BD backtests pick up the same
        # allow-lists automatically. Store the normalised internal-key form so
        # the BD endpoints can pass the value straight through as an env var.
        _write_regime_filter_state(
            sorted(strat.ALLOWED_MACRO_KEYS),
            sorted(strat.ALLOWED_MICRO_KEYS),
        )
        # EMA filter stays on by default for the interactive view; clients can
        # toggle it via the existing run-bar style refactor later.
        strat.USE_EMA_FILTER = bool(payload.get("use_ema_filter", True))

        # ── Run the backtest ──
        df = strat.fetch_data(strat.TICKER, strat.INTERVAL, strat.DAYS_BACK,
                              start_date=start_date, end_date=end_date)
        df = strat.add_indicators(df)
        trades, equity, raw_blocked = strat.run_backtest(df)

        # Trim to the requested date range
        start_ts = pd.Timestamp(start_date, tz="UTC")
        end_ts   = pd.Timestamp(end_date,   tz="UTC") + pd.Timedelta(days=1)
        if not trades.empty:
            _t = pd.to_datetime(trades["entry_ts"])
            _t = _t.dt.tz_convert("UTC") if _t.dt.tz is not None else _t.dt.tz_localize("UTC")
            trades = trades[(_t >= start_ts) & (_t < end_ts)].reset_index(drop=True)

        # ── Load fractal labels + macro from parquet ──
        # Use whichever engine is installed (pyarrow preferred, fastparquet OK).
        labels_path = BASE_DIR / "data" / "regime_labels.parquet"
        macro = {}
        if labels_path.exists():
            # Accept either metadata key for back-compat with parquets written
            # before the rename: 'regime_analysis' is canonical going forward;
            # 'regime_labeler' is the legacy key.
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

        # ── Reconstruct periods (consecutive same-regime fractals) for
        # the timeline + daily breakdown chips. Period info is derivable
        # from the parquet's per-fractal regime column — we don't need a
        # full stage-2 rerun. ──
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
                    "regime":  label_i,   # already at fine granularity in the parquet
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

        # ── Derive blocked macro/micro keys from the request payload ──
        # The toggle panel sends ALLOWED lists; "blocked" = the complement.
        allowed_macro_keys = strat.ALLOWED_MACRO_KEYS
        allowed_micro_keys = strat.ALLOWED_MICRO_KEYS
        all_macro_keys = set(rl.MACRO_REGIME_ORDER)
        all_micro_keys = set(rl.REGIME_ORDER)
        blocked_macro_keys = all_macro_keys - allowed_macro_keys
        blocked_micro_keys = all_micro_keys - allowed_micro_keys

        # ── Build a per-fractal micro asof series from the FRESHLY-loaded
        # parquet (so attribution is consistent with the in-range fractals
        # we already reconstructed periods from). ──
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

        # ── Attribute regime + macro_label to fired trades ──
        trades = _attribute(trades)

        # ── Blocked signals → DataFrame (with same attribution) ──
        if raw_blocked:
            blocked_df = pd.DataFrame(raw_blocked).rename(columns={"timestamp": "entry_ts"})
            if "entry_ts" in blocked_df.columns:
                _bts = pd.to_datetime(blocked_df["entry_ts"])
                _bts = _bts.dt.tz_convert("UTC") if _bts.dt.tz is not None else _bts.dt.tz_localize("UTC")
                blocked_df = blocked_df[(_bts >= start_ts) & (_bts < end_ts)].reset_index(drop=True)
        else:
            blocked_df = pd.DataFrame(columns=["entry_ts", "win", "pnl", "reason", "direction"])
        blocked_df = _attribute(blocked_df)

        # ── Compute filtered + unfiltered stats ──
        def _filter_by_macro(td):
            if td.empty or not blocked_macro_keys:
                return td.copy(), 0
            _ts = pd.to_datetime(td["entry_ts"])
            _ts = _ts.dt.tz_convert("UTC") if _ts.dt.tz is not None else _ts.dt.tz_localize("UTC")
            day = _ts.dt.strftime("%Y-%m-%d")
            mlbl = day.map(lambda d: macro.get(d, {}).get("label"))
            mask = mlbl.isin(blocked_macro_keys)
            return td[~mask].reset_index(drop=True), int(mask.sum())

        filtered_trades, n_excluded = _filter_by_macro(trades)
        perf_df = rl._compute_perf_df(filtered_trades)
        agg_stats = rl._compute_aggregate_stats(filtered_trades)
        total_in_range_trades = int(len(trades))

        filter_state_label, filter_state_class, macro_filter_note, macro_table_filter_note = \
            rl.compute_filter_label(blocked_macro_keys, total_in_range_trades, n_excluded)

        # ── Build HTML chunks ──
        stats_bar_inner = rl.build_stats_bar_html(
            agg_stats, filter_state_label, filter_state_class)

        macro_perf_table = rl.build_macro_perf_table(
            macro, trades, blocked_macro_keys=blocked_macro_keys,
            blocked_signals_df=blocked_df)

        perf_table = rl.build_perf_table_html(
            perf_df, regime_count,
            blocked_micro_keys=blocked_micro_keys,
            trades_df=trades,
            blocked_signals_df=blocked_df,
            allowed_macro_keys=allowed_macro_keys,
        )

        trades_per_day = rl.compute_trades_per_day(trades)
        timeline_inner = rl.build_timeline_section_html(
            periods, macro, trades_per_day, start_date, end_date, regime_count)

        # Daily performance — we need a "full_df"-shaped frame for
        # _trading_days_in_range. Reuse the indicator-augmented df from the
        # backtest, restricted to the requested range plus a small buffer.
        # Available chart days inferred from results/regime_charts/.
        available_chart_days = set()
        chart_dir = BASE_DIR / "results" / "regime_charts"
        if chart_dir.exists():
            for png in chart_dir.glob("*.png"):
                available_chart_days.add(png.stem)

        # Build a low-activity-day set the same way build_report does.
        from regime_analysis import LOW_ACTIVITY_FRACTAL_THRESHOLD
        fractals_per_day = {}
        if not in_range.empty:
            for d in in_range["timestamp"].dt.strftime("%Y-%m-%d"):
                fractals_per_day[d] = fractals_per_day.get(d, 0) + 1
        # Use rl._trading_days_in_range — but we need to mimic its arg shape.
        # Easier: derive from df.
        df_dts = pd.to_datetime(df["Datetime"])
        df_dts = df_dts.dt.tz_convert("UTC") if df_dts.dt.tz is not None else df_dts.dt.tz_localize("UTC")
        df_in_rng = df[(df_dts >= start_ts) & (df_dts < end_ts)]
        df_in_rng_dts = pd.to_datetime(df_in_rng["Datetime"])
        df_in_rng_dts = df_in_rng_dts.dt.tz_convert("UTC") if df_in_rng_dts.dt.tz is not None else df_in_rng_dts.dt.tz_localize("UTC")
        trading_days_all = sorted(set(df_in_rng_dts.dt.strftime("%Y-%m-%d")))
        low_activity_days = {
            d for d in trading_days_all
            if fractals_per_day.get(d, 0) < LOW_ACTIVITY_FRACTAL_THRESHOLD
        }

        # Temporarily override START_DATE/END_DATE on the rl module so its
        # _trading_days_in_range helper (called inside build_daily_breakdown)
        # uses this request's range.
        _orig_start, _orig_end = rl.START_DATE, rl.END_DATE
        rl.START_DATE = start_date
        rl.END_DATE   = end_date
        try:
            daily_table_html = rl.build_daily_breakdown(
                periods, trades, df, available_chart_days,
                in_range, low_activity_days,
                macro=macro, blocked_macro_keys=blocked_macro_keys,
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
            Indicates a low-activity day (fewer than {LOW_ACTIVITY_FRACTAL_THRESHOLD}
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
        regime_perf_inner = f"""
          <h2>Micro regime performance <span class="regime-dim regime-small">(v2 short-only)</span></h2>
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

    # Layer in the regime allow-lists from the RA page (or explicit payload).
    _apply_regime_filter_state_to_env(env_overrides, data)

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

    # Layer in the regime allow-lists from the RA page (or explicit payload).
    _apply_regime_filter_state_to_env(env_overrides, data)

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
    _apply_regime_filter_state_to_env(shared_params, data)

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
