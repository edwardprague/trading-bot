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

# ── Auto-install Flask if missing ─────────────────────────────────────────────
try:
    from flask import Flask, Response, jsonify, request
except ImportError:
    print("  Flask not found — installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"],
                          stdout=subprocess.DEVNULL)
    from flask import Flask, Response, jsonify, request

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).parent
REPORT_FILE   = BASE_DIR / "report.html"
STRATEGY_FILE = BASE_DIR / "strategy.py"

app = Flask(__name__)

# ── Backtest state (shared between the Flask thread and the worker thread) ─────
_bt_lock  = threading.Lock()
_bt_state = {"running": False, "ok": None, "error": None, "no_data": False, "stage": "", "progress": 0}

# ── Run-bar HTML (injected into every page response) ──────────────────────────

INJECT_HTML = """
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
  body { padding-top: 52px !important; }

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
  if (_actGroup) {
    if (_devlogBtn) { _devlogBtn.className = "rb-devlog-btn"; _devlogBtn.style.display = ""; _actGroup.appendChild(_devlogBtn); }
    if (_actSep)    { _actSep.className = "rb-sep";  _actGroup.appendChild(_actSep); }
    if (_copyBtn)   { _copyBtn.className = "rb-btn rb-btn-copy"; _actGroup.appendChild(_copyBtn); }
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
    var required = ["v1"];
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
              document.getElementById("copy-btn")];
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

function runNewVersion() {
  var instrument = getSelectedInstrument();
  var direction  = getSelectedDirection();
  var interval   = getSelectedInterval();
  var version    = getSelectedVersion();
  localStorage.setItem("rb_pending_run_type", "new_version_auto");
  setRunning();
  fetch("/run", { method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode: "new_version", instrument: instrument, direction: direction, interval: interval, strategy_version: version, ema_short: getSelectedEmaShort(), ema_mid: getSelectedEmaMid(), ema_long: getSelectedEmaLong(), stop_loss_pips: getSelectedStopPips(), rrr_risk: getSelectedRrrRisk(), rrr_reward: getSelectedRrrReward(), blocked_hours: getSelectedBlockedHours(), max_daily_losses: getSelectedMaxDD() })
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
      body: JSON.stringify({ ranges: selectedRanges, instrument: instrument, target_version: targetVersion, strategy_version: version, direction: getSelectedDirection(), interval: getSelectedInterval(), ema_short: getSelectedEmaShort(), ema_mid: getSelectedEmaMid(), ema_long: getSelectedEmaLong(), stop_loss_pips: getSelectedStopPips(), rrr_risk: getSelectedRrrRisk(), rrr_reward: getSelectedRrrReward(), blocked_hours: getSelectedBlockedHours(), max_daily_losses: getSelectedMaxDD() })
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
    body: JSON.stringify({ start_date: startDate, end_date: endDate, instrument: instrument, target_version: targetVersion, strategy_version: version, direction: getSelectedDirection(), interval: getSelectedInterval(), ema_short: getSelectedEmaShort(), ema_mid: getSelectedEmaMid(), ema_long: getSelectedEmaLong(), stop_loss_pips: getSelectedStopPips(), rrr_risk: getSelectedRrrRisk(), rrr_reward: getSelectedRrrReward(), blocked_hours: getSelectedBlockedHours(), max_daily_losses: getSelectedMaxDD() })
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
            if val:
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
