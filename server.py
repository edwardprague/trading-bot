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
_bt_state = {"running": False, "ok": None, "error": None, "no_data": False}

# ── Run-bar HTML (injected into every page response) ──────────────────────────

INJECT_HTML = """
<div id="run-bar" style="
  position: fixed; top: 0; left: 0; right: 0; height: 52px;
  z-index: 9999; display: flex; align-items: center; gap: 12px;
  padding: 0 20px;
  background: #0c0c18; border-bottom: 1px solid #1e1e32;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
">
  <select id="rb-instrument-new" class="rb-select">
    <option value="EURUSD" selected>EURUSD</option>
    <option value="GBPUSD">GBPUSD</option>
  </select>
  <button id="run-new-btn" class="rb-btn rb-btn-green" onclick="runNewVersion()">&#9654;&nbsp; Add New Version</button>

  <span class="rb-sep"></span>

  <div id="rb-range-group" style="display: flex; align-items: center; gap: 12px;">
    <select id="rb-instrument-range" class="rb-select">
      <option value="EURUSD" selected>EURUSD</option>
      <option value="GBPUSD">GBPUSD</option>
    </select>
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
</style>

<script>
(function () {
  /* ── Move action buttons into the run bar (preserve visibility set by strategy.py) ── */
  var _actGroup = document.getElementById("rb-action-group");
  var _actSep   = document.getElementById("rb-act-sep");
  var _copyBtn  = document.getElementById("copy-btn");
  var _delBtn   = document.getElementById("delete-btn");
  if (_actGroup) {
    if (_actSep)  { _actSep.className = "rb-sep";  _actGroup.appendChild(_actSep); }
    if (_copyBtn) { _copyBtn.className = "rb-btn rb-btn-copy"; _actGroup.appendChild(_copyBtn); }
    if (_delBtn)  { _delBtn.className = "rb-btn rb-btn-delete"; _actGroup.appendChild(_delBtn); }
  }

  /* ── Date overlay helper: show MM.DD.YYYY on top of native date input ── */
  function updateOverlay(inputEl, overlayEl) {
    var v = inputEl.value;  /* native value is always YYYY-MM-DD */
    if (!v) { overlayEl.textContent = ""; return; }
    var p = v.split("-");
    if (p.length === 3) overlayEl.textContent = p[1] + "." + p[2] + "." + p[0];
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
              document.getElementById("copy-btn"), document.getElementById("delete-btn")];
  btns.forEach(function (b) { if (b) b.disabled = true; });
  document.getElementById("run-status").innerHTML =
    '<span class="rb-spin"></span>Running\\u2026';
  document.getElementById("run-status").style.color = "#9090c0";
}

function resetButtons() {
  var newBtn   = document.getElementById("run-new-btn");
  var rangeBtn = document.getElementById("run-range-btn");
  newBtn.disabled   = false;
  newBtn.innerHTML   = "&#9654;&nbsp; Add New Version";
  rangeBtn.disabled = false;
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
  var name = getCurrentVersionName();
  var rangeBtn = document.getElementById("run-range-btn");
  if (name) {
    rangeBtn.innerHTML = "&#9654;&nbsp; Add Date Range (" + name + ")";
  } else {
    rangeBtn.innerHTML = "&#9654;&nbsp; Add Date Range";
  }
}

function getSelectedDirection() {
  var el = document.getElementById("ec-direction-select");
  if (el) return el.value;
  var stored = localStorage.getItem("ec_direction");
  return stored || "short_only";
}

function runNewVersion() {
  var instrument = document.getElementById("rb-instrument-new").value;
  var direction  = getSelectedDirection();
  setRunning();
  fetch("/run", { method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode: "new_version", instrument: instrument, direction: direction })
  })
  .then(function (r) { return r.json(); })
  .then(function (data) {
    if (data.started) { pollStatus(); }
    else { resetButtons(); showError(data.error); }
  })
  .catch(function () { resetButtons(); showError("Request failed"); });
}

function runDateRange() {
  var startDate = document.getElementById("rb-start").value;
  var endDate   = document.getElementById("rb-end").value;
  if (!startDate || !endDate) {
    showError("Select both start and end dates");
    return;
  }
  var instrument     = document.getElementById("rb-instrument-range").value;
  var targetVersion  = getCurrentVersionName();
  localStorage.setItem("rb_pending_run_type", "date_range");
  localStorage.setItem("rb_pending_run_version", targetVersion);
  setRunning();
  fetch("/run_range", { method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ start_date: startDate, end_date: endDate, instrument: instrument, target_version: targetVersion, direction: getSelectedDirection() })
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
        setTimeout(pollStatus, 2000);
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


def _backtest_worker(env_overrides=None):
    """Run strategy.py in a background thread and update _bt_state when done."""
    try:
        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)
        result = subprocess.run(
            [sys.executable, str(STRATEGY_FILE)],
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
            timeout=300,        # 5-minute safety timeout
            env=env,
        )
        # Print subprocess output to terminal for debugging
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        with _bt_lock:
            if result.returncode == 0:
                # Check if strategy.py reported no trades (NO_DATA marker)
                if "NO_DATA" in (result.stdout or ""):
                    _bt_state["ok"]      = True
                    _bt_state["no_data"] = True
                    _bt_state["error"]   = None
                else:
                    _bt_state["ok"]      = True
                    _bt_state["no_data"] = False
                    _bt_state["error"]   = None
            else:
                err = (result.stderr or result.stdout or "Non-zero exit code").strip()
                _bt_state["ok"]      = False
                _bt_state["no_data"] = False
                _bt_state["error"]   = err[-800:]
    except subprocess.TimeoutExpired:
        with _bt_lock:
            _bt_state["ok"]    = False
            _bt_state["error"] = "Timed out after 5 minutes"
    except Exception as exc:
        with _bt_lock:
            _bt_state["ok"]    = False
            _bt_state["error"] = str(exc)
    finally:
        with _bt_lock:
            _bt_state["running"] = False


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

    # RUN_MODE=new_version tells strategy.py to increment version
    data = request.get_json(force=True) or {}
    instrument = (data.get("instrument") or "").strip()
    direction  = (data.get("direction") or "").strip()
    env_overrides = {"RUN_MODE": "new_version"}
    if instrument:
        env_overrides["INSTRUMENT"] = instrument
    if direction:
        env_overrides["TRADE_DIRECTION"] = direction
    t = threading.Thread(
        target=_backtest_worker,
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

    instrument     = (data.get("instrument") or "").strip()
    target_version = (data.get("target_version") or "").strip()
    direction      = (data.get("direction") or "").strip()
    env_overrides = {
        "RUN_MODE":       "date_range",
        "RUN_START_DATE": start_date,
        "RUN_END_DATE":   end_date,
    }
    if instrument:
        env_overrides["INSTRUMENT"] = instrument
    if target_version:
        env_overrides["TARGET_VERSION"] = target_version
    if direction:
        env_overrides["TRADE_DIRECTION"] = direction
    t = threading.Thread(
        target=_backtest_worker,
        args=(env_overrides,),
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


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("  Trading Bot Dashboard")
    print("  ────────────────────────────────────────")
    print("  Open  →  http://localhost:8080")
    print("  Stop  →  Ctrl+C")
    print()
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
