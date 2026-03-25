"""
server.py — Trading Bot Dashboard
==================================
Serves report.html at http://localhost:8080 and injects a "Run Backtest"
button that triggers strategy.py and auto-refreshes the page on completion.

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
_bt_state = {"running": False, "ok": None, "error": None}

# ── Run-bar HTML (injected into every page response) ──────────────────────────
#
#  • Fixed bar across the top (52px tall, z-index 9999)
#  • "Run Backtest" button POSTs to /run, shows a spinner, reloads on success
#  • body padding-top pushes all existing content clear of the bar

INJECT_HTML = """
<div id="run-bar" style="
  position: fixed; top: 0; left: 0; right: 0; height: 52px;
  z-index: 9999; display: flex; align-items: center; gap: 14px;
  padding: 0 20px;
  background: #0c0c18; border-bottom: 1px solid #1e1e32;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
">
  <button id="run-btn" onclick="runBacktest()">&#9654;&nbsp; Run Backtest</button>
  <span id="run-status" style="font-size: 13px; color: #666690;"></span>
</div>

<style>
  /* Push page content below the fixed bar */
  body { padding-top: 52px !important; }

  #run-btn:hover:not(:disabled) { background: #74d8f7; }
  #run-btn:disabled { background: #1e1e38; color: #404060; cursor: not-allowed; }

  @keyframes rb-spin { to { transform: rotate(360deg); } }
  .rb-spin {
    display: inline-block; width: 13px; height: 13px; margin-right: 7px;
    border: 2px solid #303050; border-top-color: #4cc9f0;
    border-radius: 50%; animation: rb-spin 0.75s linear infinite;
    vertical-align: middle;
  }
</style>

<script>
// On load: if a backtest is already in progress (e.g. page was refreshed
// mid-run), pick up where we left off and start polling immediately.
document.addEventListener("DOMContentLoaded", function() {
  fetch("/status")
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.running) {
        var btn    = document.getElementById("run-btn");
        var status = document.getElementById("run-status");
        btn.disabled    = true;
        btn.textContent = "Running\u2026";
        status.innerHTML  = '<span class="rb-spin"></span>Running\u2026';
        status.style.color = "#9090c0";
        pollStatus();
      }
    })
    .catch(function() {});
});

function runBacktest() {
  var btn    = document.getElementById("run-btn");
  var status = document.getElementById("run-status");

  btn.disabled    = true;
  btn.textContent = "Running\u2026";
  status.innerHTML  = '<span class="rb-spin"></span>Running\u2026';
  status.style.color = "#9090c0";

  fetch("/run", { method: "POST" })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.started) {
        pollStatus();
      } else {
        btn.disabled   = false;
        btn.innerHTML  = "&#9654;&nbsp; Run Backtest";
        status.innerHTML   = "\u2717\u2009" + (data.error || "Unknown error");
        status.style.color = "#ff6b6b";
      }
    })
    .catch(function() {
      btn.disabled   = false;
      btn.innerHTML  = "&#9654;&nbsp; Run Backtest";
      status.innerHTML   = "\u2717\u2009Request failed \u2014 is the server still running?";
      status.style.color = "#ff6b6b";
    });
}

function pollStatus() {
  fetch("/status")
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.running) {
        setTimeout(pollStatus, 2000);
      } else if (data.ok) {
        var status = document.getElementById("run-status");
        status.innerHTML   = "\u2713\u2009Complete \u2014 refreshing\u2026";
        status.style.color = "#6bcb77";
        setTimeout(function() { window.location.reload(); }, 900);
      } else {
        var btn    = document.getElementById("run-btn");
        var status = document.getElementById("run-status");
        btn.disabled   = false;
        btn.innerHTML  = "&#9654;&nbsp; Run Backtest";
        status.innerHTML   = "\u2717\u2009" + (data.error || "Unknown error");
        status.style.color = "#ff6b6b";
      }
    })
    .catch(function() {
      // Brief hiccup — keep polling
      setTimeout(pollStatus, 2000);
    });
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
    Click <strong style="color: #d0d0ee;">&#9654;&nbsp;Run Backtest</strong>
    to run <code style="color:#4cc9f0">strategy.py</code>
    and generate the first report.
  </span>
</body>
</html>""".replace("__INJECT__", INJECT_HTML)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve report.html with the Run Backtest bar injected."""
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


def _backtest_worker():
    """Run strategy.py in a background thread and update _bt_state when done."""
    try:
        result = subprocess.run(
            [sys.executable, str(STRATEGY_FILE)],
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
            timeout=300,        # 5-minute safety timeout
        )
        with _bt_lock:
            if result.returncode == 0:
                _bt_state["ok"]    = True
                _bt_state["error"] = None
            else:
                err = (result.stderr or result.stdout or "Non-zero exit code").strip()
                _bt_state["ok"]    = False
                _bt_state["error"] = err[-800:]
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
    """Start strategy.py in a background thread; return immediately."""
    with _bt_lock:
        if _bt_state["running"]:
            return jsonify({"ok": False, "error": "A backtest is already running"})
        _bt_state["running"] = True
        _bt_state["ok"]      = None
        _bt_state["error"]   = None

    t = threading.Thread(target=_backtest_worker, daemon=True)
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
        # Match files starting with the version name followed by _ or .
        # e.g. v1_EURUSD_2026-03-25.png, v1_EURUSD_2026-03-25_rpf.png
        for f in results_dir.iterdir():
            if f.name.startswith(name + "_") or f.name.startswith(name + "."):
                try:
                    f.unlink()
                except OSError:
                    pass  # best-effort deletion

    return jsonify({"ok": True})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("  Trading Bot Dashboard")
    print("  ────────────────────────────────────────")
    print("  Open  →  http://localhost:8080")
    print("  Stop  →  Ctrl+C")
    print()
    # host="0.0.0.0" binds to all interfaces (IPv4 + IPv6) so that
    # macOS routing localhost → ::1 still reaches this server.
    # Port 8080 avoids the macOS AirPlay Receiver conflict on port 5000.
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
