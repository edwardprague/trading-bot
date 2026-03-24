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
import subprocess
from pathlib import Path

# ── Auto-install Flask if missing ─────────────────────────────────────────────
try:
    from flask import Flask, Response, jsonify
except ImportError:
    print("  Flask not found — installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"],
                          stdout=subprocess.DEVNULL)
    from flask import Flask, Response, jsonify

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).parent
REPORT_FILE   = BASE_DIR / "report.html"
STRATEGY_FILE = BASE_DIR / "strategy.py"

app = Flask(__name__)

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
  <button id="run-btn" onclick="runBacktest()" style="
    background: #4cc9f0; color: #0c0c14; border: none; border-radius: 6px;
    padding: 7px 20px; font-size: 13px; font-weight: 700; cursor: pointer;
    letter-spacing: 0.02em; flex-shrink: 0; transition: background 0.15s;
  ">&#9654;&nbsp; Run Backtest</button>
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
function runBacktest() {
  var btn    = document.getElementById("run-btn");
  var status = document.getElementById("run-status");

  btn.disabled  = true;
  btn.textContent = "Running\u2026";
  status.innerHTML = '<span class="rb-spin"></span>Fetching data and running backtest\u2026';
  status.style.color = "#9090c0";

  fetch("/run", { method: "POST" })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.ok) {
        status.innerHTML = "\u2713\u2009Done \u2014 reloading\u2026";
        status.style.color = "#6bcb77";
        setTimeout(function() { window.location.reload(); }, 900);
      } else {
        btn.disabled = false;
        btn.innerHTML = "&#9654;&nbsp; Run Backtest";
        status.innerHTML = "\u2717\u2009" + (data.error || "Unknown error");
        status.style.color = "#ff6b6b";
      }
    })
    .catch(function() {
      btn.disabled = false;
      btn.innerHTML = "&#9654;&nbsp; Run Backtest";
      status.innerHTML = "\u2717\u2009Request failed \u2014 is the server still running?";
      status.style.color = "#ff6b6b";
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


@app.route("/run", methods=["POST"])
def run_backtest():
    """Run strategy.py as a subprocess and return JSON {ok, output|error}."""
    try:
        result = subprocess.run(
            [sys.executable, str(STRATEGY_FILE)],
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
            timeout=300,        # 5-minute safety timeout
        )
        if result.returncode == 0:
            return jsonify({"ok": True, "output": result.stdout})
        else:
            # Return the last 800 chars of stderr/stdout so the UI can show it
            err = (result.stderr or result.stdout or "Non-zero exit code").strip()
            return jsonify({"ok": False, "error": err[-800:]})

    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "error": "Timed out after 5 minutes"})
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
    # host="0.0.0.0" binds to all interfaces (IPv4 + IPv6) so that
    # macOS routing localhost → ::1 still reaches this server.
    # Port 8080 avoids the macOS AirPlay Receiver conflict on port 5000.
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
