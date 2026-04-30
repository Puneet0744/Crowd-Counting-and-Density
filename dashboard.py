from __future__ import annotations

from threading import Thread
from typing import Optional

from flask import Flask, jsonify

from utils import SharedState


def create_app(state: SharedState) -> Flask:
    app = Flask(__name__)

    @app.get("/api/metrics")
    def api_metrics():
        return jsonify(state.snapshot())

    @app.get("/")
    def home():
        # Minimal single-file dashboard (no templates/static needed).
        return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Smart Crowd Density Dashboard</title>
    <style>
      body { font-family: system-ui, Arial, sans-serif; margin: 24px; background: #0b0f19; color: #e7eaf0; }
      .grid { display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 12px; }
      .card { background: #11182a; border: 1px solid #24304a; border-radius: 14px; padding: 16px; }
      .k { opacity: 0.75; font-size: 13px; }
      .v { font-size: 28px; font-weight: 700; margin-top: 6px; }
      .badge { display: inline-block; padding: 4px 10px; border-radius: 999px; font-weight: 700; }
      .SAFE { background: #0e3b1d; color: #7dffa8; border: 1px solid #1e7a3b; }
      .MODERATE { background: #3a2f0b; color: #ffe08a; border: 1px solid #7f6517; }
      .CROWDED { background: #3a0b0b; color: #ff9a9a; border: 1px solid #7f1717; }
      .row { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
      .alert { margin-top: 14px; padding: 12px 14px; border-radius: 12px; border: 1px solid #7f1717; background: #2a0f12; display:none; }
      .muted { opacity: 0.75; }
    </style>
  </head>
  <body>
    <h2>Smart Crowd Density Monitoring System</h2>
    <div class="muted">Live metrics (refreshes every 0.5s)</div>
    <div style="height: 14px"></div>
    <div class="grid">
      <div class="card">
        <div class="row"><div class="k">Overall Density</div><div id="densityBadge" class="badge SAFE">SAFE</div></div>
        <div id="totalCount" class="v">0</div>
        <div class="k" style="margin-top:6px;">Total People</div>
      </div>
      <div class="card">
        <div class="row"><div class="k">Max Cell Count</div><div class="muted">per grid cell</div></div>
        <div id="maxCell" class="v">0.0</div>
      </div>
      <div class="card">
        <div class="row"><div class="k">FPS</div><div class="muted">EMA</div></div>
        <div id="fpsTop" class="v">0.0</div>
      </div>
    </div>
    <div style="height: 12px"></div>
    <div class="card">
      <div class="row">
        <div class="k">Timestamp</div><div id="ts" class="muted"></div>
      </div>
      <div id="alert" class="alert"></div>
    </div>

    <script>
      function setBadge(el, value) {
        el.textContent = value;
        el.classList.remove("SAFE", "MODERATE", "CROWDED");
        el.classList.add(value);
      }
      async function tick() {
        try {
          const r = await fetch("/api/metrics", { cache: "no-store" });
          const d = await r.json();
          document.getElementById("totalCount").textContent = d.total_count;
          document.getElementById("maxCell").textContent = (d.max_cell_count || 0).toFixed(1);
          document.getElementById("fpsTop").textContent = (d.fps || 0).toFixed(1);
          setBadge(document.getElementById("densityBadge"), d.density || "SAFE");
          document.getElementById("ts").textContent = d.timestamp;
          const alertBox = document.getElementById("alert");
          if (d.alert && d.alert.length) {
            alertBox.style.display = "block";
            alertBox.textContent = d.alert;
          } else {
            alertBox.style.display = "none";
            alertBox.textContent = "";
          }
        } catch (e) {}
      }
      tick();
      setInterval(tick, 500);
    </script>
  </body>
</html>
        """

    return app


def run_dashboard_in_thread(state: SharedState, host: str = "127.0.0.1", port: int = 5000) -> Thread:
    app = create_app(state)
    t = Thread(target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False), daemon=True)
    t.start()
    return t

