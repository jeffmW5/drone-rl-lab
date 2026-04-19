#!/usr/bin/env python3
"""
Launch a live browser dashboard for drone-rl-lab training progress.

Uses only the Python stdlib plus the existing training_progress helpers.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
import webbrowser
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import training_progress as tp


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Training Dashboard</title>
  <style>
    :root {
      --bg: #0b1020;
      --bg2: #111a30;
      --card: rgba(255,255,255,0.08);
      --line: rgba(255,255,255,0.14);
      --text: #eef3ff;
      --muted: #9db0d0;
      --accent: #64d8ff;
      --accent2: #7cff9f;
      --warn: #ffcc66;
      --bad: #ff7b86;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(100,216,255,0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(124,255,159,0.12), transparent 32%),
        linear-gradient(180deg, var(--bg), var(--bg2));
      min-height: 100vh;
    }
    .wrap { max-width: 1180px; margin: 0 auto; padding: 28px; }
    .hero {
      display: flex; justify-content: space-between; align-items: flex-start; gap: 24px;
      margin-bottom: 22px;
    }
    .title { font-size: 32px; font-weight: 800; letter-spacing: 0.02em; margin: 0; }
    .subtitle { color: var(--muted); margin-top: 6px; }
    .badge {
      display: inline-flex; align-items: center; gap: 8px; padding: 10px 14px;
      border: 1px solid var(--line); border-radius: 999px; background: rgba(255,255,255,0.06);
      font-weight: 700;
    }
    .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--accent2); box-shadow: 0 0 18px var(--accent2); }
    .grid { display: grid; grid-template-columns: 1.25fr 0.95fr; gap: 18px; }
    .card {
      background: var(--card); border: 1px solid var(--line); border-radius: 22px;
      padding: 20px; backdrop-filter: blur(12px);
      box-shadow: 0 20px 60px rgba(0,0,0,0.24);
    }
    .card h2, .card h3 { margin: 0 0 12px; }
    .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.12em; }
    .big { font-size: 42px; font-weight: 800; line-height: 1; margin-top: 6px; }
    .bar {
      margin-top: 16px; height: 18px; border-radius: 999px; background: rgba(255,255,255,0.08);
      overflow: hidden; border: 1px solid rgba(255,255,255,0.09);
    }
    .bar > div {
      height: 100%; width: 0%;
      background: linear-gradient(90deg, var(--accent), var(--accent2));
      border-radius: 999px; transition: width 0.6s ease;
      box-shadow: 0 0 24px rgba(100,216,255,0.35);
    }
    .row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 18px; }
    .metric {
      padding: 14px; border-radius: 16px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.06);
    }
    .metric .v { font-size: 24px; font-weight: 750; margin-top: 5px; }
    .kv { display: grid; grid-template-columns: 140px 1fr; gap: 10px; margin: 8px 0; font-size: 15px; }
    .kv .k { color: var(--muted); }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; word-break: break-all; }
    .status {
      margin-top: 14px; padding: 12px 14px; border-radius: 14px; font-weight: 700;
      background: rgba(100,216,255,0.12); color: #d8f7ff;
    }
    .status.warn { background: rgba(255,204,102,0.14); color: #ffe4a3; }
    .status.bad { background: rgba(255,123,134,0.14); color: #ffd5d8; }
    canvas {
      width: 100%; height: 210px; display: block; border-radius: 16px;
      background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
      border: 1px solid rgba(255,255,255,0.06);
    }
    .footer { margin-top: 16px; color: var(--muted); font-size: 13px; }
    @media (max-width: 960px) {
      .grid { grid-template-columns: 1fr; }
      .row { grid-template-columns: 1fr; }
      .hero { flex-direction: column; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div>
        <h1 class="title">Training Progress</h1>
        <div class="subtitle" id="subtitle">Connecting to monitor...</div>
      </div>
      <div class="badge"><span class="dot"></span><span id="badge-text">Waiting</span></div>
    </div>

    <div class="grid">
      <div class="card">
        <div class="label">Overall Progress</div>
        <div class="big" id="percent">0.0%</div>
        <div class="bar"><div id="bar-fill"></div></div>
        <div class="row">
          <div class="metric"><div class="label">Iteration</div><div class="v" id="iter">-- / --</div></div>
          <div class="metric"><div class="label">Steps</div><div class="v" id="steps">--</div></div>
          <div class="metric"><div class="label">ETA</div><div class="v" id="eta">--</div></div>
        </div>
        <div class="row">
          <div class="metric"><div class="label">Reward</div><div class="v" id="reward">--</div></div>
          <div class="metric"><div class="label">PG Loss</div><div class="v" id="pg">--</div></div>
          <div class="metric"><div class="label">V Loss</div><div class="v" id="vl">--</div></div>
        </div>
        <div id="status-box" class="status">Waiting for first update...</div>
      </div>

      <div class="card">
        <h3>Run Details</h3>
        <div class="kv"><div class="k">Experiment</div><div id="experiment">--</div></div>
        <div class="kv"><div class="k">Pod</div><div class="mono" id="pod">--</div></div>
        <div class="kv"><div class="k">SSH</div><div class="mono" id="ssh">--</div></div>
        <div class="kv"><div class="k">Log</div><div class="mono" id="log">--</div></div>
        <div class="kv"><div class="k">Log Age</div><div id="log-age">--</div></div>
        <div class="kv"><div class="k">Process</div><div class="mono" id="proc">--</div></div>
        <div class="kv"><div class="k">Elapsed</div><div id="elapsed">--</div></div>
        <div class="footer">Auto-refreshes every 2 seconds.</div>
      </div>
    </div>

    <div class="card" style="margin-top:18px;">
      <h3>Recent Reward Trend</h3>
      <canvas id="chart" width="1100" height="210"></canvas>
    </div>
  </div>

  <script>
    function fmtSeconds(sec) {
      if (sec == null || !isFinite(sec)) return '--';
      sec = Math.max(0, Math.round(sec));
      const h = Math.floor(sec / 3600);
      const m = Math.floor((sec % 3600) / 60);
      const s = sec % 60;
      if (h) return `${h}h ${String(m).padStart(2, '0')}m ${String(s).padStart(2, '0')}s`;
      if (m) return `${m}m ${String(s).padStart(2, '0')}s`;
      return `${s}s`;
    }

    function drawChart(values) {
      const canvas = document.getElementById('chart');
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = 'rgba(255,255,255,0.02)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      if (!values || values.length === 0) return;
      const pad = 24;
      const lo = Math.min(...values);
      const hi = Math.max(...values);
      const range = Math.max(hi - lo, 1e-6);
      ctx.strokeStyle = 'rgba(255,255,255,0.10)';
      ctx.lineWidth = 1;
      for (let i = 0; i < 4; i++) {
        const y = pad + (canvas.height - 2 * pad) * i / 3;
        ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(canvas.width - pad, y); ctx.stroke();
      }
      const pts = values.map((v, i) => {
        const x = pad + (canvas.width - 2 * pad) * (values.length === 1 ? 0.5 : i / (values.length - 1));
        const y = canvas.height - pad - ((v - lo) / range) * (canvas.height - 2 * pad);
        return [x, y];
      });
      const grad = ctx.createLinearGradient(0, 0, canvas.width, 0);
      grad.addColorStop(0, '#64d8ff');
      grad.addColorStop(1, '#7cff9f');
      ctx.strokeStyle = grad;
      ctx.lineWidth = 4;
      ctx.beginPath();
      pts.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
      ctx.stroke();
      ctx.fillStyle = '#eef3ff';
      ctx.font = '12px ui-monospace, monospace';
      ctx.fillText(`min ${lo.toFixed(2)}`, pad, canvas.height - 6);
      ctx.fillText(`max ${hi.toFixed(2)}`, canvas.width - pad - 70, 16);
    }

    async function refresh() {
      const res = await fetch('/api/status');
      const data = await res.json();
      document.getElementById('subtitle').textContent = data.completed ? 'Run completed' : 'Live RunPod training monitor';
      document.getElementById('badge-text').textContent = `${data.status}`;
      document.getElementById('experiment').textContent = data.experiment || '--';
      document.getElementById('pod').textContent = data.pod_id || '--';
      document.getElementById('ssh').textContent = data.ssh_target || '--';
      document.getElementById('log').textContent = data.log_path || '--';
      document.getElementById('log-age').textContent = data.log_age_seconds == null ? '--' : `${Math.round(data.log_age_seconds)}s old`;
      document.getElementById('proc').textContent = data.process_summary || '--';
      document.getElementById('iter').textContent = `${data.iteration ?? '--'} / ${data.total_iterations ?? '--'}`;
      document.getElementById('steps').textContent = data.step ?? '--';
      document.getElementById('reward').textContent = data.reward == null ? '--' : data.reward.toFixed(2);
      document.getElementById('pg').textContent = data.pg_loss == null ? '--' : data.pg_loss.toFixed(4);
      document.getElementById('vl').textContent = data.v_loss == null ? '--' : data.v_loss.toFixed(4);
      document.getElementById('elapsed').textContent = `${fmtSeconds(data.elapsed_seconds)} / ${fmtSeconds(data.budget_seconds)}`;

      const frac = data.progress_fraction || 0;
      document.getElementById('percent').textContent = `${(frac * 100).toFixed(1)}%`;
      document.getElementById('bar-fill').style.width = `${Math.max(0, Math.min(100, frac * 100))}%`;
      document.getElementById('eta').textContent = fmtSeconds(data.eta_seconds);

      const statusBox = document.getElementById('status-box');
      let cls = 'status';
      let text = data.status;
      if (data.stale_warning) { cls += ' warn'; text += ' | log has gone stale'; }
      if (data.error) { cls += ' bad'; text = data.error; }
      statusBox.className = cls;
      statusBox.textContent = text;

      drawChart(data.recent_rewards || []);
    }

    refresh().catch(err => {
      document.getElementById('status-box').className = 'status bad';
      document.getElementById('status-box').textContent = err.toString();
    });
    setInterval(() => refresh().catch(() => {}), 2000);
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a browser training dashboard.")
    parser.add_argument("--remote", action="store_true", default=True)
    parser.add_argument("--experiment", help="Experiment name to monitor.")
    parser.add_argument("--latest", action="store_true", help="Monitor the newest log when no experiment is supplied.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


class DashboardState:
    def __init__(self, experiment: str | None, latest: bool, remote: bool):
        self.experiment = experiment
        self.latest = latest
        self.remote = remote

    def snapshot(self) -> dict[str, Any]:
        try:
            if self.remote:
                log_path = tp.find_remote_log(self.experiment, True, None)
                text = tp.read_remote_log(log_path, 300)
            else:
                raise RuntimeError("Only remote mode is currently implemented for the dashboard.")

            state = tp.parse_progress(text)
            if self.remote:
                tp.enrich_remote_state(state, self.experiment, log_path)

            frac_iter = (
                state.iteration / state.total_iterations
                if state.iteration is not None and state.total_iterations
                else None
            )
            frac_time = (
                state.elapsed_seconds / state.budget_seconds
                if state.elapsed_seconds is not None and state.budget_seconds
                else None
            )
            progress_fraction = frac_iter if frac_iter is not None else (frac_time or 0.0)
            eta_seconds = (
                int((state.elapsed_seconds / frac_iter) - state.elapsed_seconds)
                if state.elapsed_seconds is not None and frac_iter and frac_iter > 0
                else None
            )
            stale_warning = bool(
                state.log_age_seconds is not None and state.process_running and state.log_age_seconds > 180
            )

            data = asdict(state)
            data.update(
                {
                    "log_path": log_path,
                    "progress_fraction": progress_fraction,
                    "eta_seconds": eta_seconds,
                    "stale_warning": stale_warning,
                    "error": None,
                }
            )
            return data
        except Exception as exc:
            return {
                "status": "error",
                "error": str(exc),
                "experiment": self.experiment,
                "recent_rewards": [],
                "progress_fraction": 0.0,
                "eta_seconds": None,
                "completed": False,
            }


def make_handler(state: DashboardState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                payload = HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if parsed.path == "/api/status":
                payload = json.dumps(state.snapshot()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, fmt, *args):
            return

    return Handler


def launch_browser(url: str) -> None:
    browser = shutil.which("firefox")
    if browser:
        subprocess.Popen([browser, "--new-window", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        webbrowser.open(url)


def main() -> int:
    args = parse_args()
    state = DashboardState(args.experiment, args.latest, args.remote)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{args.port}/"
    print(url, flush=True)
    if not args.no_browser:
        time.sleep(0.4)
        launch_browser(url)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.shutdown()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
