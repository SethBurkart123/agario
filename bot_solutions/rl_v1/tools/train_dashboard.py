"""Live training dashboard for the AlphaZero marathon.

Serves a single-page view designed to answer one question at a glance:
IS IT RUNNING, AND IS IT GETTING BETTER?

  - A status banner: RUNNING / QUIET / STALLED, current phase, progress, ETA.
    Liveness comes from the marathon phase_progress.json file (written
    sub-minute by the gate/generate phases) plus the metrics stream.
  - The strength curve (eval mass + win-rate per cycle) — the only chart that
    matters for "is it getting better".
  - The marathon log tail.
  - Everything else (per-phase metric charts, gate verdicts) lives behind a
    collapsed "training details" section, and empty charts are hidden.

Usage:
  uv run python -m bot_solutions.rl_v1.tools.train_dashboard
  uv run python -m bot_solutions.rl_v1.tools.train_dashboard --host 0.0.0.0 --port 8123

Stdlib only -- no external deps, no CDN assets.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .. import CHECKPOINT_ROOT

DEFAULT_METRICS = str(CHECKPOINT_ROOT / "rl/metrics.jsonl")
DEFAULT_HISTORY = str(CHECKPOINT_ROOT / "marathon/history.jsonl")
DEFAULT_STATUS = str(CHECKPOINT_ROOT / "marathon/status.json")
DEFAULT_LOG = "/tmp/azero_marathon.log"
DEFAULT_PORT = 8123
LOG_TAIL_LINES = 50

# Resolved at startup, read by the request handler.
METRICS_PATH = Path(DEFAULT_METRICS)
HISTORY_PATH = Path(DEFAULT_HISTORY)
STATUS_PATH = Path(DEFAULT_STATUS)
PROGRESS_PATH = CHECKPOINT_ROOT / "marathon/phase_progress.json"
LOG_PATH = Path(DEFAULT_LOG)


def read_metrics(path: Path) -> list[dict]:
    """Parse the JSONL metrics file into a list of dicts.

    Tolerates a partial/corrupt final line (common while the trainer is mid
    write) and silently skips any other unparseable lines.
    """
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        raw = path.read_bytes()
    except OSError:
        return []
    text = raw.decode("utf-8", errors="replace")
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # Most likely the last line was only half-written; skip it.
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def read_status(path: Path) -> dict | None:
    """Parse a small JSON status file, returning None if missing/unparseable."""
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def read_log_tail(path: Path, n: int = LOG_TAIL_LINES) -> list[str]:
    """Return the last n non-blank lines of the log file (empty if missing)."""
    if not path.exists():
        return []
    try:
        raw = path.read_bytes()
    except OSError:
        return []
    text = raw.decode("utf-8", errors="replace")
    lines = [ln.rstrip("\n") for ln in text.split("\n") if ln.strip()]
    return lines[-n:]


def read_marathon() -> dict:
    """Assemble the /marathon payload, tolerating any/all files missing."""
    return {
        "status": read_status(STATUS_PATH),
        "progress": read_status(PROGRESS_PATH),
        "history": read_metrics(HISTORY_PATH),
        "log_tail": read_log_tail(LOG_PATH),
    }


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Marathon Dashboard</title>
<style>
  :root {
    --bg: #0b0e14;
    --panel: #141923;
    --panel-2: #1b212e;
    --border: #232b3a;
    --text: #e6edf3;
    --muted: #8b97a8;
    --accent: #5ad1c8;
    --good: #7ee0a8;
    --warn: #e0af68;
    --bad: #ff8497;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 20px; }

  /* ---- status banner: the one thing that matters ---- */
  .banner {
    display: flex; align-items: center; gap: 18px;
    background: linear-gradient(180deg, var(--panel-2), var(--panel));
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 16px;
  }
  .bdot {
    width: 14px; height: 14px; border-radius: 50%; flex: 0 0 auto;
    background: var(--muted);
  }
  .bdot.run { background: var(--good); box-shadow: 0 0 12px var(--good);
              animation: pulse 2s infinite ease-in-out; }
  .bdot.quiet { background: var(--warn); box-shadow: 0 0 10px var(--warn); }
  .bdot.stall { background: var(--bad); box-shadow: 0 0 10px var(--bad); }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }
  .banner .main { min-width: 0; }
  .b-status {
    font-size: 22px; font-weight: 700; line-height: 1.25;
    font-variant-numeric: tabular-nums;
  }
  .b-status .word.run { color: var(--good); }
  .b-status .word.quiet { color: var(--warn); }
  .b-status .word.stall { color: var(--bad); }
  .b-detail {
    font-size: 13px; color: var(--muted); margin-top: 3px;
    font-variant-numeric: tabular-nums;
  }
  .b-signal {
    margin-left: auto; text-align: right; flex: 0 0 auto;
    font-size: 12px; color: var(--muted);
    font-variant-numeric: tabular-nums;
  }
  .b-signal .big { font-size: 16px; font-weight: 650; color: var(--text); }

  /* progress bar inside the banner */
  .bar {
    height: 6px; border-radius: 3px; background: var(--panel-2);
    border: 1px solid var(--border);
    margin-top: 10px; overflow: hidden; width: 100%;
  }
  .bar > div {
    height: 100%; background: var(--accent); width: 0%;
    transition: width 1s linear;
  }

  /* ---- cards ---- */
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px 16px 10px;
    overflow: hidden;
    margin-bottom: 16px;
  }
  .card h2 {
    font-size: 12px; font-weight: 600; margin: 0 0 2px;
    color: var(--muted); letter-spacing: 0.5px;
  }
  .card .latest {
    font-size: 24px; font-weight: 680;
    font-variant-numeric: tabular-nums; line-height: 1.1;
  }
  .card .sub { font-size: 11px; color: var(--muted); margin-top: 1px; }
  canvas { display: block; width: 100%; }
  .empty { color: var(--muted); font-size: 13px; padding: 30px 0; text-align: center; }

  /* ---- log ---- */
  .log-head { display: flex; align-items: center; cursor: pointer; user-select: none; }
  .log-head h2 { margin: 0; }
  .log-toggle { color: var(--muted); font-size: 11px; margin-left: auto; }
  .log-body {
    margin-top: 10px;
    background: #0a0d12; border: 1px solid var(--border); border-radius: 10px;
    padding: 10px 12px;
    height: 240px; overflow-y: auto;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11.5px; line-height: 1.5; color: #b9c4d4;
    white-space: pre-wrap; word-break: break-word;
  }
  .log-body.collapsed { display: none; }

  /* ---- collapsed details ---- */
  details.more { margin-bottom: 16px; }
  details.more > summary {
    cursor: pointer; user-select: none; list-style: none;
    font-size: 12px; font-weight: 600; letter-spacing: 0.8px;
    text-transform: uppercase; color: var(--muted);
    padding: 10px 4px;
  }
  details.more > summary::before { content: "▸ "; }
  details.more[open] > summary::before { content: "▾ "; }
  .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; }
  .grid .card { margin-bottom: 0; }
  .gate-chip {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 5px 10px; border-radius: 8px; margin: 3px 4px 3px 0;
    background: var(--panel-2); border: 1px solid var(--border);
    font-size: 11.5px; font-variant-numeric: tabular-nums;
  }
  .gate-chip.pass { color: var(--good); }
  .gate-chip.fail { color: var(--bad); }
  footer { color: var(--muted); font-size: 11px; text-align: center; margin: 14px 0 4px; }
  @media (max-width: 700px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="wrap">

  <div class="banner">
    <div class="bdot" id="bdot"></div>
    <div class="main" style="flex:1;">
      <div class="b-status" id="b-status">connecting&hellip;</div>
      <div class="b-detail" id="b-detail"></div>
      <div class="bar" id="b-bar" style="display:none;"><div id="b-bar-fill"></div></div>
    </div>
    <div class="b-signal">
      <div>last signal</div>
      <div class="big" id="b-signal">-</div>
    </div>
  </div>

  <div class="card">
    <h2>STRENGTH &middot; eval mass (solid) &amp; win-rate (dashed) per cycle</h2>
    <div class="latest" id="v-strength">-</div><div class="sub" id="u-strength"></div>
    <canvas id="c-strength"></canvas>
  </div>

  <div class="card">
    <div class="log-head" id="log-head">
      <h2>MARATHON LOG</h2>
      <span class="log-toggle" id="log-toggle">click to expand</span>
    </div>
    <div class="log-body collapsed" id="log-body"><span class="empty">no log output yet</span></div>
  </div>

  <details class="more" id="details">
    <summary>Training details</summary>
    <div id="gate-row" style="margin: 0 2px 12px;"></div>
    <div class="grid" id="grid"></div>
  </details>

  <footer>polling /metrics + /marathon every 3s &middot; offline &middot; stdlib only</footer>
</div>

<script>
"use strict";

// Charts shown under "training details" — a card only appears if the metrics
// stream actually contains that key.
const CHARTS = [
  { key: "search_labels_per_sec", title: "SEARCH LABELS / SEC (GENERATE)", color: "#ffd166", decimals: 1 },
  { key: "imagined_decisions_per_sec", title: "IMAGINED DECISIONS / SEC", color: "#06d6a0", decimals: 0 },
  { key: "bc_loss",        title: "DISTILL LOSS",        color: "#c0caf5", decimals: 4 },
  { key: "bc_turn_acc",    title: "DISTILL TURN ACC",    color: "#73daca", decimals: 3 },
  { key: "bc_split_recall",title: "DISTILL SPLIT RECALL",color: "#ff7a93", decimals: 3 },
  { key: "ep_mass",        title: "EP MASS",             color: "#5ad1c8", decimals: 1 },
  { key: "ep_kills",       title: "EP KILLS",            color: "#ff9e64", decimals: 2 },
  { key: "steps_per_sec",  title: "STEPS / SEC",         color: "#9ece6a", decimals: 0 },
  { key: "entropy",        title: "ENTROPY",             color: "#bb9af7", decimals: 3 },
  { key: "value_loss",     title: "VALUE LOSS",          color: "#f7768e", decimals: 4 },
  { key: "kl",             title: "KL TO CLONE",         color: "#f0c674", decimals: 4 },
];

const DPR = Math.max(1, window.devicePixelRatio || 1);
const MAX_POINTS = 400;

const PHASE_LABEL = {
  gate: "gate check (search vs raw)",
  generate: "generating self-play games",
  distill: "distilling into the network",
  eval: "evaluating vs solo_smart",
  exit: "self-play generation",
};

function fmtNum(v, dec) {
  if (v == null || !isFinite(v)) return "-";
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(2) + "M";
  if (a >= 1e4) return Math.round(v).toLocaleString();
  return v.toFixed(dec);
}
function fmtDuration(sec) {
  if (sec == null || !isFinite(sec) || sec < 0) return "-";
  sec = Math.floor(sec);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  if (h > 0) return h + "h " + m + "m";
  if (m > 0) return m + "m " + s + "s";
  return s + "s";
}
function fmtEta(min) {
  if (min == null || !isFinite(min) || min < 0) return null;
  if (min >= 90) return (min / 60).toFixed(1) + "h";
  return Math.round(min) + "m";
}

function series(data, key) {
  const pts = [];
  for (const d of data) {
    const y = d[key];
    if (y == null || typeof y !== "number" || !isFinite(y)) continue;
    const x = (typeof d.update === "number") ? d.update : pts.length;
    pts.push({ x, y });
  }
  return pts;
}

function downsample(pts, target) {
  if (pts.length <= target) return pts;
  const out = [];
  const stride = pts.length / target;
  out.push(pts[0]);
  for (let i = 1; i < target - 1; i++) out.push(pts[Math.floor(i * stride)]);
  out.push(pts[pts.length - 1]);
  return out;
}

function niceTicks(min, max, count) {
  if (min === max) { min -= 1; max += 1; }
  const range = max - min;
  const rawStep = range / count;
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const norm = rawStep / mag;
  let step;
  if (norm < 1.5) step = 1; else if (norm < 3) step = 2;
  else if (norm < 7) step = 5; else step = 10;
  step *= mag;
  const start = Math.ceil(min / step) * step;
  const ticks = [];
  for (let v = start; v <= max + step * 1e-6; v += step) ticks.push(v);
  return ticks;
}

function tickLabel(v) {
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(1) + "M";
  if (a >= 1e3) return (v / 1e3).toFixed(1) + "K";
  if (a === 0) return "0";
  if (a < 0.01) return v.toExponential(1);
  if (a < 1) return v.toFixed(3);
  if (a < 100) return v.toFixed(2);
  return Math.round(v).toString();
}

function drawChart(canvas, pts, color) {
  const cssH = canvas.dataset.h ? +canvas.dataset.h : 0;
  const cssW = canvas.clientWidth;
  if (!cssW) return;
  const W = Math.round(cssW * DPR);
  const H = Math.round(cssH * DPR);
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, W, H);

  const padL = 52 * DPR, padR = 12 * DPR, padT = 8 * DPR, padB = 22 * DPR;
  const plotW = W - padL - padR, plotH = H - padT - padB;

  ctx.font = (11 * DPR) + "px -apple-system, sans-serif";
  ctx.textBaseline = "middle";

  if (!pts.length) return;

  let minY = Infinity, maxY = -Infinity, minX = Infinity, maxX = -Infinity;
  for (const p of pts) {
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
  }
  if (minY === maxY) { minY -= Math.abs(minY) * 0.1 || 1; maxY += Math.abs(maxY) * 0.1 || 1; }
  const padY = (maxY - minY) * 0.08;
  minY -= padY; maxY += padY;
  if (minX === maxX) maxX = minX + 1;

  const sx = (x) => padL + (x - minX) / (maxX - minX) * plotW;
  const sy = (y) => padT + (1 - (y - minY) / (maxY - minY)) * plotH;

  const yticks = niceTicks(minY, maxY, 4);
  ctx.strokeStyle = "#1f2632";
  ctx.lineWidth = 1 * DPR;
  ctx.fillStyle = "#8b97a8";
  ctx.textAlign = "right";
  for (const t of yticks) {
    if (t < minY || t > maxY) continue;
    const y = sy(t);
    ctx.beginPath();
    ctx.moveTo(padL, y); ctx.lineTo(W - padR, y);
    ctx.stroke();
    ctx.fillText(tickLabel(t), padL - 6 * DPR, y);
  }

  const xticks = niceTicks(minX, maxX, 5);
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const t of xticks) {
    if (t < minX || t > maxX) continue;
    ctx.fillText(tickLabel(t), sx(t), H - padB + 6 * DPR);
  }
  ctx.textBaseline = "middle";

  const grad = ctx.createLinearGradient(0, padT, 0, padT + plotH);
  grad.addColorStop(0, color + "33");
  grad.addColorStop(1, color + "00");
  ctx.beginPath();
  ctx.moveTo(sx(pts[0].x), sy(pts[0].y));
  for (const p of pts) ctx.lineTo(sx(p.x), sy(p.y));
  ctx.lineTo(sx(pts[pts.length - 1].x), padT + plotH);
  ctx.lineTo(sx(pts[0].x), padT + plotH);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  ctx.beginPath();
  ctx.moveTo(sx(pts[0].x), sy(pts[0].y));
  for (const p of pts) ctx.lineTo(sx(p.x), sy(p.y));
  ctx.strokeStyle = color;
  ctx.lineWidth = 2 * DPR;
  ctx.lineJoin = "round";
  ctx.stroke();

  const last = pts[pts.length - 1];
  ctx.beginPath();
  ctx.arc(sx(last.x), sy(last.y), 3.2 * DPR, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
}

// Two-line chart: mass (left axis, filled accent) + win_rate% (right axis).
function drawStrength(canvas, evals) {
  const cssH = canvas.dataset.h ? +canvas.dataset.h : 0;
  const cssW = canvas.clientWidth;
  if (!cssW) return;
  const W = Math.round(cssW * DPR), H = Math.round(cssH * DPR);
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  ctx.font = (11 * DPR) + "px -apple-system, sans-serif";
  ctx.textBaseline = "middle";

  if (!evals.length) {
    ctx.fillStyle = "#8b97a8"; ctx.textAlign = "center";
    ctx.fillText("waiting for eval data...", W / 2, H / 2);
    return;
  }

  const massPts = [], winPts = [];
  for (const e of evals) {
    const x = (typeof e.cycle === "number") ? e.cycle : massPts.length + 1;
    if (typeof e.mass === "number" && isFinite(e.mass)) massPts.push({ x, y: e.mass });
    if (typeof e.win_rate === "number" && isFinite(e.win_rate)) winPts.push({ x, y: e.win_rate });
  }
  if (!massPts.length && !winPts.length) {
    ctx.fillStyle = "#8b97a8"; ctx.textAlign = "center";
    ctx.fillText("waiting for eval data...", W / 2, H / 2);
    return;
  }

  const padL = 52 * DPR, padR = 46 * DPR, padT = 10 * DPR, padB = 22 * DPR;
  const plotW = W - padL - padR, plotH = H - padT - padB;

  let minX = Infinity, maxX = -Infinity;
  for (const p of massPts.concat(winPts)) {
    if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
  }
  if (minX === maxX) maxX = minX + 1;
  let minM = Infinity, maxM = -Infinity;
  for (const p of massPts) { if (p.y < minM) minM = p.y; if (p.y > maxM) maxM = p.y; }
  if (!isFinite(minM)) { minM = 0; maxM = 1; }
  if (minM === maxM) { minM -= Math.abs(minM) * 0.1 || 1; maxM += Math.abs(maxM) * 0.1 || 1; }
  const padM = (maxM - minM) * 0.08; minM -= padM; maxM += padM;

  const sx = (x) => padL + (x - minX) / (maxX - minX) * plotW;
  const syM = (y) => padT + (1 - (y - minM) / (maxM - minM)) * plotH;
  const syW = (y) => padT + (1 - y / 100) * plotH; // win_rate 0..100 right axis

  const yticks = niceTicks(minM, maxM, 4);
  ctx.strokeStyle = "#1f2632"; ctx.lineWidth = 1 * DPR;
  ctx.fillStyle = "#8b97a8"; ctx.textAlign = "right";
  for (const t of yticks) {
    if (t < minM || t > maxM) continue;
    const y = syM(t);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillText(tickLabel(t), padL - 6 * DPR, y);
  }
  ctx.fillStyle = "#ff9e64"; ctx.textAlign = "left";
  for (const t of [0, 25, 50, 75, 100]) {
    ctx.fillText(t + "%", W - padR + 6 * DPR, syW(t));
  }
  const xticks = niceTicks(minX, maxX, 5);
  ctx.fillStyle = "#8b97a8"; ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (const t of xticks) {
    if (t < minX || t > maxX) continue;
    ctx.fillText(Math.round(t).toString(), sx(t), H - padB + 6 * DPR);
  }
  ctx.textBaseline = "middle";

  if (winPts.length) {
    ctx.beginPath();
    ctx.moveTo(sx(winPts[0].x), syW(winPts[0].y));
    for (const p of winPts) ctx.lineTo(sx(p.x), syW(p.y));
    ctx.strokeStyle = "#ff9e64"; ctx.lineWidth = 1.6 * DPR;
    ctx.setLineDash([5 * DPR, 4 * DPR]); ctx.lineJoin = "round"; ctx.stroke();
    ctx.setLineDash([]);
  }

  if (massPts.length) {
    const color = "#5ad1c8";
    const grad = ctx.createLinearGradient(0, padT, 0, padT + plotH);
    grad.addColorStop(0, color + "33"); grad.addColorStop(1, color + "00");
    ctx.beginPath();
    ctx.moveTo(sx(massPts[0].x), syM(massPts[0].y));
    for (const p of massPts) ctx.lineTo(sx(p.x), syM(p.y));
    ctx.lineTo(sx(massPts[massPts.length - 1].x), padT + plotH);
    ctx.lineTo(sx(massPts[0].x), padT + plotH);
    ctx.closePath(); ctx.fillStyle = grad; ctx.fill();

    ctx.beginPath();
    ctx.moveTo(sx(massPts[0].x), syM(massPts[0].y));
    for (const p of massPts) ctx.lineTo(sx(p.x), syM(p.y));
    ctx.strokeStyle = color; ctx.lineWidth = 2 * DPR; ctx.lineJoin = "round"; ctx.stroke();

    const last = massPts[massPts.length - 1];
    ctx.beginPath(); ctx.arc(sx(last.x), syM(last.y), 3.2 * DPR, 0, Math.PI * 2);
    ctx.fillStyle = color; ctx.fill();
  }
}

// ===================== STATUS BANNER =====================

let lastData = [];
let lastMarathon = { status: null, progress: null, history: [], log_tail: [] };
let logCollapsed = true;
// Client-side liveness from the metrics stream: epoch (ms) when we last saw
// the newest metrics row change. Covers phases that stream rows frequently
// (distill) but don't touch the heartbeat file.
let lastMetricsChangeAt = 0;
let lastMetricsFingerprint = "";

function renderBanner() {
  const st = lastMarathon.status;
  const pr = lastMarathon.progress;
  const now = Date.now() / 1000;

  // Freshest liveness signal wins.
  let age = Infinity;
  if (pr && typeof pr.updated_at === "number") age = now - pr.updated_at;
  if (lastMetricsChangeAt > 0) age = Math.min(age, now - lastMetricsChangeAt / 1000);

  let cls, word;
  if (!st && !pr && age === Infinity) { cls = ""; word = "NO DATA"; }
  else if (age < 180) { cls = "run"; word = "RUNNING"; }
  else if (age < 1800) { cls = "quiet"; word = "QUIET"; }
  else { cls = "stall"; word = "STALLED"; }

  // Phase label: heartbeat phase if it's plausibly current, else the
  // marathon stage from status.json.
  let phase = null;
  if (pr && pr.phase && age < 3600) phase = pr.phase;
  else if (st && st.stage) phase = st.stage;
  const phaseTxt = phase ? (PHASE_LABEL[phase] || phase) : "";

  let line = '<span class="word ' + cls + '">' + word + "</span>";
  if (st && st.cycle != null) line += " &middot; cycle " + st.cycle;
  if (phaseTxt) line += " &middot; " + phaseTxt;
  if (pr && typeof pr.fraction === "number" && cls === "run") {
    line += " &middot; " + Math.round(100 * pr.fraction) + "%";
    const eta = fmtEta(pr.eta_minutes);
    if (eta) line += " <span style='color:var(--muted);font-weight:500;'>(~" + eta + " left)</span>";
  }
  document.getElementById("b-status").innerHTML = line;

  const parts = [];
  if (pr && pr.detail && cls === "run") parts.push(pr.detail);
  if (st && typeof st.stage_started_at === "number")
    parts.push("in this stage for " + fmtDuration(now - st.stage_started_at));
  if (st && st.best_score != null)
    parts.push("best score " + Math.round(st.best_score).toLocaleString());
  if (word === "QUIET")
    parts.push("eval and restarts can be quiet for a few minutes — only worry if this goes red");
  if (word === "STALLED")
    parts.push("no heartbeat or metrics for " + fmtDuration(age) + " — check the log below");
  document.getElementById("b-detail").textContent = parts.join("  ·  ");

  const bar = document.getElementById("b-bar");
  if (pr && typeof pr.fraction === "number" && cls === "run") {
    bar.style.display = "";
    document.getElementById("b-bar-fill").style.width =
      Math.max(0, Math.min(100, 100 * pr.fraction)) + "%";
  } else {
    bar.style.display = "none";
  }

  document.getElementById("b-signal").textContent =
    age === Infinity ? "-" : fmtDuration(age) + " ago";
  const dot = document.getElementById("bdot");
  dot.className = "bdot" + (cls ? " " + cls : "");
}

// ===================== DETAILS CHARTS =====================

// Cards are created lazily, only for keys present in the data.
const cardEls = {};

function ensureCard(c) {
  if (cardEls[c.key]) return cardEls[c.key];
  const div = document.createElement("div");
  div.className = "card";
  div.innerHTML =
    "<h2>" + c.title + "</h2>" +
    '<div class="latest" id="v-' + c.key + '">-</div>' +
    '<div class="sub" id="u-' + c.key + '"></div>' +
    '<canvas id="c-' + c.key + '"></canvas>';
  document.getElementById("grid").appendChild(div);
  cardEls[c.key] = div;
  return div;
}

function renderDetails(data) {
  for (const c of CHARTS) {
    let pts = series(data, c.key);
    if (!pts.length) {
      if (cardEls[c.key]) cardEls[c.key].style.display = "none";
      continue;
    }
    const card = ensureCard(c);
    card.style.display = "";
    const canvas = document.getElementById("c-" + c.key);
    canvas.dataset.h = 140; canvas.style.height = "140px";
    const lp = pts[pts.length - 1];
    document.getElementById("v-" + c.key).textContent = fmtNum(lp.y, c.decimals);
    document.getElementById("u-" + c.key).textContent = "update " + lp.x;
    drawChart(canvas, downsample(pts, MAX_POINTS), c.color);
  }
}

function renderGates(gates) {
  const row = document.getElementById("gate-row");
  if (!gates.length) { row.innerHTML = ""; return; }
  const html = ['<span style="font-size:11px;color:var(--muted);margin-right:6px;">GATE (monitoring only):</span>'];
  for (const g of gates.slice(-8)) {
    const pass = !!g.passed;
    const ratio = (typeof g.ratio === "number") ? g.ratio.toFixed(2) : "?";
    html.push(
      '<span class="gate-chip ' + (pass ? "pass" : "fail") + '">' +
      (g.cycle != null ? "c" + g.cycle + " " : "") + ratio + "</span>"
    );
  }
  row.innerHTML = html.join("");
}

// ===================== MAIN RENDER =====================

function renderMarathon(m) {
  const evals = [], gates = [];
  for (const r of (m.history || [])) {
    if ((r.type || "eval") === "gate") gates.push(r); else evals.push(r);
  }

  const canvas = document.getElementById("c-strength");
  canvas.dataset.h = 260; canvas.style.height = "260px";
  drawStrength(canvas, evals);
  const lastEval = evals.length ? evals[evals.length - 1] : null;
  if (lastEval) {
    document.getElementById("v-strength").textContent =
      (lastEval.mass != null ? fmtNum(lastEval.mass, 1) : "-");
    document.getElementById("u-strength").textContent =
      "cycle " + (lastEval.cycle != null ? lastEval.cycle : "?") +
      (lastEval.win_rate != null ? " · win " + lastEval.win_rate + "%" : "");
  } else {
    document.getElementById("v-strength").textContent = "-";
    document.getElementById("u-strength").textContent = "";
  }

  renderGates(gates);

  const body = document.getElementById("log-body");
  const lines = m.log_tail || [];
  const atBottom = body.scrollHeight - body.scrollTop - body.clientHeight < 30;
  if (!lines.length) {
    body.innerHTML = '<span class="empty">no log output yet</span>';
  } else {
    body.textContent = lines.join("\n"); // textContent avoids HTML injection
  }
  if (atBottom) body.scrollTop = body.scrollHeight;
}

document.getElementById("log-head").addEventListener("click", () => {
  logCollapsed = !logCollapsed;
  document.getElementById("log-body").classList.toggle("collapsed", logCollapsed);
  document.getElementById("log-toggle").textContent =
    logCollapsed ? "click to expand" : "click to collapse";
});

async function pollMarathon() {
  try {
    const r = await fetch("/marathon", { cache: "no-store" });
    if (r.ok) {
      lastMarathon = await r.json();
      renderMarathon(lastMarathon);
      renderBanner();
    }
  } catch (e) { /* keep last render */ }
}

async function poll() {
  try {
    const r = await fetch("/metrics", { cache: "no-store" });
    if (r.ok) {
      const data = await r.json();
      lastData = data;
      if (data.length) {
        const last = data[data.length - 1];
        const fp = JSON.stringify(last);
        if (fp !== lastMetricsFingerprint) {
          lastMetricsFingerprint = fp;
          lastMetricsChangeAt = Date.now();
        }
      }
      renderDetails(data);
    }
  } catch (e) { /* keep last render */ }
}

window.addEventListener("resize", () => {
  renderDetails(lastData); renderMarathon(lastMarathon); renderBanner();
});
poll();
pollMarathon();
setInterval(poll, 3000);
setInterval(pollMarathon, 3000);
setInterval(renderBanner, 1000); // tick ages between polls
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet console
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path.startswith("/metrics"):
            data = read_metrics(METRICS_PATH)
            body = json.dumps(data).encode("utf-8")
            self._send(200, body, "application/json")
        elif self.path.startswith("/marathon"):
            body = json.dumps(read_marathon()).encode("utf-8")
            self._send(200, body, "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    do_HEAD = do_GET


def main(argv: list[str] | None = None) -> int:
    global METRICS_PATH, HISTORY_PATH, STATUS_PATH, PROGRESS_PATH, LOG_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (0.0.0.0 to serve over the network)")
    parser.add_argument("--metrics", default=DEFAULT_METRICS,
                        help="path to metrics.jsonl")
    parser.add_argument("--history", default=DEFAULT_HISTORY,
                        help="path to marathon history.jsonl (eval + gate rows)")
    parser.add_argument("--status", default=DEFAULT_STATUS,
                        help="path to marathon status.json")
    parser.add_argument("--log", default=DEFAULT_LOG,
                        help="path to the marathon stage log file")
    args = parser.parse_args(argv)

    METRICS_PATH = Path(args.metrics)
    HISTORY_PATH = Path(args.history)
    STATUS_PATH = Path(args.status)
    PROGRESS_PATH = STATUS_PATH.parent / "phase_progress.json"
    LOG_PATH = Path(args.log)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"Training dashboard serving at {url}")
    print(f"Reading metrics from {METRICS_PATH}")
    print(f"Reading marathon state from {STATUS_PATH}, {HISTORY_PATH}, {LOG_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
