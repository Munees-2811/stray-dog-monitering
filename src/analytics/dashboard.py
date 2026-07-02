"""
Analytics dashboard generator.

Aggregates the session files written by ``SessionRecorder`` into a single
self-contained HTML dashboard (inline CSS + vanilla-JS SVG charts, no CDN —
it works fully offline). Open it from the desktop app ("Analytics" section)
or generate it headless:

    python -m src.analytics.dashboard            # writes outputs/dashboard.html

Views:
    - KPI row ............ alerts, dogs, persons, frames, peak risk / avg FPS
    - Risk over time ..... per-session risk timeline w/ threshold + alert marks
                           (peak risk per session when "All sessions" selected)
    - Detections ......... dogs vs persons (2-series line / grouped columns)
    - Alerts by hour ..... when dogs get dangerous during the day
    - Risk distribution .. histogram of observed risk
    - Alert signals ...... which risk features actually fire alerts
    - Sessions table ..... every monitoring run with its stats

A session filter above the charts scopes every view, and each chart carries a
matching data table so no value is gated behind hover or color.
"""

import json
import webbrowser
from datetime import datetime
from pathlib import Path

from src.config import PROJECT_ROOT, load_config
from src.analytics.recorder import load_sessions

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stray Dog Monitoring — Analytics</title>
<style>
  .viz-root {
    --surface-1: #fcfcfb;
    --page: #f9f9f7;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --text-muted: #898781;
    --grid: #e1e0d9;
    --baseline: #c3c2b7;
    --border: rgba(11,11,11,0.10);
    --series-1: #2a78d6;   /* blue  — risk / dogs / sequential hue */
    --series-2: #1baf7a;   /* aqua  — persons */
    --critical: #d03b3b;   /* status: alerts, threshold */
  }
  @media (prefers-color-scheme: dark) {
    .viz-root {
      --surface-1: #1a1a19;
      --page: #0d0d0d;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted: #898781;
      --grid: #2c2c2a;
      --baseline: #383835;
      --border: rgba(255,255,255,0.10);
      --series-1: #3987e5;
      --series-2: #199e70;
      --critical: #d03b3b;
    }
  }
  * { box-sizing: border-box; }
  body.viz-root {
    margin: 0; padding: 24px;
    background: var(--page); color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    font-size: 14px;
  }
  header { margin-bottom: 16px; }
  header h1 { font-size: 20px; font-weight: 650; margin: 0 0 2px; }
  header p  { margin: 0; color: var(--text-secondary); font-size: 13px; }

  .filter-row {
    display: flex; align-items: center; gap: 10px;
    margin: 16px 0; flex-wrap: wrap;
  }
  .filter-row label { color: var(--text-secondary); font-size: 13px; }
  .filter-row select {
    background: var(--surface-1); color: var(--text-primary);
    border: 1px solid var(--border); border-radius: 8px;
    padding: 6px 10px; font: inherit; font-size: 13px;
  }

  .kpis {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px; margin-bottom: 16px;
  }
  .tile {
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 12px; padding: 14px 16px;
  }
  .tile .label { color: var(--text-secondary); font-size: 12px; margin-bottom: 6px; }
  .tile .value { font-size: 26px; font-weight: 600; line-height: 1.1; }
  .tile .sub   { color: var(--text-muted); font-size: 11px; margin-top: 4px; }

  .grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
  }
  .card {
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px; min-width: 0;
  }
  .card.full { grid-column: 1 / -1; }
  @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }
  .card h2 { font-size: 14px; font-weight: 600; margin: 0 0 2px; }
  .card .sub { color: var(--text-muted); font-size: 12px; margin: 0 0 10px; }
  .chart { position: relative; }
  .chart svg { display: block; width: 100%; }

  .legend { display: flex; gap: 14px; margin: 8px 0 0; font-size: 12px;
            color: var(--text-secondary); flex-wrap: wrap; }
  .legend .key { display: inline-flex; align-items: center; gap: 6px; }
  .legend .swatch-line { width: 14px; height: 0; border-top: 2px solid; border-radius: 2px; }
  .legend .swatch-rect { width: 10px; height: 10px; border-radius: 3px; }

  .tooltip {
    position: fixed; pointer-events: none; z-index: 10; display: none;
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px 10px; font-size: 12px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.18); min-width: 110px;
  }
  .tooltip .tt-title { color: var(--text-muted); font-size: 11px; margin-bottom: 4px; }
  .tooltip .tt-row { display: flex; align-items: center; gap: 6px; margin-top: 2px; }
  .tooltip .tt-key { width: 12px; height: 0; border-top: 2px solid; border-radius: 2px; }
  .tooltip .tt-val { font-weight: 600; }
  .tooltip .tt-name { color: var(--text-secondary); }

  details.data-table { margin-top: 10px; }
  details.data-table summary {
    cursor: pointer; color: var(--text-muted); font-size: 12px; user-select: none;
  }
  table { border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 12px; }
  th { text-align: left; color: var(--text-secondary); font-weight: 600;
       border-bottom: 1px solid var(--baseline); padding: 5px 8px; }
  td { border-bottom: 1px solid var(--grid); padding: 5px 8px;
       font-variant-numeric: tabular-nums; }
  .num { text-align: right; }
  .empty { color: var(--text-muted); padding: 30px 0; text-align: center; font-size: 13px; }
  footer { color: var(--text-muted); font-size: 11px; margin-top: 18px; }
</style>
</head>
<body class="viz-root">
<header>
  <h1>Stray Dog Monitoring — Analytics</h1>
  <p>Aggression-risk detections, alerts and model performance across monitoring sessions.</p>
</header>

<div class="filter-row">
  <label for="sessionSel">Session</label>
  <select id="sessionSel"></select>
  <span id="scopeNote" style="color:var(--text-muted);font-size:12px;"></span>
</div>

<div class="kpis" id="kpis"></div>

<div class="grid">
  <div class="card full">
    <h2 id="riskTitle">Risk over time</h2>
    <p class="sub" id="riskSub"></p>
    <div class="chart" id="riskChart"></div>
    <details class="data-table"><summary>Data table</summary>
      <div id="riskTable"></div></details>
  </div>

  <div class="card">
    <h2>Detections — dogs vs persons</h2>
    <p class="sub" id="detSub"></p>
    <div class="chart" id="detChart"></div>
    <div class="legend" id="detLegend"></div>
    <details class="data-table"><summary>Data table</summary>
      <div id="detTable"></div></details>
  </div>

  <div class="card">
    <h2>Alerts by hour of day</h2>
    <p class="sub">When monitored dogs turned dangerous</p>
    <div class="chart" id="hourChart"></div>
    <details class="data-table"><summary>Data table</summary>
      <div id="hourTable"></div></details>
  </div>

  <div class="card">
    <h2>Risk distribution</h2>
    <p class="sub">Share of processed frames by peak risk score</p>
    <div class="chart" id="histChart"></div>
    <details class="data-table"><summary>Data table</summary>
      <div id="histTable"></div></details>
  </div>

  <div class="card">
    <h2>Alert signals</h2>
    <p class="sub">Average contribution of each risk feature at alert time</p>
    <div class="chart" id="featChart"></div>
    <details class="data-table"><summary>Data table</summary>
      <div id="featTable"></div></details>
  </div>

  <div class="card full">
    <h2>Sessions</h2>
    <p class="sub">Every monitoring run and its outcome</p>
    <div id="sessionsTable"></div>
  </div>
</div>

<div class="tooltip" id="tooltip"></div>
<footer id="genNote"></footer>

<script>
"use strict";
const SESSIONS = __SESSIONS_JSON__;
const GENERATED = "__GENERATED__";

// ── helpers ──────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}
function css(name) {
  return getComputedStyle(document.body).getPropertyValue(name).trim();
}
function fmt(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e4) return (n / 1e3).toFixed(1) + "K";
  return Math.round(n).toLocaleString("en-US");
}
function mmss(t) {
  const m = Math.floor(t / 60), s = Math.floor(t % 60);
  return String(m).padStart(2, "0") + ":" + String(s).padStart(2, "0");
}
function niceMax(v) {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}
// integer-friendly max for count axes: divisible by 4 so ticks are whole
function niceMaxInt(v) { return Math.max(4, Math.ceil(v / 4) * 4); }
// "20260702_120000" -> "07-02 12:00"
function sessLabel(id) {
  return id.slice(4, 6) + "-" + id.slice(6, 8) + " " +
         id.slice(9, 11) + ":" + id.slice(11, 13);
}
function textNode(parent, tag, textVal, attrs) {
  const e = el(tag, attrs);
  e.textContent = textVal;   // labels are untrusted data — textContent only
  parent.appendChild(e);
  return e;
}
function buildTable(container, headers, rows) {
  container.replaceChildren();
  const t = document.createElement("table");
  const trh = document.createElement("tr");
  headers.forEach((h, i) => {
    const th = document.createElement("th");
    th.textContent = h;
    if (i > 0) th.className = "num";
    trh.appendChild(th);
  });
  t.appendChild(trh);
  rows.forEach(r => {
    const tr = document.createElement("tr");
    r.forEach((c, i) => {
      const td = document.createElement("td");
      td.textContent = c;
      if (i > 0) td.className = "num";
      tr.appendChild(td);
    });
    t.appendChild(tr);
  });
  container.appendChild(t);
}

// ── tooltip ──────────────────────────────────────────────────────────
const tip = $("tooltip");
function showTip(x, y, title, rows) {
  tip.replaceChildren();
  const tt = document.createElement("div");
  tt.className = "tt-title"; tt.textContent = title;
  tip.appendChild(tt);
  rows.forEach(([name, value, color]) => {
    const row = document.createElement("div"); row.className = "tt-row";
    if (color) {
      const k = document.createElement("span");
      k.className = "tt-key"; k.style.borderTopColor = color;
      row.appendChild(k);
    }
    const v = document.createElement("span");
    v.className = "tt-val"; v.textContent = value;
    row.appendChild(v);
    const n = document.createElement("span");
    n.className = "tt-name"; n.textContent = name;
    row.appendChild(n);
    tip.appendChild(row);
  });
  tip.style.display = "block";
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let px = x + 14, py = y - h - 10;
  if (px + w > innerWidth - 8) px = x - w - 14;
  if (py < 8) py = y + 14;
  tip.style.left = px + "px"; tip.style.top = py + "px";
}
function hideTip() { tip.style.display = "none"; }

// ── chart primitives ─────────────────────────────────────────────────
const M = { top: 14, right: 16, bottom: 26, left: 40 };

function frame(container, height) {
  container.replaceChildren();
  const width = Math.max(320, container.clientWidth);
  const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}` });
  container.appendChild(svg);
  return { svg, width, height,
           iw: width - M.left - M.right, ih: height - M.top - M.bottom };
}
function yGrid(svg, iw, ih, yMax, unit) {
  const steps = 4;
  for (let i = 0; i <= steps; i++) {
    const v = yMax * i / steps;
    const y = M.top + ih - (ih * i / steps);
    if (i > 0) svg.appendChild(el("line", {
      x1: M.left, x2: M.left + iw, y1: y, y2: y,
      stroke: css("--grid"), "stroke-width": 1 }));
    const tick = unit === "pct" ? Math.round(v * 100) + "%"
      : (yMax <= 2 ? v.toFixed(2) : fmt(v));   // decimals for 0–1 risk axes
    textNode(svg, "text", tick, {
      x: M.left - 6, y: y + 4, "text-anchor": "end",
      fill: css("--text-muted"), "font-size": 11 });
  }
  svg.appendChild(el("line", {   // baseline
    x1: M.left, x2: M.left + iw, y1: M.top + ih, y2: M.top + ih,
    stroke: css("--baseline"), "stroke-width": 1 }));
}
// column with 4px rounded data-end, square at the baseline
function columnPath(x, y, w, h) {
  const r = Math.min(4, w / 2, h);
  if (h <= 0) return "";
  return `M ${x} ${y + h} L ${x} ${y + r} Q ${x} ${y} ${x + r} ${y}` +
         ` L ${x + w - r} ${y} Q ${x + w} ${y} ${x + w} ${y + r} L ${x + w} ${y + h} Z`;
}

// generic column chart: cats[], series = [{name,color,values[]}]
function columnChart(container, cats, series, opts) {
  opts = opts || {};
  const { svg, iw, ih } = frame(container, opts.height || 230);
  const vMax = Math.max(1, ...series.flatMap(s => s.values));
  const yMax = opts.pct ? 1 : (opts.intTicks ? niceMaxInt(vMax) : niceMax(vMax));
  yGrid(svg, iw, ih, yMax, opts.pct ? "pct" : "");
  const n = cats.length, ns = series.length;
  const band = iw / n;
  const barW = Math.min(24, Math.max(3, (band - 8) / ns - 2));
  const groupW = barW * ns + 2 * (ns - 1);   // 2px surface gap between bars

  cats.forEach((cat, i) => {
    const gx = M.left + band * i + (band - groupW) / 2;
    series.forEach((s, si) => {
      const v = s.values[i];
      const h = yMax > 0 ? (v / yMax) * ih : 0;
      const x = gx + si * (barW + 2);
      const y = M.top + ih - h;
      const p = el("path", { d: columnPath(x, y, barW, h), fill: s.color });
      svg.appendChild(p);
      // hit target bigger than the mark — full band strip
      const hit = el("rect", { x: gx - 2, y: M.top, width: groupW + 4, height: ih,
                               fill: "transparent" });
      svg.appendChild(hit);
      const move = (ev) => {
        showTip(ev.clientX, ev.clientY, opts.catLabel ? opts.catLabel(cat) : cat,
          series.map(ss => [ss.name,
            opts.pct ? (ss.values[i] * 100).toFixed(1) + "%" : fmt(ss.values[i]),
            ss.color]));
        p.setAttribute("opacity", "0.82");
      };
      hit.addEventListener("pointermove", move);
      hit.addEventListener("pointerleave", () => {
        hideTip(); p.setAttribute("opacity", "1");
      });
    });
    if (!opts.tickEvery || i % opts.tickEvery === 0) {
      textNode(svg, "text", opts.catLabel ? opts.catLabel(cat) : cat, {
        x: M.left + band * i + band / 2, y: M.top + ih + 16,
        "text-anchor": "middle", fill: css("--text-muted"), "font-size": 11 });
    }
  });
}

// horizontal bars with value at the tip
function barHChart(container, labels, values, color, opts) {
  opts = opts || {};
  const rowH = 34, labelW = 92;
  const height = M.top + labels.length * rowH + 8;
  const { svg, width } = frame(container, height);
  const iw = width - labelW - 60;
  const vMax = opts.max || niceMax(Math.max(0.01, ...values));
  labels.forEach((lab, i) => {
    const y = M.top + i * rowH + 6;
    const barW2 = Math.max(2, (values[i] / vMax) * iw);
    textNode(svg, "text", lab, { x: labelW - 8, y: y + 14, "text-anchor": "end",
      fill: css("--text-secondary"), "font-size": 12 });
    // rounded data-end on the right, square at the left baseline
    const h = 18, r = 4;
    const d = `M ${labelW} ${y} L ${labelW + barW2 - r} ${y}` +
      ` Q ${labelW + barW2} ${y} ${labelW + barW2} ${y + r} L ${labelW + barW2} ${y + h - r}` +
      ` Q ${labelW + barW2} ${y + h} ${labelW + barW2 - r} ${y + h} L ${labelW} ${y + h} Z`;
    const p = el("path", { d, fill: color });
    svg.appendChild(p);
    textNode(svg, "text", opts.fmt ? opts.fmt(values[i]) : values[i], {
      x: labelW + barW2 + 8, y: y + 14, fill: css("--text-primary"),
      "font-size": 12, "font-weight": 600 });
    const hit = el("rect", { x: 0, y: y - 4, width, height: rowH,
                             fill: "transparent" });
    svg.appendChild(hit);
    hit.addEventListener("pointermove", (ev) =>
      showTip(ev.clientX, ev.clientY, lab,
        [["average at alert", opts.fmt ? opts.fmt(values[i]) : String(values[i]), color]]));
    hit.addEventListener("pointerleave", hideTip);
  });
  svg.appendChild(el("line", { x1: labelW, x2: labelW, y1: M.top - 6,
    y2: M.top + labels.length * rowH, stroke: css("--baseline"), "stroke-width": 1 }));
}

// multi-series line chart with crosshair + one tooltip for every series
function lineChart(container, xs, series, opts) {
  opts = opts || {};
  const fr = frame(container, opts.height || 260);
  const svg = fr.svg, ih = fr.ih;
  const iw = opts.endLabels ? fr.iw - 52 : fr.iw;  // room for end labels
  const vMax2 = Math.max(0.01, ...series.flatMap(s => s.values));
  const yMax = opts.yMax || (opts.intTicks ? niceMaxInt(vMax2) : niceMax(vMax2));
  yGrid(svg, iw, ih, yMax, opts.pct ? "pct" : "");
  const n = xs.length;
  const X = (i) => M.left + (n > 1 ? (i / (n - 1)) * iw : iw / 2);
  const Y = (v) => M.top + ih - (v / yMax) * ih;

  // x ticks (~5)
  const tickN = Math.min(6, n);
  for (let t = 0; t < tickN; t++) {
    const i = Math.round(t * (n - 1) / Math.max(1, tickN - 1));
    textNode(svg, "text", opts.xLabel ? opts.xLabel(xs[i]) : xs[i], {
      x: X(i), y: M.top + ih + 16, "text-anchor": "middle",
      fill: css("--text-muted"), "font-size": 11 });
  }

  // threshold reference line
  if (opts.threshold != null && opts.threshold <= yMax) {
    const ty = Y(opts.threshold);
    svg.appendChild(el("line", { x1: M.left, x2: M.left + iw, y1: ty, y2: ty,
      stroke: css("--critical"), "stroke-width": 1, opacity: 0.65 }));
    textNode(svg, "text", "alert threshold", {
      x: M.left + iw, y: ty - 5, "text-anchor": "end",
      fill: css("--critical"), "font-size": 10, opacity: 0.9 });
  }

  series.forEach(s => {
    if (opts.area && series.length === 1) {   // 10% wash for a single series
      let d = `M ${X(0)} ${Y(s.values[0])}`;
      for (let i = 1; i < n; i++) d += ` L ${X(i)} ${Y(s.values[i])}`;
      d += ` L ${X(n - 1)} ${M.top + ih} L ${X(0)} ${M.top + ih} Z`;
      svg.appendChild(el("path", { d, fill: s.color, opacity: 0.10 }));
    }
    let d = `M ${X(0)} ${Y(s.values[0])}`;
    for (let i = 1; i < n; i++) d += ` L ${X(i)} ${Y(s.values[i])}`;
    svg.appendChild(el("path", { d, fill: "none", stroke: s.color,
      "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    // end marker: >=8px dot with 2px surface ring
    svg.appendChild(el("circle", { cx: X(n - 1), cy: Y(s.values[n - 1]), r: 6,
      fill: s.color, stroke: css("--surface-1"), "stroke-width": 2 }));
    if (opts.endLabels) {
      textNode(svg, "text", s.name, {
        x: X(n - 1) + 9, y: Y(s.values[n - 1]) + 4,
        fill: css("--text-secondary"), "font-size": 11 });
    }
  });

  // alert markers (status color, surface ring)
  (opts.markers || []).forEach(mk => {
    if (mk.i < 0 || mk.i >= n) return;
    svg.appendChild(el("circle", { cx: X(mk.i), cy: Y(mk.v), r: 5,
      fill: css("--critical"), stroke: css("--surface-1"), "stroke-width": 2 }));
  });

  // crosshair + tooltip — snaps to nearest X, lists every series
  const cross = el("line", { y1: M.top, y2: M.top + ih,
    stroke: css("--baseline"), "stroke-width": 1, opacity: 0 });
  svg.appendChild(cross);
  const hit = el("rect", { x: M.left, y: M.top, width: iw, height: ih,
                           fill: "transparent" });
  svg.appendChild(hit);
  hit.addEventListener("pointermove", (ev) => {
    const rect = svg.getBoundingClientRect();
    const relX = ev.clientX - rect.left - M.left;
    const i = Math.max(0, Math.min(n - 1,
      Math.round((relX / Math.max(1, iw)) * (n - 1))));
    cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i));
    cross.setAttribute("opacity", 0.8);
    showTip(ev.clientX, ev.clientY,
      opts.xLabel ? opts.xLabel(xs[i]) : String(xs[i]),
      series.map(s => [s.name,
        opts.pct ? (s.values[i] * 100).toFixed(0) + "%"
                 : fmt(s.values[i]), s.color]));
  });
  hit.addEventListener("pointerleave", () => {
    cross.setAttribute("opacity", 0); hideTip();
  });
}

// ── aggregation ──────────────────────────────────────────────────────
function scope() {
  const v = $("sessionSel").value;
  return v === "all" ? SESSIONS : SESSIONS.filter(s => s.session_id === v);
}
function sum(arr, f) { return arr.reduce((a, x) => a + f(x), 0); }
// peak concurrent objects in a session — derived from the instantaneous
// timeline (bucketed as max), so it means "most in frame at once", never a
// per-frame running total. Works on older session files too.
function peakDogs(s) { return Math.max(0, ...s.timeline.map(p => p.dogs)); }
function peakPersons(s) { return Math.max(0, ...s.timeline.map(p => p.persons)); }

// ── render ───────────────────────────────────────────────────────────
function render() {
  const S = scope();
  const single = S.length === 1 && $("sessionSel").value !== "all";
  $("scopeNote").textContent = S.length + " session(s) in view";

  // KPIs
  const kpis = [
    ["Alerts fired", fmt(sum(S, s => s.alerts_total)), "aggression alerts"],
    ["Peak dogs in frame", S.length ? Math.max(0, ...S.map(peakDogs)) : 0,
      "most seen at once"],
    ["Peak persons in frame", S.length ? Math.max(0, ...S.map(peakPersons)) : 0,
      "most seen at once"],
    ["Frames processed", fmt(sum(S, s => s.frames_processed)),
      mmss(sum(S, s => s.duration_s)) + " monitored"],
    ["Peak risk", S.length ? Math.max(...S.map(s => s.peak_risk)).toFixed(2) : "0.00",
      "avg FPS " + (S.length ? (sum(S, s => s.avg_fps) / S.length).toFixed(1) : "0")],
  ];
  $("kpis").replaceChildren();
  kpis.forEach(([label, value, sub]) => {
    const d = document.createElement("div"); d.className = "tile";
    const l = document.createElement("div"); l.className = "label"; l.textContent = label;
    const v = document.createElement("div"); v.className = "value"; v.textContent = value;
    const s2 = document.createElement("div"); s2.className = "sub"; s2.textContent = sub;
    d.append(l, v, s2); $("kpis").appendChild(d);
  });

  const blue = css("--series-1"), aqua = css("--series-2");

  // Risk over time
  if (S.length === 0) {
    $("riskChart").replaceChildren();
    $("riskChart").insertAdjacentHTML("afterbegin",
      '<div class="empty">No sessions recorded yet — run the monitor first.</div>');
    $("riskTable").replaceChildren();
  } else if (single) {
    const sess = S[0];
    const tl = sess.timeline;
    $("riskTitle").textContent = "Risk over time";
    $("riskSub").textContent = "Highest per-dog risk each moment — session " +
      sess.session_id + " (" + sess.model + ")";
    const markers = sess.alerts.map(a => {
      let best = 0, bd = Infinity;
      tl.forEach((p, i) => { const d = Math.abs(p.t - a.t);
        if (d < bd) { bd = d; best = i; } });
      return { i: best, v: a.risk };
    });
    lineChart($("riskChart"), tl.map(p => p.t),
      [{ name: "risk", color: blue, values: tl.map(p => p.risk) }],
      { pct: false, yMax: 1, area: true, threshold: sess.risk_threshold,
        markers, xLabel: mmss });
    buildTable($("riskTable"), ["Time", "Risk", "Dogs", "Persons"],
      tl.filter((_, i) => i % Math.ceil(tl.length / 25) === 0)
        .map(p => [mmss(p.t), p.risk.toFixed(2), p.dogs, p.persons]));
  } else {
    $("riskTitle").textContent = "Peak risk per session";
    $("riskSub").textContent =
      "Select a single session above to see its full risk timeline";
    columnChart($("riskChart"), S.map(s => s.session_id),
      [{ name: "peak risk", color: blue, values: S.map(s => s.peak_risk) }],
      { pct: true, tickEvery: Math.ceil(S.length / 8), catLabel: sessLabel });
    buildTable($("riskTable"), ["Session", "Peak risk", "Alerts"],
      S.map(s => [s.session_id, s.peak_risk.toFixed(2), s.alerts_total]));
  }

  // Detections: dogs vs persons
  $("detLegend").replaceChildren();
  const mkKey = (name, color, line) => {
    const k = document.createElement("span"); k.className = "key";
    const sw = document.createElement("span");
    sw.className = line ? "swatch-line" : "swatch-rect";
    if (line) sw.style.borderTopColor = color; else sw.style.background = color;
    const t = document.createElement("span"); t.textContent = name;
    k.append(sw, t); return k;
  };
  if (single) {
    const tl = S[0].timeline;
    $("detSub").textContent = "Objects in frame across the session";
    lineChart($("detChart"), tl.map(p => p.t), [
      { name: "dogs", color: blue, values: tl.map(p => p.dogs) },
      { name: "persons", color: aqua, values: tl.map(p => p.persons) },
    ], { height: 230, xLabel: mmss, endLabels: true, intTicks: true });
    $("detLegend").append(mkKey("dogs", blue, true), mkKey("persons", aqua, true));
    buildTable($("detTable"), ["Time", "Dogs", "Persons"],
      tl.filter((_, i) => i % Math.ceil(tl.length / 25) === 0)
        .map(p => [mmss(p.t), p.dogs, p.persons]));
  } else if (S.length) {
    $("detSub").textContent = "Peak concurrent detections per session";
    columnChart($("detChart"), S.map(s => s.session_id), [
      { name: "peak dogs", color: blue, values: S.map(peakDogs) },
      { name: "peak persons", color: aqua, values: S.map(peakPersons) },
    ], { height: 230, tickEvery: Math.ceil(S.length / 6), intTicks: true,
         catLabel: sessLabel });
    $("detLegend").append(mkKey("peak dogs", blue, false),
                          mkKey("peak persons", aqua, false));
    buildTable($("detTable"), ["Session", "Peak dogs", "Peak persons"],
      S.map(s => [s.session_id, peakDogs(s), peakPersons(s)]));
  } else {
    $("detChart").replaceChildren(); $("detTable").replaceChildren();
  }

  // Alerts by hour of day
  const hours = new Array(24).fill(0);
  S.forEach(s => {
    const h0 = new Date(s.started).getHours();
    s.alerts.forEach(a => {
      const h = (h0 + Math.floor(((new Date(s.started).getMinutes() * 60) +
        a.t) / 3600)) % 24;
      hours[h] += 1;
    });
  });
  columnChart($("hourChart"), [...Array(24).keys()],
    [{ name: "alerts", color: blue, values: hours }],
    { height: 210, tickEvery: 4, intTicks: true,
      catLabel: h => String(h).padStart(2, "0") });
  buildTable($("hourTable"), ["Hour", "Alerts"],
    hours.map((v, h) => [String(h).padStart(2, "0") + ":00", v])
         .filter(r => r[1] > 0));

  // Risk distribution histogram (10 bins over timeline samples)
  const bins = new Array(10).fill(0);
  let total = 0;
  S.forEach(s => s.timeline.forEach(p => {
    bins[Math.min(9, Math.floor(p.risk * 10))] += 1; total += 1;
  }));
  const shares = bins.map(b => total ? b / total : 0);
  columnChart($("histChart"), bins.map((_, i) => i),
    [{ name: "share of frames", color: blue, values: shares }],
    { height: 210, pct: true, tickEvery: 2,
      catLabel: i => (i / 10).toFixed(1) + "–" + ((i + 1) / 10).toFixed(1) });
  buildTable($("histTable"), ["Risk band", "Frames", "Share"],
    bins.map((b, i) => [
      (i / 10).toFixed(1) + " – " + ((i + 1) / 10).toFixed(1),
      fmt(b), total ? (100 * b / total).toFixed(1) + "%" : "0%"]));

  // Alert signal contributions
  const featNames = ["distance", "velocity", "posture", "human_pose"];
  const featLabels = ["Distance", "Velocity", "Posture", "Human pose"];
  const allAlerts = S.flatMap(s => s.alerts);
  const featAvg = featNames.map(f => allAlerts.length
    ? sum(allAlerts, a => (a.features && a.features[f]) || 0) / allAlerts.length
    : 0);
  $("featChart").replaceChildren();
  if (allAlerts.length) {
    barHChart($("featChart"), featLabels, featAvg, blue,
      { max: 1, fmt: v => v.toFixed(2) });
  } else {
    $("featChart").insertAdjacentHTML("afterbegin",
      '<div class="empty">No alerts in the selected scope.</div>');
  }
  buildTable($("featTable"), ["Signal", "Avg value at alert"],
    featLabels.map((l, i) => [l, featAvg[i].toFixed(2)]));

  // Sessions table
  buildTable($("sessionsTable"),
    ["Session", "Started", "Source", "Model", "Alert mode", "Frames",
     "Peak dogs", "Peak persons", "Alerts", "Peak risk", "Avg FPS"],
    SESSIONS.slice().reverse().map(s => [
      s.session_id, s.started.replace("T", " "), s.source, s.model,
      s.alert_type.toUpperCase(), fmt(s.frames_processed), peakDogs(s),
      peakPersons(s), fmt(s.alerts_total), s.peak_risk.toFixed(2),
      s.avg_fps.toFixed(1)]));

  $("genNote").textContent = "Generated " + GENERATED + " · " +
    SESSIONS.length + " session file(s) · data: data/sessions/";
}

// filter row
(function initFilter() {
  const sel = $("sessionSel");
  const optAll = document.createElement("option");
  optAll.value = "all"; optAll.textContent = "All sessions";
  sel.appendChild(optAll);
  SESSIONS.slice().reverse().forEach(s => {
    const o = document.createElement("option");
    o.value = s.session_id;
    o.textContent = s.session_id + "  ·  " + s.model + "  ·  " +
      s.alerts_total + " alert(s)";
    sel.appendChild(o);
  });
  sel.addEventListener("change", render);
})();

let rsT = null;
addEventListener("resize", () => {
  clearTimeout(rsT); rsT = setTimeout(render, 150);
});
render();
</script>
</body>
</html>
"""


def generate_dashboard(sessions_dir=None, out_path=None, open_browser=False):
    """Build the HTML dashboard from saved sessions. Returns the output path."""
    cfg = load_config()
    acfg = cfg.get("analytics", {}) or {}
    sessions_dir = sessions_dir or acfg.get("sessions_dir", "data/sessions")
    out_path = out_path or acfg.get("dashboard_file", "outputs/dashboard.html")

    sessions = load_sessions(sessions_dir)
    for s in sessions:
        s.pop("_path", None)

    html = (_TEMPLATE
            .replace("__SESSIONS_JSON__", json.dumps(sessions))
            .replace("__GENERATED__",
                     datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    out = Path(out_path)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    if open_browser:
        webbrowser.open(out.as_uri())
    return str(out)


if __name__ == "__main__":
    path = generate_dashboard(open_browser=True)
    print(f"[analytics] Dashboard written to {path}")
