"""``agentguard serve`` — a live sidecar watching every run at once.

Production agents already emit traces; point them here and AgentGuard
watches all of them concurrently, no code changes in the agent:

* ``POST /v1/traces`` — OTLP/HTTP **JSON** with OpenInference semantic
  conventions (what Arize Phoenix / OpenLLMetry instrumentation emits).
  LLM / TOOL / RETRIEVER spans are mapped onto Steps; runs are keyed by
  trace id.
* ``POST /ingest`` — the native escape hatch:
  ``{"run_id": "...", "step": {"kind": "tool_call", ...}}``
* ``GET /runs`` — machine-readable state of every watched run.
* ``GET /`` — a live dashboard: per-run risk meter, trajectory sparkline,
  dominant signal. Polls every 2s.

Stdlib only — no OTel SDK, no web framework. Protobuf-encoded OTLP is not
supported; configure your exporter for http/json (OTLP_EXPORTER_OTLP_PROTOCOL
=http/json) or use /ingest.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional

from .guard import Guard, Watcher

# -- run registry ---------------------------------------------------------------


class RunRegistry:
    def __init__(self, guard: Guard):
        self.guard = guard
        self._runs: Dict[str, Watcher] = {}
        self._updated: Dict[str, float] = {}
        self._lock = threading.Lock()

    def record(self, run_id: str, step_dict: dict) -> float:
        with self._lock:
            if run_id not in self._runs:
                self._runs[run_id] = self.guard.watch(run_id=run_id)
            risk = self._runs[run_id].record(step_dict)
            self._updated[run_id] = time.time()
            return risk

    def snapshot(self) -> List[dict]:
        with self._lock:
            out = []
            for run_id, w in self._runs.items():
                top = max(w.subscores, key=w.subscores.get) if w.subscores else ""
                out.append({
                    "run_id": run_id,
                    "risk": round(w.risk, 4),
                    "peak_risk": round(max(w.risk_trajectory, default=0.0), 4),
                    "steps": len(w.history),
                    "subscores": {k: round(v, 4) for k, v in w.subscores.items()},
                    "top_signal": top if w.subscores.get(top, 0) > 0.05 else "",
                    "trajectory": [round(r, 3) for r in w.risk_trajectory[-40:]],
                    "over_threshold": w.risk >= self.guard.threshold,
                    "updated_at": self._updated[run_id],
                })
            out.sort(key=lambda r: r["risk"], reverse=True)
            return out


# -- OTLP/OpenInference mapping ---------------------------------------------------


def _attr_map(span: dict) -> dict:
    out = {}
    for kv in span.get("attributes", []):
        v = kv.get("value", {})
        out[kv.get("key")] = (
            v.get("stringValue")
            or v.get("intValue")
            or v.get("doubleValue")
            or v.get("boolValue")
        )
    return out


def _span_latency(span: dict) -> Optional[float]:
    try:
        return (int(span["endTimeUnixNano"]) - int(span["startTimeUnixNano"])) / 1e9
    except (KeyError, ValueError, TypeError):
        return None


def otlp_to_steps(payload: dict) -> List[tuple]:
    """OTLP JSON -> [(run_id, step_dict)], OpenInference conventions."""
    items: List[tuple] = []
    for rs in payload.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            for span in ss.get("spans", []):
                run_id = span.get("traceId", "unknown")
                attrs = _attr_map(span)
                kind = str(attrs.get("openinference.span.kind", "")).upper()
                error = span.get("status", {}).get("code") == 2  # STATUS_CODE_ERROR
                latency = _span_latency(span)
                name = span.get("name")

                if kind == "LLM":
                    tokens = attrs.get("llm.token_count.total")
                    items.append((run_id, {
                        "kind": "llm_output", "name": name,
                        "content": attrs.get("output.value"),
                        "tokens": int(tokens) if tokens else None,
                        "latency_s": latency, "error": error,
                    }))
                elif kind == "TOOL":
                    items.append((run_id, {
                        "kind": "tool_call",
                        "name": attrs.get("tool.name") or name,
                        "content": attrs.get("input.value"),
                    }))
                    items.append((run_id, {
                        "kind": "tool_result",
                        "name": attrs.get("tool.name") or name,
                        "content": attrs.get("output.value"),
                        "latency_s": latency, "error": error,
                    }))
                elif kind == "RETRIEVER":
                    chunks = []
                    i = 0
                    while f"retrieval.documents.{i}.document.content" in attrs:
                        chunks.append(str(attrs[f"retrieval.documents.{i}.document.content"]))
                        i += 1
                    items.append((run_id, {
                        "kind": "retrieval", "name": name,
                        "content": {"query": str(attrs.get("input.value", "")),
                                    "chunks": chunks},
                        "latency_s": latency, "error": error,
                    }))
    # spans arrive newest-last within a batch; preserve given order
    return items


# -- HTTP server ---------------------------------------------------------------------


def make_handler(registry: RunRegistry):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet by default
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json_body(self) -> Optional[dict]:
            try:
                length = int(self.headers.get("Content-Length", 0))
                return json.loads(self.rfile.read(length))
            except (ValueError, TypeError):
                return None

        def do_GET(self):
            if self.path.split("?")[0] == "/runs":
                self._send(200, json.dumps({
                    "threshold": registry.guard.threshold,
                    "runs": registry.snapshot(),
                }).encode())
            elif self.path.split("?")[0] == "/":
                self._send(200, DASHBOARD_HTML.encode(), "text/html; charset=utf-8")
            else:
                self._send(404, b'{"error": "not found"}')

        def do_POST(self):
            body = self._json_body()
            if body is None:
                self._send(400, b'{"error": "invalid json"}')
                return
            path = self.path.split("?")[0]
            if path == "/ingest":
                run_id = str(body.get("run_id", "default"))
                steps = body.get("steps") or ([body["step"]] if "step" in body else [])
                risk = None
                for s in steps:
                    try:
                        risk = registry.record(run_id, s)
                    except (KeyError, ValueError):
                        continue
                self._send(200, json.dumps({"run_id": run_id, "risk": risk}).encode())
            elif path == "/v1/traces":
                n = 0
                for run_id, step in otlp_to_steps(body):
                    try:
                        registry.record(run_id, step)
                        n += 1
                    except (KeyError, ValueError):
                        continue
                # OTLP/HTTP success reply shape
                self._send(200, json.dumps({"partialSuccess": {},
                                            "agentguard_steps": n}).encode())
            else:
                self._send(404, b'{"error": "not found"}')

    return Handler


def serve(port: int = 4318, host: str = "127.0.0.1",
          predictors: Optional[List[str]] = None, threshold: float = 0.8,
          registry: Optional[RunRegistry] = None) -> ThreadingHTTPServer:
    registry = registry or RunRegistry(Guard(predictors=predictors, threshold=threshold))
    server = ThreadingHTTPServer((host, port), make_handler(registry))
    server.registry = registry  # for tests
    return server


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="agentguard serve",
                                 description="Live sidecar + dashboard.")
    ap.add_argument("--port", type=int, default=4318)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--predictors", default="loop,tool_cascade,budget_drift,injection")
    args = ap.parse_args(argv)

    server = serve(args.port, args.host, args.predictors.split(","), args.threshold)
    print(f"agentguard serve — dashboard http://{args.host}:{args.port}/  "
          f"(OTLP JSON: POST /v1/traces, native: POST /ingest)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


# -- dashboard --------------------------------------------------------------------
# Colors are the AgentGuard-adopted reference viz palette (validated set):
# status roles carry meter severity (value always shown as text beside the
# fill — never color alone); the sparkline is a single muted series with the
# current point in the severity color, so no categorical palette is needed.

DASHBOARD_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AgentGuard — live runs</title>
<style>
  :root {
    --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink-2: #52514e;
    --muted: #898781; --grid: #e1e0d9; --border: rgba(11,11,11,.10);
    --accent: #2a78d6; --accent-track: #cde2fb;
    --warn: #fab219; --warn-track: #fdeecb;
    --crit: #d03b3b; --crit-track: #f6d8d8; --good: #0ca30c;
  }
  @media (prefers-color-scheme: dark) { :root {
    --surface: #1a1a19; --page: #0d0d0d; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --border: rgba(255,255,255,.10);
    --accent: #3987e5; --accent-track: #16324f;
    --warn: #fab219; --warn-track: #46350a;
    --crit: #d03b3b; --crit-track: #4a1a1a; --good: #0ca30c;
  } }
  * { box-sizing: border-box; margin: 0; }
  body { background: var(--page); color: var(--ink);
         font: 14px/1.45 -apple-system, "Segoe UI", Roboto, sans-serif; padding: 24px; }
  h1 { font-size: 16px; font-weight: 600; }
  .sub { color: var(--ink-2); margin: 2px 0 20px; }
  .tiles { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }
  .tile { background: var(--surface); border: 1px solid var(--border);
          border-radius: 8px; padding: 12px 16px; min-width: 150px; }
  .tile .label { color: var(--ink-2); font-size: 12px; }
  .tile .value { font-size: 26px; font-weight: 600; margin-top: 2px; }
  .tile .value.at-risk { color: var(--crit); }
  .card { background: var(--surface); border: 1px solid var(--border);
          border-radius: 8px; overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; min-width: 720px; }
  th { text-align: left; font-size: 12px; font-weight: 500; color: var(--muted);
       padding: 10px 14px; border-bottom: 1px solid var(--grid); }
  td { padding: 10px 14px; border-bottom: 1px solid var(--grid);
       font-variant-numeric: tabular-nums; }
  tr:last-child td { border-bottom: none; }
  .runid { font-family: ui-monospace, Menlo, monospace; font-size: 13px; }
  .meter { position: relative; width: 140px; height: 8px; border-radius: 4px; }
  .meter .fill { position: absolute; inset: 0 auto 0 0; border-radius: 4px; }
  .signal { color: var(--ink-2); }
  .risk-val { font-weight: 600; }
  .empty { padding: 32px; color: var(--muted); text-align: center; }
  .spark { display: block; }
  #tip { position: fixed; pointer-events: none; background: var(--surface);
         border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px;
         font-size: 12px; display: none; box-shadow: 0 2px 8px rgba(0,0,0,.12); }
</style></head><body>
<h1>AgentGuard — live runs</h1>
<div class="sub" id="sub">watching…</div>
<div class="tiles">
  <div class="tile"><div class="label">Runs watched</div><div class="value" id="t-runs">0</div></div>
  <div class="tile"><div class="label">Over threshold</div><div class="value at-risk" id="t-risk">0</div></div>
  <div class="tile"><div class="label">Highest risk</div><div class="value" id="t-max">0.00</div></div>
</div>
<div class="card"><table>
  <thead><tr><th>run</th><th>steps</th><th>risk trajectory</th><th>risk</th>
  <th></th><th>dominant signal</th></tr></thead>
  <tbody id="rows"><tr><td colspan="6" class="empty">no runs yet — POST steps to
  /ingest or OTLP JSON to /v1/traces</td></tr></tbody>
</table></div>
<div id="tip"></div>
<script>
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const sev = r => r >= 0.8 ? ["--crit","--crit-track"] : r >= 0.5 ? ["--warn","--warn-track"] : ["--accent","--accent-track"];

function spark(traj, color) {
  const W = 120, H = 26, n = traj.length;
  if (n < 2) return `<svg class="spark" width="${W}" height="${H}"></svg>`;
  const x = i => 4 + i * (W - 12) / (n - 1);
  const y = v => H - 4 - v * (H - 8);
  const pts = traj.map((v, i) => `${x(i)},${y(v)}`).join(" ");
  const lx = x(n - 1), ly = y(traj[n - 1]);
  return `<svg class="spark" width="${W}" height="${H}" data-traj="${traj.join(',')}">` +
    `<polyline points="${pts}" fill="none" stroke="${css('--muted')}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>` +
    `<circle cx="${lx}" cy="${ly}" r="4" fill="${color}" stroke="${css('--surface')}" stroke-width="2"/></svg>`;
}

async function refresh() {
  let data;
  try { data = await (await fetch("/runs")).json(); } catch { return; }
  const runs = data.runs, th = data.threshold;
  document.getElementById("sub").textContent =
    `threshold ${th} · updated ${new Date().toLocaleTimeString()}`;
  document.getElementById("t-runs").textContent = runs.length;
  document.getElementById("t-risk").textContent = runs.filter(r => r.over_threshold).length;
  document.getElementById("t-max").textContent =
    (runs.length ? Math.max(...runs.map(r => r.risk)) : 0).toFixed(2);
  const rows = runs.map(r => {
    const [fillVar, trackVar] = sev(r.risk);
    const fill = css(fillVar), track = css(trackVar);
    return `<tr>
      <td class="runid">${r.run_id.slice(0, 20)}</td>
      <td>${r.steps}</td>
      <td>${spark(r.trajectory, fill)}</td>
      <td class="risk-val">${r.risk.toFixed(2)}</td>
      <td><div class="meter" style="background:${track}">
        <div class="fill" style="width:${Math.max(4, r.risk * 140)}px;background:${fill}"></div>
      </div></td>
      <td class="signal">${r.top_signal ? r.top_signal + " = " + r.subscores[r.top_signal].toFixed(2) : "—"}</td>
    </tr>`;
  }).join("");
  document.getElementById("rows").innerHTML =
    rows || `<tr><td colspan="6" class="empty">no runs yet — POST steps to /ingest or OTLP JSON to /v1/traces</td></tr>`;
}

const tip = document.getElementById("tip");
document.addEventListener("mousemove", e => {
  const svg = e.target.closest("svg.spark");
  if (!svg || !svg.dataset.traj) { tip.style.display = "none"; return; }
  const traj = svg.dataset.traj.split(",").map(Number);
  const rect = svg.getBoundingClientRect();
  const i = Math.max(0, Math.min(traj.length - 1,
    Math.round((e.clientX - rect.left - 4) / ((rect.width - 12) / (traj.length - 1)))));
  tip.textContent = `step ${i}: risk ${traj[i].toFixed(2)}`;
  tip.style.left = (e.clientX + 12) + "px";
  tip.style.top = (e.clientY - 28) + "px";
  tip.style.display = "block";
});

refresh();
setInterval(refresh, 2000);
</script></body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
