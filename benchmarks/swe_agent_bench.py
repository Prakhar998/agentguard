#!/usr/bin/env python3
"""Benchmark AgentGuard on REAL agent runs — SWE-agent trajectories.

Data: nebius/SWE-agent-trajectories (HuggingFace) — 80k runs of SWE-agent
on SWE-bench-style issues, each labeled ``target`` (True = the generated
patch resolved the issue) with an ``exit_status`` (how the run ended:
submitted / exit_cost / exit_context / exit_error ...).

    python benchmarks/swe_agent_bench.py --resolved 100 --unresolved 200

Honesty notes, also printed in the report:
* An unresolved run is not necessarily a *dysfunctional* run — many agents
  finish cleanly with a wrong patch. AgentGuard predicts dysfunction
  (loops, cascades, budget blowout), so the interesting split is
  exit_cost / exit_context (the run died of resource exhaustion — exactly
  the failures worth halting early) vs clean submissions.
* Per-message token counts aren't in the data; tokens are estimated as
  len(text)/4. Tool errors are inferred from observation text (traceback/
  "command not found"/...), a heuristic, not ground truth.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentguard.guard import Guard  # noqa: E402
from agentguard.replay import replay, summarize  # noqa: E402

API = "https://datasets-server.huggingface.co"
DATASET = "nebius/SWE-agent-trajectories"
CACHE = Path(__file__).resolve().parent / "data"

ERROR_MARKERS = (
    "traceback (most recent call last)", "command not found",
    "no such file or directory", "syntax error", "syntaxerror",
    "permission denied", "not found\n", "error:", "exception:",
)


# -- download --------------------------------------------------------------------


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read())


def fetch_runs(target: bool, n: int) -> list:
    """Fetch n runs with the given resolution label, cached on disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / f"runs_target_{target}_{n}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    rows, offset = [], 0
    where = urllib.parse.quote(f'"target"={"true" if target else "false"}')
    while len(rows) < n and offset < 8000:
        # /filter is the fast path; fall back to /rows + client-side label
        # filtering when the filter index errors (it often does for the
        # majority label)
        try:
            data = _get(f"{API}/filter?dataset={urllib.parse.quote(DATASET)}"
                        f"&config=default&split=train&where={where}"
                        f"&offset={offset}&length=100")
        except (urllib.error.HTTPError, urllib.error.URLError):
            data = _get(f"{API}/rows?dataset={urllib.parse.quote(DATASET)}"
                        f"&config=default&split=train&offset={offset}&length=100")
        batch = data.get("rows", [])
        if not batch:
            break
        rows.extend(r["row"] for r in batch
                    if not r.get("truncated_cells") and r["row"]["target"] == target)
        offset += 100
        print(f"  fetched {len(rows)}/{n} target={target} runs", end="\r")
    print()
    rows = rows[:n]
    cache_file.write_text(json.dumps(rows))
    return rows


# -- conversion: SWE-agent ReAct messages -> AgentGuard steps ----------------------

CMD_BLOCK = re.compile(r"```\n?(.*?)```", re.S)


def looks_like_error(observation: str) -> bool:
    head = observation[:400].lower()
    return any(m in head for m in ERROR_MARKERS)


def to_steps(trajectory: list) -> list:
    steps = []
    for i, msg in enumerate(trajectory):
        text = msg.get("text") or ""
        role = msg.get("role")
        if role == "ai":
            steps.append({"kind": "llm_output", "content": text,
                          "tokens": max(1, len(text) // 4)})
            m = CMD_BLOCK.search(text)
            if m:
                command = m.group(1).strip()
                name = command.split()[0][:40] if command.split() else "shell"
                steps.append({"kind": "tool_call", "name": name, "content": command})
        elif role == "user" and i > 1:  # skip the issue statement itself
            steps.append({"kind": "tool_result", "content": text[:2000],
                          "error": looks_like_error(text)})
    return steps


# -- benchmark -----------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolved", type=int, default=100)
    ap.add_argument("--unresolved", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--predictors", default="loop,tool_cascade,budget_drift")
    ap.add_argument("--out", default=str(Path(__file__).parent / "RESULTS.md"))
    args = ap.parse_args()

    print("downloading real SWE-agent runs (cached in benchmarks/data)...")
    resolved = fetch_runs(True, args.resolved)
    unresolved = fetch_runs(False, args.unresolved)

    traces, exit_of = [], {}
    for rows, outcome in ((resolved, "success"), (unresolved, "failed")):
        for r in rows:
            run_id = f"{r['instance_id']}::{r['model_name']}"[:60]
            traces.append((run_id, outcome, to_steps(r["trajectory"])))
            exit_of[run_id] = (r.get("exit_status") or "unknown").strip()

    guard = Guard(predictors=args.predictors.split(","), threshold=args.threshold)
    print(f"replaying {len(traces)} runs through AgentGuard "
          f"({args.predictors}, threshold {args.threshold})...")
    results = replay(traces, guard)
    summary = summarize(results, args.threshold)

    # per-exit-status breakdown on unresolved runs
    by_exit = defaultdict(lambda: [0, 0, []])  # exit -> [flagged, total, leads]
    for r in results:
        if r.outcome != "failed":
            continue
        exit_status = exit_of.get(r.run_id, "unknown")
        bucket = "exit_cost/context" if "exit_cost" in exit_status or "exit_context" in exit_status \
            else ("exit_error" if "error" in exit_status else "clean submission (wrong patch)")
        by_exit[bucket][1] += 1
        if r.flagged_at is not None:
            by_exit[bucket][0] += 1
            by_exit[bucket][2].append(r.lead_steps)

    fa = summary.get("false_alarm_rate", 0.0)
    lines = [
        "# AgentGuard on real SWE-agent runs",
        "",
        f"Dataset: [`{DATASET}`](https://huggingface.co/datasets/{DATASET}) — "
        f"{len(traces)} runs ({len(resolved)} resolved, {len(unresolved)} unresolved), "
        f"deterministic predictors only (`{args.predictors}`), threshold {args.threshold}.",
        "",
        "| metric | value |",
        "|---|---|",
        f"| flag rate on unresolved runs | **{summary.get('catch_rate', 0):.0%}** |",
        f"| false-alarm rate on resolved runs | **{fa:.0%}** |",
        f"| mean early-warning lead (flagged runs) | **{summary.get('mean_lead_steps', 0):.1f} steps** |",
        f"| median lead | {summary.get('median_lead_steps', 0)} steps |",
        "",
        "## By how the unresolved run actually ended",
        "",
        "| exit class | flagged | mean lead |",
        "|---|---|---|",
    ]
    for bucket, (flagged, total, leads) in sorted(by_exit.items()):
        mean_lead = sum(leads) / len(leads) if leads else 0.0
        lines.append(f"| {bucket} | {flagged}/{total} ({flagged / total:.0%}) "
                     f"| {mean_lead:.1f} steps |")
    # threshold sweep from peak risks (flag <=> peak >= t)
    peaks = {"resolved": [], "dysfunction": [], "wrong_patch": []}
    for r in results:
        if r.outcome == "success":
            peaks["resolved"].append(r.peak_risk)
        else:
            exit_status = exit_of.get(r.run_id, "")
            cls = "dysfunction" if ("exit_cost" in exit_status or "exit_context" in exit_status) \
                else "wrong_patch"
            peaks[cls].append(r.peak_risk)
    lines += [
        "",
        "## Threshold sweep (pick your operating point)",
        "",
        "| threshold | dysfunction flagged | wrong-patch flagged | false alarms (resolved) |",
        "|---|---|---|---|",
    ]
    for t in (0.6, 0.7, 0.8, 0.9):
        def rate(key):
            vals = peaks[key]
            return sum(p >= t for p in vals) / len(vals) if vals else 0.0
        lines.append(f"| {t} | {rate('dysfunction'):.0%} | {rate('wrong_patch'):.0%} "
                     f"| {rate('resolved'):.0%} |")

    lines += [
        "",
        "## Honest reading",
        "",
        "- **Dysfunction vs wrong answers.** AgentGuard predicts *dysfunction* "
        "(loops, error cascades, budget blowout). Runs that died of "
        "`exit_cost`/`exit_context` are the dysfunction class — the flag rate "
        "there is the number that matters, and every step of lead time is "
        "unspent budget. A run that submitted a clean-but-wrong patch is often "
        "indistinguishable from a healthy run mid-flight, and the low flag rate "
        "on that class is expected, not a miss.",
        "- Tokens are estimated (len/4); tool errors are inferred from "
        "observation text. Both proxies are stated in the converter.",
        "- Reproduce: `python benchmarks/swe_agent_bench.py`.",
    ]
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
