"""Replay historical agent traces through AgentGuard — the backtest.

    agentguard replay traces.jsonl --threshold 0.8
    agentguard replay --demo

Answers the question that sells prediction: *of the runs that failed, how
many would AgentGuard have flagged, and how many steps early?* Same
evaluation discipline as the ProactiveGuard paper (early-warning recall +
lead time), applied to your own traces.

Trace format (JSONL, one object per line):

    {"run_id": "r1", "kind": "tool_call", "name": "search", "content": {...}}
    {"run_id": "r1", "kind": "llm_output", "content": "...", "tokens": 120}
    {"run_id": "r1", "outcome": "failed"}          # optional outcome line

A JSON file holding ``[{"run_id": ..., "outcome": ..., "steps": [...]}]``
is also accepted. Only ``kind`` is required per step — anything you can't
export simply doesn't contribute.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .guard import Guard
from .telemetry import Step


@dataclass
class RunResult:
    run_id: str
    outcome: Optional[str]         # "failed" | "success" | None (unknown)
    steps: int
    peak_risk: float
    flagged_at: Optional[int]      # step index where risk first crossed
    subscores_at_flag: Dict[str, float] = field(default_factory=dict)

    @property
    def lead_steps(self) -> Optional[int]:
        if self.flagged_at is None:
            return None
        return self.steps - 1 - self.flagged_at


# -- trace loading -------------------------------------------------------------


def load_traces(path: str) -> List[Tuple[str, Optional[str], List[dict]]]:
    """Returns [(run_id, outcome, [step dicts])] preserving step order."""
    with open(path) as f:
        text = f.read().strip()

    runs: Dict[str, List[dict]] = {}
    outcomes: Dict[str, Optional[str]] = {}
    order: List[str] = []

    def note(run_id: str) -> None:
        if run_id not in runs:
            runs[run_id] = []
            outcomes[run_id] = None
            order.append(run_id)

    if text.startswith("["):  # JSON array of run objects
        for obj in json.loads(text):
            run_id = str(obj.get("run_id", f"run_{len(order)}"))
            note(run_id)
            outcomes[run_id] = obj.get("outcome")
            runs[run_id].extend(obj.get("steps", []))
    else:  # JSONL
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            run_id = str(obj.get("run_id", "run_0"))
            note(run_id)
            if "outcome" in obj and "kind" not in obj:
                outcomes[run_id] = obj["outcome"]
            elif "kind" in obj:
                runs[run_id].append(obj)

    return [(rid, outcomes[rid], runs[rid]) for rid in order if runs[rid]]


def demo_traces(n_per_type: int = 15, seed: int = 7) -> List[Tuple[str, Optional[str], List[dict]]]:
    """Synthesize labeled scenario runs (no dependencies needed)."""
    from .scenarios import generate_runs

    out = []
    for i, (steps, labels) in enumerate(generate_runs(n_per_type=n_per_type, seed=seed)):
        outcome = "failed" if any(labels) else "success"
        step_dicts = [
            {
                "kind": s.kind.value,
                "name": s.name,
                "content": s.content,
                "tokens": s.tokens,
                "latency_s": s.latency_s,
                "error": s.error,
            }
            for s in steps
        ]
        out.append((f"demo_{outcome}_{i}", outcome, step_dicts))
    return out


# -- replay --------------------------------------------------------------------


def replay(
    traces: Iterable[Tuple[str, Optional[str], List[dict]]],
    guard: Optional[Guard] = None,
) -> List[RunResult]:
    guard = guard or Guard()
    results = []
    for run_id, outcome, step_dicts in traces:
        w = guard.watch(run_id=run_id)
        flagged_at, subs = None, {}
        for d in step_dicts:
            try:
                risk = w.record(Step.from_dict(d))
            except (KeyError, ValueError):
                continue  # malformed step: skip, keep replaying
            if flagged_at is None and risk >= guard.threshold:
                flagged_at = len(w.history) - 1
                subs = dict(w.subscores)
        results.append(
            RunResult(
                run_id=run_id,
                outcome=outcome,
                steps=len(w.history),
                peak_risk=max(w.risk_trajectory, default=0.0),
                flagged_at=flagged_at,
                subscores_at_flag=subs,
            )
        )
    return results


def summarize(results: List[RunResult], threshold: float) -> Dict:
    failed = [r for r in results if r.outcome == "failed"]
    success = [r for r in results if r.outcome == "success"]
    caught = [r for r in failed if r.flagged_at is not None]
    false_alarms = [r for r in success if r.flagged_at is not None]
    leads = [r.lead_steps for r in caught]

    summary: Dict = {
        "runs": len(results),
        "threshold": threshold,
        "labeled_failed": len(failed),
        "labeled_success": len(success),
    }
    if failed:
        summary["catch_rate"] = len(caught) / len(failed)
        if leads:
            summary["mean_lead_steps"] = sum(leads) / len(leads)
            summary["median_lead_steps"] = sorted(leads)[len(leads) // 2]
    if success:
        summary["false_alarm_rate"] = len(false_alarms) / len(success)
    return summary


def print_report(results: List[RunResult], summary: Dict) -> None:
    print(f"\n{'run':<22} {'outcome':<9} {'steps':>5} {'peak':>5}  flagged")
    print("-" * 60)
    for r in results:
        flag = f"step {r.flagged_at} ({r.lead_steps} early)" if r.flagged_at is not None else "-"
        print(f"{r.run_id[:22]:<22} {r.outcome or '?':<9} {r.steps:>5} {r.peak_risk:>5.2f}  {flag}")

    print("\n" + "=" * 60)
    print(f"runs replayed: {summary['runs']}   threshold: {summary['threshold']}")
    if "catch_rate" in summary:
        print(f"failed runs caught before the end: "
              f"{summary['catch_rate']:.0%} ({summary['labeled_failed']} failed runs)")
    if "mean_lead_steps" in summary:
        print(f"early-warning lead: mean {summary['mean_lead_steps']:.1f} steps, "
              f"median {summary['median_lead_steps']} steps before the run ended")
    if "false_alarm_rate" in summary:
        print(f"false alarms on successful runs: "
              f"{summary['false_alarm_rate']:.0%} ({summary['labeled_success']} success runs)")
    print("=" * 60)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="agentguard replay",
        description="Backtest AgentGuard against historical agent traces.",
    )
    ap.add_argument("traces", nargs="?", help="JSONL/JSON trace file (see module docstring)")
    ap.add_argument("--demo", action="store_true",
                    help="replay synthesized scenario runs instead of a file")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--predictors", default="loop,tool_cascade,budget_drift",
                    help="comma-separated predictor names")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    if not args.demo and not args.traces:
        ap.error("provide a trace file or --demo")

    traces = demo_traces() if args.demo else load_traces(args.traces)
    guard = Guard(predictors=args.predictors.split(","), threshold=args.threshold)
    results = replay(traces, guard)
    summary = summarize(results, args.threshold)

    if args.json:
        payload = {
            "summary": summary,
            "runs": [
                {
                    "run_id": r.run_id, "outcome": r.outcome, "steps": r.steps,
                    "peak_risk": r.peak_risk, "flagged_at": r.flagged_at,
                    "lead_steps": r.lead_steps,
                }
                for r in results
            ],
        }
        json.dump(payload, sys.stdout, indent=2)
        print()
    else:
        print_report(results, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
