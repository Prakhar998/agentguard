"""AgentGuard as an MCP server — any MCP-capable agent can self-monitor.

    agentguard mcp        # stdio transport

Register it with your MCP client (Claude Code shown):

    claude mcp add agentguard -- agentguard mcp

The agent (or its harness) streams its own steps in and reads its risk
back — self-monitoring as a tool call:

    start_run(run_id="task-42", goal="refactor the parser")
    record_step(run_id="task-42", kind="tool_call", name="search", content="...")
    -> {"risk": 0.31, "subscores": {...}, "over_threshold": false}

Requires ``pip install agentguard[mcp]``.
"""

from __future__ import annotations

from typing import Dict, Optional

from mcp.server.fastmcp import FastMCP

from .guard import Guard, Watcher
from .memory import FailureMemory

server = FastMCP(
    "agentguard",
    instructions=(
        "Predictive failure detection for agent runs. Call start_run once, "
        "record_step for every tool call/result and LLM output, and check "
        "over_threshold in each response; when true, change approach "
        "(reset context, different tool, or stop) instead of continuing."
    ),
)

_memory = FailureMemory()
_watchers: Dict[str, Watcher] = {}


def _watcher(run_id: str) -> Watcher:
    if run_id not in _watchers:
        raise ValueError(f"unknown run_id {run_id!r}; call start_run first")
    return _watchers[run_id]


@server.tool()
def start_run(
    run_id: str,
    goal: Optional[str] = None,
    predictors: Optional[str] = None,
    threshold: float = 0.8,
) -> dict:
    """Start watching an agent run. predictors: comma-separated names
    (default: loop,tool_cascade,budget_drift; add semantic_drift,
    goal_drift, retrieval_drift, grounding_gap, injection as needed)."""
    names = predictors.split(",") if predictors else None
    guard = Guard(predictors=names, threshold=threshold, memory=_memory)
    _watchers[run_id] = guard.watch(run_id=run_id, goal=goal)
    return {"run_id": run_id, "watching": True,
            "predictors": [p.name for p in _watchers[run_id].predictors]}


@server.tool()
def record_step(
    run_id: str,
    kind: str,
    name: Optional[str] = None,
    content: Optional[str] = None,
    tokens: Optional[int] = None,
    latency_s: Optional[float] = None,
    error: bool = False,
) -> dict:
    """Record one step (kind: tool_call | tool_result | llm_output |
    retrieval | final) and get the updated risk back."""
    w = _watcher(run_id)
    risk = w.record({"kind": kind, "name": name, "content": content,
                     "tokens": tokens, "latency_s": latency_s, "error": error})
    return {
        "risk": round(risk, 4),
        "subscores": {k: round(v, 4) for k, v in w.subscores.items()},
        "over_threshold": risk >= w.guard.threshold,
        "steps": len(w.history),
    }


@server.tool()
def get_risk(run_id: str) -> dict:
    """Current risk and per-predictor sub-scores for a run."""
    w = _watcher(run_id)
    return {
        "risk": round(w.risk, 4),
        "subscores": {k: round(v, 4) for k, v in w.subscores.items()},
        "over_threshold": w.risk >= w.guard.threshold,
        "steps": len(w.history),
    }


@server.tool()
def explain_run(run_id: str, k: int = 3) -> dict:
    """Explain the current risk: the run's failure signature plus the k
    most similar past failures from this server's failure memory."""
    w = _watcher(run_id)
    return {
        "signature": w.signature(),
        "similar_past_failures": [
            {"run_id": m.get("run_id"), "similarity": round(m.get("similarity", 0), 3),
             "summary": m.get("summary")}
            for m in _memory.similar_failures(w, k=k)
        ],
    }


@server.tool()
def end_run(run_id: str, failed: bool = False) -> dict:
    """Finish a run. If it failed (or was halted), its signature is stored
    in the failure memory to explain future runs."""
    w = _watchers.pop(run_id, None)
    if w is None:
        return {"ended": False}
    if failed and not w.interventions:
        w.intervene("halt")  # record the outcome so memory stores it
    w.close()
    return {"ended": True, "peak_risk": round(max(w.risk_trajectory, default=0.0), 4),
            "remembered": bool(w.interventions)}


def main() -> int:
    server.run()  # stdio transport
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
