"""Helpers for plain while-loop agents — build Steps without touching the
dataclass directly.

    w.record(tool_call("search", {"q": "population of mars"}))
    w.record(tool_result("search", "no results", error=True, latency_s=0.4))
    w.record(llm_output("I should search again...", tokens=120))
"""

from __future__ import annotations

from typing import Any, Optional

from ..telemetry import Step, StepKind


def tool_call(
    name: str,
    args: Any = None,
    *,
    tokens: Optional[int] = None,
    latency_s: Optional[float] = None,
    error: bool = False,
) -> Step:
    return Step(StepKind.TOOL_CALL, name=name, content=args, tokens=tokens,
                latency_s=latency_s, error=error)


def tool_result(
    name: str,
    result: Any = None,
    *,
    tokens: Optional[int] = None,
    latency_s: Optional[float] = None,
    error: bool = False,
) -> Step:
    return Step(StepKind.TOOL_RESULT, name=name, content=result, tokens=tokens,
                latency_s=latency_s, error=error)


def llm_output(
    text: str,
    *,
    model: Optional[str] = None,
    tokens: Optional[int] = None,
    latency_s: Optional[float] = None,
) -> Step:
    return Step(StepKind.LLM_OUTPUT, name=model, content=text, tokens=tokens,
                latency_s=latency_s)


def final(text: str = "", *, tokens: Optional[int] = None) -> Step:
    return Step(StepKind.FINAL, content=text, tokens=tokens)
