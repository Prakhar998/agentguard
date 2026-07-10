"""LangChain adapter — a callback handler that streams a LangChain run
into an AgentGuard Watcher.

    from agentguard import Guard
    from agentguard.adapters.langchain import AgentGuardCallback

    handler = AgentGuardCallback(Guard(), auto_intervene="halt")
    agent.invoke({"input": "..."}, config={"callbacks": [handler]})

Every tool start/end/error and LLM completion becomes a Step. With
``auto_intervene`` set, crossing the risk threshold raises
:class:`AgentGuardHalt` out of the callback, stopping the run — the one
place "propose, don't enforce" is deliberately overridden, because you
opted in.

Requires ``pip install agentguard[langchain]`` (langchain-core).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from ..guard import Guard, Watcher
from ..telemetry import Step, StepKind


class AgentGuardHalt(RuntimeError):
    """Raised out of the run when auto_intervene triggers."""

    def __init__(self, watcher: Watcher, message: str):
        super().__init__(message)
        self.watcher = watcher


def _token_count(response: Any) -> Optional[int]:
    """Best-effort total-token extraction across LangChain versions."""
    llm_output = getattr(response, "llm_output", None) or {}
    usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    if usage.get("total_tokens"):
        return int(usage["total_tokens"])
    try:
        msg = response.generations[0][0].message
        meta = getattr(msg, "usage_metadata", None) or {}
        if meta.get("total_tokens"):
            return int(meta["total_tokens"])
    except (AttributeError, IndexError):
        pass
    return None


def _response_text(response: Any) -> str:
    try:
        gen = response.generations[0][0]
    except (AttributeError, IndexError):
        return ""
    text = getattr(gen, "text", "") or ""
    msg = getattr(gen, "message", None)
    if msg is not None and getattr(msg, "tool_calls", None):
        calls = "; ".join(
            f"{c.get('name')}({c.get('args')})" for c in msg.tool_calls
        )
        text = f"{text} [tool_calls: {calls}]".strip()
    return text


class AgentGuardCallback(BaseCallbackHandler):
    """Feed a LangChain run into AgentGuard, live."""

    # raise inside callbacks instead of swallowing our halt
    raise_error = True

    def __init__(
        self,
        guard: Optional[Guard] = None,
        run_id: Optional[str] = None,
        auto_intervene: Optional[str] = None,
    ):
        self.guard = guard or Guard()
        self.watcher = self.guard.watch(run_id=run_id)
        self.auto_intervene = auto_intervene
        self._started: Dict[UUID, float] = {}

    # -- plumbing ------------------------------------------------------

    def _record(self, step: Step) -> None:
        risk = self.watcher.record(step)
        if self.auto_intervene and risk >= self.guard.threshold:
            proposal = self.watcher.intervene(self.auto_intervene)
            raise AgentGuardHalt(
                self.watcher,
                f"AgentGuard risk {risk:.2f} >= {self.guard.threshold} at step "
                f"{proposal.step_index}; proposed {proposal.action!r} "
                f"(subscores: {proposal.subscores})",
            )

    def _elapsed(self, run_id: UUID) -> Optional[float]:
        start = self._started.pop(run_id, None)
        return None if start is None else time.time() - start

    # -- tools -----------------------------------------------------------

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs) -> None:
        self._started[run_id] = time.time()
        name = (serialized or {}).get("name", "tool")
        self._record(Step(StepKind.TOOL_CALL, name=name, content=input_str))

    def on_tool_end(self, output, *, run_id, **kwargs) -> None:
        self._record(
            Step(
                StepKind.TOOL_RESULT,
                content=str(output)[:2000],
                latency_s=self._elapsed(run_id),
            )
        )

    def on_tool_error(self, error, *, run_id, **kwargs) -> None:
        self._record(
            Step(
                StepKind.TOOL_RESULT,
                content=repr(error)[:500],
                error=True,
                latency_s=self._elapsed(run_id),
            )
        )

    # -- LLM ------------------------------------------------------------

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs) -> None:
        self._started[run_id] = time.time()

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        self._started[run_id] = time.time()

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        self._record(
            Step(
                StepKind.LLM_OUTPUT,
                content=_response_text(response),
                tokens=_token_count(response),
                latency_s=self._elapsed(run_id),
            )
        )

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        self._record(
            Step(
                StepKind.LLM_OUTPUT,
                content=repr(error)[:500],
                error=True,
                latency_s=self._elapsed(run_id),
            )
        )

    # -- agent ------------------------------------------------------------

    def on_agent_finish(self, finish, *, run_id, **kwargs) -> None:
        output = getattr(finish, "return_values", {}) or {}
        self._record(Step(StepKind.FINAL, content=str(output)[:2000]))
        self.watcher.close()

    # -- convenience -------------------------------------------------------

    @property
    def risk(self) -> float:
        return self.watcher.risk

    @property
    def subscores(self):
        return self.watcher.subscores
