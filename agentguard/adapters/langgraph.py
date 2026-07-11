"""LangGraph adapter — per-node guarding for agent graphs.

LangGraph stamps every callback's metadata with ``langgraph_node``, so a
single callback handler can route each event to the right per-agent
Watcher inside a :class:`agentguard.multi.MultiGuard`:

    from agentguard.multi import MultiGuard
    from agentguard.adapters.langgraph import MultiAgentCallback

    mg = MultiGuard(predictors=["loop", "tool_cascade", "budget_drift"])
    mg.add_edge("researcher", "writer")

    handler = MultiAgentCallback(mg, auto_intervene="halt")
    graph.invoke(state, config={"callbacks": [handler]})

Works with anything that forwards LangChain callbacks and metadata — a
compiled LangGraph, a LangChain runnable pipeline, or your own code
passing ``metadata={"langgraph_node": "researcher"}``.

Requires ``pip install agentguard[langchain]`` (langchain-core).
"""

from __future__ import annotations

import time
from typing import Dict, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from ..multi import MultiGuard
from ..telemetry import Step, StepKind
from .langchain import AgentGuardHalt, _response_text, _token_count

DEFAULT_NODE = "graph"


class MultiAgentCallback(BaseCallbackHandler):
    """Route LangChain/LangGraph callbacks into per-node Watchers."""

    raise_error = True

    def __init__(self, multi_guard: Optional[MultiGuard] = None,
                 auto_intervene: Optional[str] = None,
                 node_key: str = "langgraph_node"):
        self.mg = multi_guard or MultiGuard()
        self.auto_intervene = auto_intervene
        self.node_key = node_key
        self._started: Dict[UUID, float] = {}
        self._node_of: Dict[UUID, str] = {}
        self._queries: Dict[UUID, str] = {}

    # -- plumbing -----------------------------------------------------------

    def _node(self, run_id: UUID, metadata: Optional[dict]) -> str:
        if metadata and metadata.get(self.node_key):
            self._node_of[run_id] = str(metadata[self.node_key])
        return self._node_of.get(run_id, DEFAULT_NODE)

    def _record(self, run_id: UUID, step: Step) -> None:
        node = self._node_of.get(run_id, DEFAULT_NODE)
        effective = self.mg.record(node, step)
        if self.auto_intervene and effective >= self.mg.guard.threshold:
            watcher = self.mg.watcher(node)
            proposal = watcher.intervene(self.auto_intervene)
            raise AgentGuardHalt(
                watcher,
                f"AgentGuard: agent {node!r} effective risk {effective:.2f} >= "
                f"{self.mg.guard.threshold} at its step {proposal.step_index} "
                f"(own subscores: {proposal.subscores}); proposed {proposal.action!r}",
            )

    def _elapsed(self, run_id: UUID) -> Optional[float]:
        start = self._started.pop(run_id, None)
        return None if start is None else time.time() - start

    # -- tools ------------------------------------------------------------------

    def on_tool_start(self, serialized, input_str, *, run_id, metadata=None, **kwargs):
        self._started[run_id] = time.time()
        self._node(run_id, metadata)
        name = (serialized or {}).get("name", "tool")
        self._record(run_id, Step(StepKind.TOOL_CALL, name=name, content=input_str))

    def on_tool_end(self, output, *, run_id, **kwargs):
        self._record(run_id, Step(StepKind.TOOL_RESULT, content=str(output)[:2000],
                                  latency_s=self._elapsed(run_id)))

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._record(run_id, Step(StepKind.TOOL_RESULT, content=repr(error)[:500],
                                  error=True, latency_s=self._elapsed(run_id)))

    # -- LLM ---------------------------------------------------------------------

    def on_llm_start(self, serialized, prompts, *, run_id, metadata=None, **kwargs):
        self._started[run_id] = time.time()
        self._node(run_id, metadata)

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        self._started[run_id] = time.time()
        self._node(run_id, metadata)

    def on_llm_end(self, response, *, run_id, **kwargs):
        self._record(run_id, Step(StepKind.LLM_OUTPUT, content=_response_text(response),
                                  tokens=_token_count(response),
                                  latency_s=self._elapsed(run_id)))

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._record(run_id, Step(StepKind.LLM_OUTPUT, content=repr(error)[:500],
                                  error=True, latency_s=self._elapsed(run_id)))

    # -- retriever ------------------------------------------------------------------

    def on_retriever_start(self, serialized, query, *, run_id, metadata=None, **kwargs):
        self._started[run_id] = time.time()
        self._node(run_id, metadata)
        self._queries[run_id] = str(query)

    def on_retriever_end(self, documents, *, run_id, **kwargs):
        chunks = [getattr(d, "page_content", str(d))[:1000] for d in (documents or [])]
        self._record(run_id, Step(StepKind.RETRIEVAL,
                                  content={"query": self._queries.pop(run_id, ""),
                                           "chunks": chunks},
                                  latency_s=self._elapsed(run_id)))

    # -- convenience -------------------------------------------------------------------

    @property
    def report(self) -> Dict[str, dict]:
        return self.mg.report()

    @property
    def system_risk(self) -> float:
        return self.mg.system_risk
