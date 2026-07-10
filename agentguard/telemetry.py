"""The one telemetry schema every adapter normalizes into.

A run is a sequence of :class:`Step` objects. Predictors only ever see
Steps, so any agent framework (raw loop, LangChain, LlamaIndex, custom)
plugs in by emitting Steps.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class StepKind(str, Enum):
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    LLM_OUTPUT = "llm_output"
    FINAL = "final"


@dataclass
class Step:
    """A single event in an agent run.

    Only ``kind`` is required. A predictor that needs a field the caller
    didn't provide simply contributes nothing for that step.
    """

    kind: StepKind
    name: Optional[str] = None          # tool name or model name
    content: Any = None                 # tool args / tool result / LLM text
    tokens: Optional[int] = None        # tokens consumed by this step
    latency_s: Optional[float] = None   # wall time of this step
    error: bool = False                 # did this step fail?
    index: Optional[int] = None         # filled in by Watcher.record
    ts: float = field(default_factory=time.time)

    @classmethod
    def from_dict(cls, data: dict) -> "Step":
        """Build a Step from a plain dict (kind may be a string)."""
        data = dict(data)
        kind = data.pop("kind")
        if not isinstance(kind, StepKind):
            kind = StepKind(str(kind).lower())
        allowed = {"name", "content", "tokens", "latency_s", "error", "index", "ts"}
        fields = {k: v for k, v in data.items() if k in allowed}
        return cls(kind=kind, **fields)

    def content_text(self) -> str:
        """Best-effort string form of content, for hashing/embedding."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        try:
            import json

            return json.dumps(self.content, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return str(self.content)
