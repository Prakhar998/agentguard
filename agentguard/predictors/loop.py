"""Loop predictor — the single most common agent failure.

Detects three flavors of "the agent is going in circles":

1. The same tool called with near-identical arguments, repeatedly.
2. A short cycle of tool calls (search -> summarize -> search -> ...)
   repeating.
3. The same LLM output text recurring.

The score rises with each repetition: one repeat is suspicious, three is
almost certainly a loop.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Optional, Tuple

from ..telemetry import Step, StepKind
from .base import Predictor


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace and digits so 'page=2' vs 'page=3'
    style trivial variations still count as the same action."""
    text = text.lower().strip()
    text = re.sub(r"\d+", "#", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _sig(name: Optional[str], content_text: str) -> str:
    digest = hashlib.md5(_normalize(content_text).encode()).hexdigest()[:12]
    return f"{name or '?'}::{digest}"


class LoopPredictor(Predictor):
    name = "loop"

    def __init__(self, max_cycle_len: int = 4, repeat_saturation: int = 4):
        # repeats needed (beyond the first occurrence) for score 1.0
        self.repeat_saturation = repeat_saturation
        self.max_cycle_len = max_cycle_len
        self._tool_sigs: List[str] = []
        self._llm_sigs: List[str] = []

    def reset(self) -> None:
        self._tool_sigs = []
        self._llm_sigs = []

    def update(self, step: Step, history: List[Step]) -> float:
        if step.kind == StepKind.TOOL_CALL:
            self._tool_sigs.append(_sig(step.name, step.content_text()))
        elif step.kind == StepKind.LLM_OUTPUT:
            self._llm_sigs.append(_sig(step.name, step.content_text()))

        score = max(
            self._repeat_score(self._tool_sigs),
            self._cycle_score(self._tool_sigs),
            self._repeat_score(self._llm_sigs),
        )
        return self._clip(score)

    # -- signals ---------------------------------------------------------

    def _repeat_score(self, sigs: List[str], window: int = 12) -> float:
        """Identical action repeated in the recent window."""
        if not sigs:
            return 0.0
        recent = sigs[-window:]
        repeats = recent.count(recent[-1]) - 1
        return repeats / self.repeat_saturation

    def _cycle_score(self, sigs: List[str]) -> float:
        """A period-p cycle (p >= 2) repeating at the tail of the run."""
        best = 0.0
        n = len(sigs)
        for period in range(2, self.max_cycle_len + 1):
            if n < period * 2:
                continue
            # count how many times the trailing period-block repeats
            block = sigs[n - period : n]
            repeats = 0
            pos = n - period
            while pos - period >= 0 and sigs[pos - period : pos] == block:
                repeats += 1
                pos -= period
            if repeats:
                best = max(best, repeats / (self.repeat_saturation - 1))
        return best
