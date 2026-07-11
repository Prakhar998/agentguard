"""Injection predictor — context poisoning arriving through the data plane.

Agents trust what their tools and retrievers hand back. Prompt injection
rides exactly that trust: instruction-shaped text embedded in a web page,
a retrieved chunk, an email body. This predictor scans **inbound data**
(TOOL_RESULT content, RETRIEVAL chunks — never the agent's own outputs)
for payload patterns and scores the run's exposure.

Purely defensive and fully explainable: every hit is a named pattern you
can read out in an incident review. One weak hit is noise; a strong
pattern or accumulating hits is an alarm.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from ..telemetry import Step, StepKind
from .base import Predictor

# (name, compiled pattern, weight) — weight is the score a single hit earns
PATTERNS: List[Tuple[str, "re.Pattern", float]] = [
    (
        "override-instructions",
        re.compile(r"(ignore|disregard|forget)\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)", re.I),
        1.0,
    ),
    (
        "role-reassignment",
        re.compile(r"\byou are (now|actually)\b|\bpretend (to be|you are)\b|\bact as (if|though|a)\b", re.I),
        0.7,
    ),
    (
        "system-prompt-probe",
        re.compile(r"(reveal|print|show|repeat|output).{0,30}(system prompt|initial instructions|hidden instructions)", re.I),
        1.0,
    ),
    (
        "new-instructions-block",
        re.compile(r"\b(new|real|actual|updated) (instructions?|task|objective)\s*[:\-]", re.I),
        0.8,
    ),
    (
        "exfiltration-directive",
        re.compile(r"(send|post|forward|upload).{0,40}(api key|token|password|credentials|secret|conversation)", re.I),
        1.0,
    ),
    (
        "concealment-directive",
        re.compile(r"do not (tell|inform|mention|reveal|show).{0,30}(the )?(user|human|anyone)", re.I),
        1.0,
    ),
    (
        "agent-addressing",
        re.compile(r"\b(dear|attention|hey|hello)?,?\s*(ai|llm|language model|assistant|agent)\s*[:,]\s*(please|you must|do)", re.I),
        0.8,
    ),
    (
        "hidden-html-imperative",
        re.compile(r"<!--.{0,200}\b(you must|please|instruction|ignore)\b.{0,200}-->", re.I | re.S),
        0.9,
    ),
    (
        "tool-invocation-smuggling",
        re.compile(r"\b(call|invoke|use|run) the [\w_]+ (tool|function) with\b", re.I),
        0.6,
    ),
]


class InjectionPredictor(Predictor):
    name = "injection"

    def __init__(self, decay: float = 0.85):
        # exposure decays as clean steps go by, so one old weak hit
        # doesn't haunt a long healthy run — but strong hits saturate.
        self.decay = decay
        self._exposure = 0.0
        self.hits: List[dict] = []  # audit trail: what fired, where

    def reset(self) -> None:
        self._exposure = 0.0
        self.hits = []

    @staticmethod
    def _inbound_text(step: Step) -> str:
        if step.kind == StepKind.TOOL_RESULT:
            return step.content_text()
        if step.kind == StepKind.RETRIEVAL and isinstance(step.content, dict):
            return " ".join(str(c) for c in step.content.get("chunks", []))
        return ""

    def update(self, step: Step, history: List[Step]) -> float:
        text = self._inbound_text(step)
        if not text:
            return self._clip(self._exposure)

        step_score = 0.0
        for name, pattern, weight in PATTERNS:
            m = pattern.search(text)
            if m:
                step_score = max(step_score, weight)
                self.hits.append(
                    {"step": step.index, "pattern": name, "match": m.group(0)[:120]}
                )

        if step_score:
            # accumulate: a second hit on top of lingering exposure saturates
            self._exposure = min(1.0, self._exposure * 0.5 + step_score)
        else:
            self._exposure *= self.decay

        return self._clip(self._exposure)
