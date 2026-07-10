"""Tool-cascade predictor.

Sliding-window tool-error rate. One failed tool call is noise; errors
clustering (three in the last five tool steps) is a cascade — the agent is
usually feeding a bad output into the next call.

This is the agent-world analogue of the rising WAL-fsync / heartbeat-latency
precursor that ProactiveGuard found most predictive for consensus nodes.
"""

from __future__ import annotations

from typing import List

from ..telemetry import Step, StepKind
from .base import Predictor


class ToolCascadePredictor(Predictor):
    name = "tool_cascade"

    def __init__(self, window: int = 8, noise_rate: float = 0.15, cascade_rate: float = 0.6):
        # error rates at/below noise_rate score 0; at/above cascade_rate score 1
        self.window = window
        self.noise_rate = noise_rate
        self.cascade_rate = cascade_rate
        self._tool_errors: List[bool] = []

    def reset(self) -> None:
        self._tool_errors = []

    def update(self, step: Step, history: List[Step]) -> float:
        if step.kind in (StepKind.TOOL_CALL, StepKind.TOOL_RESULT):
            self._tool_errors.append(bool(step.error))

        recent = self._tool_errors[-self.window :]
        if len(recent) < 2 or sum(recent) < 2:
            return 0.0  # a single error is noise, never a cascade

        rate = sum(recent) / len(recent)
        score = (rate - self.noise_rate) / (self.cascade_rate - self.noise_rate)

        # consecutive trailing errors are worse than scattered ones
        streak = 0
        for err in reversed(recent):
            if not err:
                break
            streak += 1
        if streak >= 3:
            score = max(score, 0.6 + 0.15 * (streak - 3))

        return self._clip(score)
