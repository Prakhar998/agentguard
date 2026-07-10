"""Budget-drift predictor.

Token velocity (tokens/step) of the recent window versus the run's own
early baseline. A run that starts burning tokens far above its baseline is
usually in runaway verbosity or context expansion — the budget blows out
long before the run officially "fails".
"""

from __future__ import annotations

from typing import List

from ..telemetry import Step
from .base import Predictor


class BudgetDriftPredictor(Predictor):
    name = "budget_drift"

    def __init__(
        self,
        baseline_steps: int = 5,
        window: int = 5,
        drift_start: float = 1.5,
        drift_saturation: float = 5.0,
    ):
        # velocity ratios <= drift_start score 0; >= drift_saturation score 1
        self.baseline_steps = baseline_steps
        self.window = window
        self.drift_start = drift_start
        self.drift_saturation = drift_saturation
        self._token_counts: List[int] = []

    def reset(self) -> None:
        self._token_counts = []

    def update(self, step: Step, history: List[Step]) -> float:
        if step.tokens is not None and step.tokens > 0:
            self._token_counts.append(step.tokens)

        counts = self._token_counts
        if len(counts) < self.baseline_steps + self.window:
            return 0.0  # not enough of a baseline to judge drift

        baseline = sum(counts[: self.baseline_steps]) / self.baseline_steps
        if baseline <= 0:
            return 0.0
        current = sum(counts[-self.window :]) / self.window

        ratio = current / baseline
        score = (ratio - self.drift_start) / (self.drift_saturation - self.drift_start)
        return self._clip(score)
