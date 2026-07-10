"""Predictor interface.

A predictor is a small online estimator: it sees each Step as it happens
(plus the history so far) and returns its current belief, in [0, 1], that
the run is heading toward failure. Predictors must be cheap — they run on
every step of a live run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..telemetry import Step


class Predictor(ABC):
    """Base class for all failure predictors."""

    #: short identifier used in Guard(predictors=[...]) and subscores
    name: str = "predictor"

    @abstractmethod
    def update(self, step: Step, history: List[Step]) -> float:
        """Consume one step, return current risk sub-score in [0, 1].

        ``history`` includes ``step`` as its last element.
        """

    def reset(self) -> None:
        """Clear per-run state. Called when a new Watcher starts."""

    @staticmethod
    def _clip(score: float) -> float:
        return max(0.0, min(1.0, score))
