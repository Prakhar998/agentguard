"""Goal-drift predictor (embeddings).

Embed the run's stated goal once; embed every LLM output; track the
distance between them. Healthy runs converge toward the goal region (or
at least hold position while working); a lost run wanders steadily
*away* from the best position it ever reached.

Needs the goal text: pass it at ``guard.watch(goal="...")``. Without a
goal the predictor stays silent — no goal, no drift to measure.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .._embed import EmbedFn, cosine, resolve_embed_fn
from ..telemetry import Step, StepKind
from .base import Predictor


class GoalDriftPredictor(Predictor):
    name = "goal_drift"

    def __init__(
        self,
        goal: Optional[str] = None,
        embed_fn: Optional[EmbedFn] = None,
        drift_scale: float = 0.3,
        min_outputs: int = 3,
    ):
        self.embed_fn = resolve_embed_fn(embed_fn)
        self.drift_scale = drift_scale  # similarity lost from the run's peak -> 1.0
        self.min_outputs = min_outputs
        self._goal_text = goal
        self._goal_emb: Optional[Sequence[float]] = None
        self._sims: List[float] = []

    def bind(self, watcher) -> None:
        """Called by the Watcher; picks up watch(goal=...) if set."""
        if watcher.goal and not self._goal_text:
            self._goal_text = watcher.goal

    def reset(self) -> None:
        self._goal_emb = None
        self._sims = []

    def update(self, step: Step, history: List[Step]) -> float:
        if not self._goal_text or step.kind != StepKind.LLM_OUTPUT:
            return self._score()
        text = step.content_text()
        if not text.strip():
            return self._score()

        if self._goal_emb is None:
            self._goal_emb = self.embed_fn(self._goal_text)
        self._sims.append(cosine(self.embed_fn(text), self._goal_emb))
        return self._score()

    def _score(self) -> float:
        sims = self._sims
        if len(sims) < self.min_outputs:
            return 0.0
        peak = max(sims[:-1])
        current = sum(sims[-2:]) / 2
        return self._clip((peak - current) / self.drift_scale)
