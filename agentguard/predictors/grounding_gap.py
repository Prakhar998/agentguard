"""Grounding-gap predictor (RAG, embeddings).

After each RETRIEVAL, measure how close subsequent LLM outputs stay to
the retrieved context in embedding space. A grounded answer lives near
its sources; an output drifting away from everything that was retrieved
is a hallucination precursor.

This is groundedness (RAGAS-style) made *predictive*: scored live,
relative to the run's own best grounding, so the alarm fires while the
run can still be reset — not in a post-hoc eval.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .._embed import EmbedFn, cosine, resolve_embed_fn
from ..telemetry import Step, StepKind
from .base import Predictor


class GroundingGapPredictor(Predictor):
    name = "grounding_gap"

    def __init__(
        self,
        embed_fn: Optional[EmbedFn] = None,
        drop_scale: float = 0.35,
        absolute_floor: float = 0.05,
        window: int = 4,
        context_size: int = 3,
    ):
        self.embed_fn = resolve_embed_fn(embed_fn)
        self.drop_scale = drop_scale          # grounding drop from baseline -> 1.0
        self.absolute_floor = absolute_floor  # grounding below this is instantly bad
        self.window = window
        self.context_size = context_size      # how many recent retrievals form the context
        self._context_embeddings: List[Sequence[float]] = []
        self._groundings: List[float] = []

    def reset(self) -> None:
        self._context_embeddings = []
        self._groundings = []

    def update(self, step: Step, history: List[Step]) -> float:
        if step.kind == StepKind.RETRIEVAL and not step.error:
            content = step.content if isinstance(step.content, dict) else {}
            chunks = [str(c) for c in content.get("chunks", [])]
            if chunks:
                self._context_embeddings.append(self.embed_fn(" ".join(chunks)[:4000]))
            return self._score()

        if step.kind != StepKind.LLM_OUTPUT or not self._context_embeddings:
            return self._score()

        text = step.content_text()
        if not text.strip():
            return self._score()

        out_emb = self.embed_fn(text)
        # grounded against the *best-matching* recent retrieval, so an
        # answer only needs support from one of its sources
        grounding = max(
            cosine(out_emb, ctx)
            for ctx in self._context_embeddings[-self.context_size :]
        )
        self._groundings.append(grounding)
        return self._score()

    def _score(self) -> float:
        g = self._groundings
        if len(g) < 2:
            return 0.0

        recent = g[-self.window :]
        current = sum(recent[-2:]) / len(recent[-2:])

        if current <= self.absolute_floor:
            return 1.0

        baseline = max(g[: max(2, len(g) - 1)])
        drop = baseline - current
        return self._clip(drop / self.drop_scale)
