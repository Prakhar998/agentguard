"""Semantic-drift predictor (embeddings).

Embed every LLM output and watch how the run moves through semantic
space. Healthy runs make progress: consecutive outputs are related but
not identical. Stuck runs do one of two things:

* **stall** — consecutive outputs are near-identical (the model is
  restating itself), or
* **oscillate** — output N is closer to output N-2 than to N-1
  (A -> B -> A: bouncing between two thoughts).

``embed_fn`` is pluggable: pass your OpenAI/Anthropic/sentence-transformers
embedder, or let the keyless local fallback handle it (see
agentguard._embed).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .._embed import EmbedFn, cosine, resolve_embed_fn
from ..telemetry import Step, StepKind
from .base import Predictor


class SemanticDriftPredictor(Predictor):
    name = "semantic_drift"

    def __init__(
        self,
        embed_fn: Optional[EmbedFn] = None,
        stall_similarity: float = 0.97,
        oscillation_margin: float = 0.1,
        window: int = 6,
        saturation: int = 3,
    ):
        self.embed_fn = resolve_embed_fn(embed_fn)
        self.stall_similarity = stall_similarity
        self.oscillation_margin = oscillation_margin
        self.window = window
        self.saturation = saturation  # this many stuck outputs in-window -> 1.0
        self._embeddings: List[Sequence[float]] = []
        self._stuck_flags: List[bool] = []

    def reset(self) -> None:
        self._embeddings = []
        self._stuck_flags = []

    def update(self, step: Step, history: List[Step]) -> float:
        if step.kind != StepKind.LLM_OUTPUT:
            return self._current_score()

        text = step.content_text()
        if not text.strip():
            return self._current_score()

        emb = self.embed_fn(text)
        self._embeddings.append(emb)

        stuck = False
        if len(self._embeddings) >= 2:
            sim_prev = cosine(emb, self._embeddings[-2])
            if sim_prev >= self.stall_similarity:
                stuck = True  # stalling: restating the previous thought
            elif len(self._embeddings) >= 3:
                sim_prev2 = cosine(emb, self._embeddings[-3])
                if sim_prev2 - sim_prev >= self.oscillation_margin and sim_prev2 >= self.stall_similarity:
                    stuck = True  # oscillating: back where it was two outputs ago
        self._stuck_flags.append(stuck)

        return self._current_score()

    def _current_score(self) -> float:
        recent = self._stuck_flags[-self.window :]
        if not recent:
            return 0.0
        return self._clip(sum(recent) / self.saturation)
