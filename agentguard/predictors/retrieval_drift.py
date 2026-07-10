"""Retrieval-drift predictor (RAG, embeddings).

Watches RETRIEVAL steps for the two ways a RAG loop goes bad:

* **re-retrieval loops** — successive retrievals return near-identical
  chunk sets: the agent keeps asking the vector store the same thing and
  hoping for a different answer (the RAG flavor of the search loop).
* **retrieval starvation** — query→chunk relevance declining retrieval
  over retrieval: the agent has wandered past what the corpus can answer,
  and every retrieval is grasping at less-relevant straws.

Both are computed with embeddings (pluggable ``embed_fn``, keyless local
fallback) so they work against any vector store — AgentGuard only sees
the Step stream.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .._embed import EmbedFn, cosine, resolve_embed_fn
from ..telemetry import Step, StepKind
from .base import Predictor


def _retrieval_parts(step: Step) -> tuple:
    content = step.content if isinstance(step.content, dict) else {}
    query = str(content.get("query", ""))
    chunks = [str(c) for c in content.get("chunks", [])]
    return query, chunks


class RetrievalDriftPredictor(Predictor):
    name = "retrieval_drift"

    def __init__(
        self,
        embed_fn: Optional[EmbedFn] = None,
        duplicate_similarity: float = 0.95,
        repeat_saturation: int = 3,
        starvation_drop: float = 0.35,
        window: int = 5,
    ):
        self.embed_fn = resolve_embed_fn(embed_fn)
        self.duplicate_similarity = duplicate_similarity
        self.repeat_saturation = repeat_saturation  # this many dup retrievals -> 1.0
        self.starvation_drop = starvation_drop      # relevance drop from baseline -> 1.0
        self.window = window
        self._chunkset_embeddings: List[Sequence[float]] = []
        self._relevances: List[float] = []

    def reset(self) -> None:
        self._chunkset_embeddings = []
        self._relevances = []

    def update(self, step: Step, history: List[Step]) -> float:
        if step.kind != StepKind.RETRIEVAL:
            return self._score()
        if step.error:
            # a failed retrieval isn't drift; tool_cascade owns errors
            return self._score()

        query, chunks = _retrieval_parts(step)
        if not chunks:
            # empty result set is maximal starvation
            self._relevances.append(0.0)
            return self._score()

        chunk_emb = self.embed_fn(" ".join(chunks)[:4000])
        self._chunkset_embeddings.append(chunk_emb)
        if query:
            self._relevances.append(cosine(self.embed_fn(query), chunk_emb))

        return self._score()

    # -- signals -----------------------------------------------------------

    def _duplicate_score(self) -> float:
        """Trailing streak of retrievals near-identical to a previous one."""
        embs = self._chunkset_embeddings
        streak = 0
        for i in range(len(embs) - 1, 0, -1):
            best = max(cosine(embs[i], embs[j]) for j in range(max(0, i - self.window), i))
            if best >= self.duplicate_similarity:
                streak += 1
            else:
                break
        return streak / self.repeat_saturation

    def _starvation_score(self) -> float:
        """Relevance of recent retrievals vs the run's own early baseline."""
        rel = self._relevances
        if len(rel) < 3:
            return 0.0
        baseline = max(rel[: max(2, len(rel) // 2)])
        if baseline <= 0:
            return 0.0
        recent = sum(rel[-2:]) / 2
        drop = baseline - recent
        return drop / self.starvation_drop

    def _score(self) -> float:
        if len(self._chunkset_embeddings) < 2 and len(self._relevances) < 3:
            return 0.0
        return self._clip(max(self._duplicate_score(), self._starvation_score()))
