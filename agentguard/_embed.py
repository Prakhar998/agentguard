"""Embedding plumbing shared by semantic_drift and the failure memory.

``resolve_embed_fn`` picks the best available embedder:

1. whatever callable you pass in (OpenAI, Anthropic, anything text->vector)
2. sentence-transformers, if installed (``pip install agentguard[embeddings]``)
3. a hashed character-n-gram embedding — pure stdlib, no key, no model

The fallback is deliberately simple: it captures enough lexical overlap to
tell "the agent keeps saying the same thing" from "the agent is saying new
things", which is all semantic_drift needs to work keyless.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Callable, List, Optional, Sequence

EmbedFn = Callable[[str], Sequence[float]]

_HASH_DIM = 256


def hash_embed(text: str, dim: int = _HASH_DIM) -> List[float]:
    """Hashed character-trigram embedding. Deterministic, stdlib-only."""
    vec = [0.0] * dim
    text = re.sub(r"\s+", " ", text.lower().strip())
    if not text:
        return vec
    for i in range(max(1, len(text) - 2)):
        gram = text[i : i + 3]
        h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0 if (h >> 16) % 2 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


_st_embed_cache: Optional[EmbedFn] = None


def _sentence_transformers_embed() -> Optional[EmbedFn]:
    global _st_embed_cache
    if _st_embed_cache is not None:
        return _st_embed_cache
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(text: str) -> Sequence[float]:
        return model.encode([text], show_progress_bar=False)[0].tolist()

    _st_embed_cache = embed
    return embed


def resolve_embed_fn(embed_fn: Optional[EmbedFn] = None, prefer_local: bool = False) -> EmbedFn:
    if embed_fn is not None:
        return embed_fn
    if prefer_local:
        return hash_embed

    # lazy: pick sentence-transformers vs local fallback on first use, so
    # constructing a Guard never blocks on model loading
    backend: list = []

    def embed(text: str) -> Sequence[float]:
        if not backend:
            backend.append(_sentence_transformers_embed() or hash_embed)
        return backend[0](text)

    return embed


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
