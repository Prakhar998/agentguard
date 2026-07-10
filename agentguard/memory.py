"""Failure memory — RAG over past failure signatures.

When a run goes bad, its *signature* (sub-score trajectory + tool-sequence
tail + a short summary of how it failed) is embedded and stored in a
vector store. When a new run's risk climbs, retrieve the k nearest past
failures to explain the alarm:

    "this looks like the 14 past runs that looped on search→summarize→search"

This is retrieval-augmented *explanation*: the retrieval result augments
the alert a human sees, not a prompt. Backends: in-memory (default,
stdlib), Chroma or FAISS via ``pip install agentguard[memory]``.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

from ._embed import EmbedFn, cosine, resolve_embed_fn

# -- vector store backends ---------------------------------------------------


class InMemoryStore:
    """Exact cosine k-NN over python lists. Fine for thousands of runs."""

    def __init__(self) -> None:
        self._vectors: List[Sequence[float]] = []
        self._metadata: List[dict] = []

    def add(self, vector: Sequence[float], metadata: dict) -> None:
        self._vectors.append(vector)
        self._metadata.append(metadata)

    def search(self, vector: Sequence[float], k: int) -> List[dict]:
        scored = [
            {"similarity": cosine(vector, v), **meta}
            for v, meta in zip(self._vectors, self._metadata)
        ]
        scored.sort(key=lambda item: item["similarity"], reverse=True)
        return scored[:k]

    def __len__(self) -> int:
        return len(self._vectors)

    # optional persistence so a demo survives restarts
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"vectors": [list(v) for v in self._vectors],
                       "metadata": self._metadata}, f)

    def load(self, path: str) -> "InMemoryStore":
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            self._vectors = data["vectors"]
            self._metadata = data["metadata"]
        return self


class ChromaStore:
    """Chroma-backed store (``pip install agentguard[memory]``)."""

    def __init__(self, path: Optional[str] = None, collection: str = "agentguard_failures"):
        import chromadb  # noqa: import guarded by extra

        client = chromadb.PersistentClient(path=path) if path else chromadb.Client()
        self._collection = client.get_or_create_collection(collection)
        self._count = self._collection.count()

    def add(self, vector: Sequence[float], metadata: dict) -> None:
        self._count += 1
        flat = {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}
        self._collection.add(
            ids=[f"failure_{self._count}"],
            embeddings=[list(vector)],
            metadatas=[flat],
            documents=[metadata.get("summary", "")],
        )

    def search(self, vector: Sequence[float], k: int) -> List[dict]:
        res = self._collection.query(query_embeddings=[list(vector)], n_results=k)
        out = []
        for i in range(len(res["ids"][0])):
            meta = dict(res["metadatas"][0][i] or {})
            meta["summary"] = res["documents"][0][i]
            # chroma returns distance; convert to a similarity-flavored score
            meta["similarity"] = 1.0 - float(res["distances"][0][i]) / 2.0
            out.append(meta)
        return out

    def __len__(self) -> int:
        return self._collection.count()


# -- signatures ----------------------------------------------------------------


def run_signature_text(watcher) -> str:
    """Human-readable + embeddable fingerprint of a run's failure shape."""
    sig = watcher.signature()
    subs = sig["subscores"]
    dominant = max(subs, key=subs.get) if subs else "none"
    tools = " -> ".join(sig["tool_sequence_tail"]) or "no tools"
    return (
        f"dominant signal {dominant}; "
        + "; ".join(f"{k}={v:.2f}" for k, v in subs.items())
        + f"; peak risk {sig['peak_risk']:.2f} over {sig['steps']} steps; "
        f"tool tail: {tools}"
    )


class FailureMemory:
    """Store failure signatures; retrieve similar ones to explain new alarms.

    Pass an instance as ``Guard(memory=...)`` — the Watcher stores any run
    that had interventions on close, and ``Watcher.explain()`` retrieves.
    """

    def __init__(self, embed_fn: Optional[EmbedFn] = None, store=None):
        self.embed_fn = resolve_embed_fn(embed_fn)
        self.store = store if store is not None else InMemoryStore()

    def __len__(self) -> int:
        return len(self.store)

    def add_run(self, watcher, summary: Optional[str] = None) -> str:
        """Store a finished (failed/intervened) run. Returns the signature text."""
        text = run_signature_text(watcher)
        sig = watcher.signature()
        self.store.add(
            self.embed_fn(text),
            {
                "summary": summary or text,
                "signature": text,
                "run_id": sig["run_id"],
                "peak_risk": sig["peak_risk"],
                "steps": sig["steps"],
            },
        )
        return text

    def similar_failures(self, watcher, k: int = 3) -> List[Dict]:
        """k nearest past failures to the current run's live signature."""
        if len(self.store) == 0:
            return []
        text = run_signature_text(watcher)
        return self.store.search(self.embed_fn(text), k=k)
