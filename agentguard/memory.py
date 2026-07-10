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
import random
import re
from typing import Dict, List, Optional, Sequence

from ._embed import EmbedFn, cosine, resolve_embed_fn


def _kmeans(vectors: List[List[float]], k: int, iterations: int) -> tuple:
    rng = random.Random(42)
    centroids = [list(v) for v in rng.sample(vectors, k)]
    assign = [0] * len(vectors)
    for _ in range(iterations):
        changed = False
        for i, v in enumerate(vectors):
            best = max(range(k), key=lambda c: cosine(v, centroids[c]))
            if best != assign[i]:
                assign[i] = best
                changed = True
        for c in range(k):
            members = [vectors[i] for i, a in enumerate(assign) if a == c]
            if members:
                centroids[c] = [sum(col) / len(members) for col in zip(*members)]
        if not changed:
            break
    return assign, centroids


def _silhouette(vectors: List[List[float]], assign: List[int]) -> float:
    """Mean silhouette on cosine distance; cheap and fine at memory scale."""
    def dist(a, b):
        return 1.0 - cosine(a, b)

    total, counted = 0.0, 0
    clusters = set(assign)
    for i, v in enumerate(vectors):
        own = [j for j, a in enumerate(assign) if a == assign[i] and j != i]
        if not own:
            continue
        a = sum(dist(v, vectors[j]) for j in own) / len(own)
        b = min(
            (
                sum(dist(v, vectors[j]) for j, aj in enumerate(assign) if aj == c)
                / max(1, sum(1 for aj in assign if aj == c))
            )
            for c in clusters
            if c != assign[i]
        )
        total += (b - a) / (max(a, b) or 1.0)
        counted += 1
    return total / counted if counted else -1.0

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


def trajectory_embed(watcher, points: int = 8) -> List[float]:
    """Fixed-length embedding of the run's sub-score *time series*.

    Retrieval on this vector matches failure *shape* — a slow-ramp loop
    finds other slow-ramp loops, a sudden cascade finds sudden cascades —
    independent of what any signature text says. Each signal's trajectory
    is resampled to ``points`` values by linear interpolation, then peak
    risk and normalized length are appended.
    """
    traj = watcher.subscore_trajectory
    names = sorted({name for snap in traj for name in snap})
    vec: List[float] = []
    for name in names or ["_"]:
        series = [snap.get(name, 0.0) for snap in traj] or [0.0]
        if len(series) == 1:
            vec.extend(series * points)
            continue
        for i in range(points):
            pos = i * (len(series) - 1) / (points - 1)
            lo = int(pos)
            frac = pos - lo
            hi = min(lo + 1, len(series) - 1)
            vec.append(series[lo] * (1 - frac) + series[hi] * frac)
    vec.append(max(watcher.risk_trajectory, default=0.0))
    vec.append(min(1.0, len(traj) / 50.0))
    return vec


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z_]{3,}", text.lower()))


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
                "trajectory": trajectory_embed(watcher),
            },
        )
        return text

    def similar_failures(self, watcher, k: int = 3, mode: str = "hybrid") -> List[Dict]:
        """k nearest past failures to the current run's live signature.

        ``mode``:
          * ``dense`` — cosine over signature-text embeddings
          * ``trajectory`` — cosine over sub-score time-series embeddings
            (matches failure *shape*)
          * ``keyword`` — token overlap on signature text
          * ``hybrid`` (default) — reciprocal-rank fusion of all three
        """
        if len(self.store) == 0:
            return []
        text = run_signature_text(watcher)
        if mode == "dense" or not isinstance(self.store, InMemoryStore):
            return self.store.search(self.embed_fn(text), k=k)

        entries = list(zip(self.store._vectors, self.store._metadata))
        query_dense = self.embed_fn(text)
        query_traj = trajectory_embed(watcher)
        query_tokens = _tokens(text)

        def ranking(scores: List[float]) -> List[int]:
            return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        dense = [cosine(query_dense, v) for v, _ in entries]
        traj = [
            cosine(query_traj, m.get("trajectory", [])) if m.get("trajectory") else 0.0
            for _, m in entries
        ]
        keyword = [
            (len(query_tokens & _tokens(m.get("signature", ""))) /
             (len(query_tokens | _tokens(m.get("signature", ""))) or 1))
            for _, m in entries
        ]

        if mode == "trajectory":
            fused = traj
        elif mode == "keyword":
            fused = keyword
        else:  # reciprocal-rank fusion, the standard hybrid-search combiner
            fused = [0.0] * len(entries)
            for scores in (dense, traj, keyword):
                for rank, idx in enumerate(ranking(scores)):
                    fused[idx] += 1.0 / (60 + rank)

        out = []
        for idx in ranking(fused)[:k]:
            meta = dict(entries[idx][1])
            meta.pop("trajectory", None)
            meta["similarity"] = dense[idx]
            meta["score"] = fused[idx]
            out.append(meta)
        return out

    # -- failure taxonomy ------------------------------------------------

    def taxonomy(self, max_k: int = 5, iterations: int = 25) -> List[Dict]:
        """Cluster stored failures into a taxonomy of failure modes.

        Pure-python k-means over trajectory embeddings (shape of failure),
        k chosen by silhouette. Returns clusters sorted by size, each with
        an exemplar (the run closest to the centroid).
        """
        if not isinstance(self.store, InMemoryStore) or len(self.store) < 2:
            return []
        entries = [
            (m.get("trajectory") or list(v), m)
            for v, m in zip(self.store._vectors, self.store._metadata)
        ]
        dim = max(len(t) for t, _ in entries)
        vectors = [t + [0.0] * (dim - len(t)) for t, _ in entries]
        metas = [m for _, m in entries]

        best: tuple = (None, -2.0)  # (assignment, silhouette)
        for k in range(2, min(max_k, len(vectors)) + 1):
            assign, centroids = _kmeans(vectors, k, iterations)
            score = _silhouette(vectors, assign)
            if score > best[1]:
                best = ((assign, centroids), score)
        if best[0] is None:
            return []
        assign, centroids = best[0]

        clusters: List[Dict] = []
        for c in range(len(centroids)):
            members = [i for i, a in enumerate(assign) if a == c]
            if not members:
                continue
            exemplar = max(members, key=lambda i: cosine(vectors[i], centroids[c]))
            clusters.append(
                {
                    "size": len(members),
                    "exemplar": metas[exemplar].get("summary", ""),
                    "run_ids": [metas[i].get("run_id") for i in members],
                    "mean_peak_risk": sum(metas[i].get("peak_risk", 0.0) for i in members)
                    / len(members),
                }
            )
        clusters.sort(key=lambda cl: cl["size"], reverse=True)
        return clusters
