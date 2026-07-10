"""Semantic drift (keyless local embeddings) and RAG failure memory.

These tests pin the stdlib hash embedder so they are deterministic and
run without sentence-transformers/network."""

import unittest

from agentguard import Guard
from agentguard.adapters.raw import llm_output, tool_call
from agentguard._embed import cosine, hash_embed
from agentguard.memory import FailureMemory, InMemoryStore
from agentguard.predictors.semantic_drift import SemanticDriftPredictor


def run_through(predictor, steps):
    predictor.reset()
    history, score = [], 0.0
    for s in steps:
        history.append(s)
        score = predictor.update(s, history)
    return score


class TestHashEmbed(unittest.TestCase):
    def test_similar_text_similar_vectors(self):
        a = hash_embed("I need to search for more information about failures")
        b = hash_embed("I need to search for more information about failures!")
        c = hash_embed("The capital of France is Paris, a city in Europe")
        self.assertGreater(cosine(a, b), 0.9)
        self.assertLess(cosine(a, c), 0.5)


def local_predictor():
    return SemanticDriftPredictor(embed_fn=hash_embed)


class TestSemanticDrift(unittest.TestCase):
    def test_quiet_on_progressing_run(self):
        steps = [
            llm_output("First I will gather the raw sales numbers for Q3."),
            llm_output("The dataset has 4,000 rows; now cleaning null entries."),
            llm_output("Cleaned. Computing month-over-month growth next."),
            llm_output("Growth is 12%. Drafting the final summary report now."),
        ]
        score = run_through(local_predictor(), steps)
        self.assertLess(score, 0.4)

    def test_fires_on_stall(self):
        steps = [
            llm_output("I still need more information about failure modes. Let me search.")
            for _ in range(5)
        ]
        score = run_through(local_predictor(), steps)
        self.assertGreaterEqual(score, 0.8)

    def test_fires_on_oscillation(self):
        a = "Plan A: query the database directly for the answer to this."
        b = "Actually, better to scrape the website for that same answer."
        steps = [llm_output(t) for t in [a, b, a, b, a, b]]
        score = run_through(local_predictor(), steps)
        self.assertGreaterEqual(score, 0.6)

    def test_ignores_non_llm_steps(self):
        steps = [tool_call("search", {"q": "x"}) for _ in range(10)]
        score = run_through(local_predictor(), steps)
        self.assertEqual(score, 0.0)


class TestFailureMemory(unittest.TestCase):
    def _looping_watcher(self, guard, run_id):
        w = guard.watch(run_id=run_id)
        for _ in range(12):
            w.record(tool_call("search", {"q": "same"}))
            w.record(tool_call("summarize", {"text": "same"}))
            if w.risk > 0.8 and not w.interventions:
                w.intervene("halt")
        return w

    def test_store_and_retrieve_similar_failure(self):
        memory = FailureMemory(embed_fn=hash_embed)
        guard = Guard(memory=memory)

        past = self._looping_watcher(guard, "past_run")
        past.close()  # had interventions -> stored
        self.assertEqual(len(memory), 1)

        live = self._looping_watcher(guard, "live_run")
        matches = live.explain(k=1)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["run_id"], "past_run")
        self.assertGreater(matches[0]["similarity"], 0.8)

    def test_healthy_run_not_stored(self):
        memory = FailureMemory(embed_fn=hash_embed)
        guard = Guard(memory=memory)
        with guard.watch() as w:
            w.record(llm_output("all good, finishing"))
        self.assertEqual(len(memory), 0)

    def test_persistence_roundtrip(self):
        import tempfile, os

        memory = FailureMemory(embed_fn=hash_embed)
        guard = Guard(memory=memory)
        self._looping_watcher(guard, "r1").close()

        path = os.path.join(tempfile.mkdtemp(), "failures.json")
        memory.store.save(path)
        restored = FailureMemory(store=InMemoryStore().load(path))
        self.assertEqual(len(restored), 1)


if __name__ == "__main__":
    unittest.main()
