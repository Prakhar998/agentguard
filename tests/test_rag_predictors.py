"""RAG guard predictors: retrieval_drift, grounding_gap, goal_drift.

Pinned to the stdlib hash embedder — deterministic, no network/models.
"""

import unittest

from agentguard import Guard
from agentguard._embed import hash_embed
from agentguard.adapters.raw import llm_output, retrieval
from agentguard.predictors.goal_drift import GoalDriftPredictor
from agentguard.predictors.grounding_gap import GroundingGapPredictor
from agentguard.predictors.retrieval_drift import RetrievalDriftPredictor


def run_through(predictor, steps):
    predictor.reset()
    history, score = [], 0.0
    for s in steps:
        history.append(s)
        score = predictor.update(s, history)
    return score


CORPUS = {
    "pricing": "Our enterprise plan costs $500 per month including support and SSO.",
    "refunds": "Refunds are processed within 14 days of a written cancellation request.",
    "security": "All data is encrypted at rest with AES-256 and in transit with TLS 1.3.",
    "onboarding": "New workspaces are provisioned automatically within five minutes.",
}


class TestRetrievalDrift(unittest.TestCase):
    def _p(self):
        return RetrievalDriftPredictor(embed_fn=hash_embed)

    def test_quiet_on_varied_relevant_retrievals(self):
        steps = [
            retrieval("enterprise plan monthly price", [CORPUS["pricing"]]),
            retrieval("refund processing time policy", [CORPUS["refunds"]]),
            retrieval("data encryption standards used", [CORPUS["security"]]),
            retrieval("workspace provisioning speed", [CORPUS["onboarding"]]),
        ]
        self.assertLess(run_through(self._p(), steps), 0.35)

    def test_fires_on_re_retrieval_loop(self):
        steps = [
            retrieval(f"pricing question attempt {i}", [CORPUS["pricing"]])
            for i in range(5)
        ]
        self.assertGreaterEqual(run_through(self._p(), steps), 0.8)

    def test_fires_on_relevance_starvation(self):
        steps = [
            retrieval("enterprise plan cost per month support", [CORPUS["pricing"]]),
            retrieval("refund cancellation request days", [CORPUS["refunds"]]),
            # queries the corpus can't answer -> retrieved text unrelated
            retrieval("what is the meaning of dreams in norse mythology", [CORPUS["security"]]),
            retrieval("recipe for sourdough bread starter hydration", [CORPUS["onboarding"]]),
        ]
        self.assertGreaterEqual(run_through(self._p(), steps), 0.6)

    def test_empty_retrieval_counts_as_starvation(self):
        steps = [
            retrieval("enterprise plan cost per month", [CORPUS["pricing"]]),
            retrieval("refund cancellation days policy", [CORPUS["refunds"]]),
            retrieval("nothing matches this", []),
            retrieval("still nothing matches", []),
        ]
        self.assertGreaterEqual(run_through(self._p(), steps), 0.6)

    def test_ignores_non_retrieval_steps(self):
        steps = [llm_output(f"thinking about topic {c}") for c in "abcdef"]
        self.assertEqual(run_through(self._p(), steps), 0.0)


class TestGroundingGap(unittest.TestCase):
    def _p(self):
        return GroundingGapPredictor(embed_fn=hash_embed)

    def test_quiet_when_outputs_stay_grounded(self):
        steps = [
            retrieval("pricing", [CORPUS["pricing"]]),
            llm_output("The enterprise plan costs $500 per month with support and SSO."),
            retrieval("refunds", [CORPUS["refunds"]]),
            llm_output("Refunds are processed within 14 days of a cancellation request."),
        ]
        self.assertLess(run_through(self._p(), steps), 0.3)

    def test_fires_when_outputs_drift_from_sources(self):
        steps = [
            retrieval("pricing", [CORPUS["pricing"]]),
            llm_output("The enterprise plan costs $500 per month including support."),
            llm_output("Also, the plan probably ships with a free llama and moon tours."),
            llm_output("Ancient sailors used celestial navigation across the ocean at night."),
        ]
        self.assertGreaterEqual(run_through(self._p(), steps), 0.6)

    def test_silent_without_retrievals(self):
        steps = [llm_output("no rag here, nothing to ground against")] * 4
        self.assertEqual(run_through(self._p(), steps), 0.0)


class TestGoalDrift(unittest.TestCase):
    def test_quiet_while_converging_on_goal(self):
        p = GoalDriftPredictor(goal="compute total enterprise revenue for Q3",
                               embed_fn=hash_embed)
        steps = [
            llm_output("First gathering the enterprise revenue figures for Q3."),
            llm_output("Enterprise revenue rows for Q3 loaded, summing them now."),
            llm_output("Total enterprise revenue for Q3 computed: summarizing."),
        ]
        self.assertLess(run_through(p, steps), 0.4)

    def test_fires_when_wandering_off_goal(self):
        p = GoalDriftPredictor(goal="compute total enterprise revenue for Q3",
                               embed_fn=hash_embed)
        steps = [
            llm_output("Gathering the enterprise revenue figures for Q3 revenue totals."),
            llm_output("Interesting — the logo colors changed in the latest brand refresh."),
            llm_output("The office dog schedule for next week looks quite busy overall."),
            llm_output("Medieval castles often had spiral staircases turning clockwise."),
        ]
        self.assertGreaterEqual(run_through(p, steps), 0.6)

    def test_silent_without_goal(self):
        p = GoalDriftPredictor(embed_fn=hash_embed)
        steps = [llm_output(f"random musing {c}") for c in "abcdef"]
        self.assertEqual(run_through(p, steps), 0.0)

    def test_goal_flows_from_watch_kwarg(self):
        guard = Guard(predictors=[GoalDriftPredictor(embed_fn=hash_embed)])
        w = guard.watch(goal="summarize the security policy")
        self.assertEqual(w.predictors[0]._goal_text, "summarize the security policy")


if __name__ == "__main__":
    unittest.main()
