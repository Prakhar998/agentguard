"""Learned risk model (requires numpy; skipped without it).

Trains a small ensemble on a reduced synthetic dataset and checks the
regression property that matters: high risk on failing runs, low on
healthy ones — the ProactiveGuard e2e discipline.
"""

import unittest

try:
    import numpy as np  # noqa: F401

    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

if HAVE_NUMPY:
    from agentguard import Guard
    from agentguard.adapters.raw import llm_output, tool_call, tool_result
    from agentguard.predictors.model import (
        NUM_FEATURES,
        LearnedRiskPredictor,
        ProactiveGuardNet,
        RiskEnsemble,
    )
    from agentguard.train import build_dataset


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
class TestLearnedModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = build_dataset(n_per_type=8, seed=7)
        cls.X, cls.y = X, y
        cls.model = RiskEnsemble(n_models=2, seed=7).fit(X, y)

    def test_feature_dimension(self):
        self.assertEqual(self.X.shape[1], NUM_FEATURES)

    def test_holdout_binary_accuracy(self):
        pred = self.model.predict(self.X)
        binary_acc = ((pred > 0) == (self.y > 0)).mean()
        self.assertGreater(binary_acc, 0.85)

    def test_focal_net_trains_alone(self):
        net = ProactiveGuardNet(hidden_dims=[32, 32], seed=7)
        net.fit(self.X[:500], self.y[:500], epochs=30)
        self.assertEqual(net.predict_proba(self.X[:10]).shape, (10, 3))
        # learned attention is a per-feature gate in (0, 1)
        att = net.feature_attention()
        self.assertEqual(att.shape, (NUM_FEATURES,))
        self.assertTrue(((att > 0) & (att < 1)).all())

    def test_live_predictor_separates_runs(self):
        guard = Guard(predictors=[LearnedRiskPredictor(model=self.model)])

        loop_watcher = guard.watch()
        for _ in range(8):
            loop_watcher.record(tool_call("search", {"q": "identical query"}))
            loop_watcher.record(llm_output("I still need more information.", tokens=100))

        healthy_watcher = guard.watch()
        for i in range(8):
            uid = "abcdefgh"[i] * 3
            healthy_watcher.record(tool_call("search", {"q": f"topic {uid} details"}))
            healthy_watcher.record(tool_result("search", f"result {uid}", tokens=40))
            healthy_watcher.record(
                llm_output(f"Found {uid}, moving to the next subtask.", tokens=110)
            )

        self.assertGreater(loop_watcher.risk, 0.6)
        self.assertLess(healthy_watcher.risk, 0.3)
        self.assertGreater(loop_watcher.risk, healthy_watcher.risk + 0.4)


if __name__ == "__main__":
    unittest.main()
