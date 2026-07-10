"""Predictors must fire on known-bad step sequences and stay quiet on
healthy ones — same e2e regression discipline as the ProactiveGuard suite.

Run with: python -m pytest tests/ -q   (or python -m unittest)
"""

import unittest

from agentguard import Guard, Step, StepKind
from agentguard.adapters.raw import llm_output, tool_call, tool_result
from agentguard.aggregate import Aggregator, ConformalCalibrator
from agentguard.predictors.budget_drift import BudgetDriftPredictor
from agentguard.predictors.loop import LoopPredictor
from agentguard.predictors.tool_cascade import ToolCascadePredictor


def run_through(predictor, steps):
    predictor.reset()
    history, score = [], 0.0
    for s in steps:
        history.append(s)
        score = predictor.update(s, history)
    return score


def healthy_run(n_rounds=6):
    steps = []
    for i in range(n_rounds):
        steps.append(tool_call("search", {"q": f"distinct question number {i} about {'abcdefg'[i % 7]}"}))
        steps.append(tool_result("search", f"useful result {i}", tokens=50))
        steps.append(llm_output(f"Good progress on subtask {'abcdefg'[i % 7]}, moving on.", tokens=100))
    return steps


class TestLoopPredictor(unittest.TestCase):
    def test_quiet_on_healthy_run(self):
        score = run_through(LoopPredictor(), healthy_run())
        self.assertLess(score, 0.3)

    def test_fires_on_repeated_tool_call(self):
        steps = [tool_call("search", {"q": "same thing"}) for _ in range(5)]
        score = run_through(LoopPredictor(), steps)
        self.assertGreaterEqual(score, 0.8)

    def test_fires_on_cycle(self):
        steps = []
        for _ in range(4):
            steps.append(tool_call("search", {"q": "topic"}))
            steps.append(tool_call("summarize", {"text": "results"}))
        score = run_through(LoopPredictor(), steps)
        self.assertGreaterEqual(score, 0.6)

    def test_normalizes_trivial_variation(self):
        # page=1 / page=2 / page=3 is still the same action
        steps = [tool_call("search", {"q": f"same thing page {i}"}) for i in range(5)]
        score = run_through(LoopPredictor(), steps)
        self.assertGreaterEqual(score, 0.8)

    def test_fires_on_repeated_llm_output(self):
        steps = [llm_output("I need to search for more information.") for _ in range(5)]
        score = run_through(LoopPredictor(), steps)
        self.assertGreaterEqual(score, 0.8)


class TestToolCascadePredictor(unittest.TestCase):
    def test_quiet_on_healthy_run(self):
        score = run_through(ToolCascadePredictor(), healthy_run())
        self.assertEqual(score, 0.0)

    def test_single_error_is_noise(self):
        steps = healthy_run(2)
        steps.append(tool_result("search", "timeout", error=True))
        steps.extend(healthy_run(1))
        score = run_through(ToolCascadePredictor(), steps)
        self.assertLess(score, 0.2)

    def test_fires_on_error_cluster(self):
        steps = healthy_run(2)
        for _ in range(4):
            steps.append(tool_call("fetch", {"url": "distinct-每次-different"}))
            steps.append(tool_result("fetch", "connection refused", error=True))
        score = run_through(ToolCascadePredictor(), steps)
        self.assertGreaterEqual(score, 0.7)


class TestBudgetDriftPredictor(unittest.TestCase):
    def test_quiet_on_steady_budget(self):
        steps = [llm_output(f"step {i}", tokens=100) for i in range(20)]
        score = run_through(BudgetDriftPredictor(), steps)
        self.assertEqual(score, 0.0)

    def test_quiet_when_tokens_missing(self):
        steps = [llm_output(f"step {i}") for i in range(20)]
        score = run_through(BudgetDriftPredictor(), steps)
        self.assertEqual(score, 0.0)

    def test_fires_on_token_blowout(self):
        steps = [llm_output(f"step {i}", tokens=100) for i in range(6)]
        steps += [llm_output(f"veryverbose {i}", tokens=100 * (i + 5)) for i in range(8)]
        score = run_through(BudgetDriftPredictor(), steps)
        self.assertGreaterEqual(score, 0.8)


class TestGuardAPI(unittest.TestCase):
    def test_eight_line_api(self):
        guard = Guard(predictors=["loop", "tool_cascade", "budget_drift"])
        halted_at = None
        with guard.watch() as w:
            for i in range(40):
                w.record({"kind": "tool_call", "name": "search", "content": {"q": "loop"}})
                if w.risk > 0.8:
                    proposal = w.intervene("halt")
                    halted_at = proposal.step_index
                    break
        self.assertIsNotNone(halted_at)
        self.assertLess(halted_at, 15)
        self.assertEqual(w.interventions[0].action, "halt")

    def test_healthy_run_never_flags(self):
        guard = Guard()
        with guard.watch() as w:
            for s in healthy_run(8):
                w.record(s)
        self.assertLess(w.risk, 0.4)
        self.assertEqual(w.interventions, [])

    def test_record_accepts_dict_and_assigns_index(self):
        w = Guard().watch()
        w.record({"kind": "llm_output", "content": "hi", "tokens": 10})
        w.record(Step(StepKind.FINAL))
        self.assertEqual([s.index for s in w.history], [0, 1])

    def test_on_risk_fires_once_on_crossing(self):
        calls = []
        guard = Guard(on_risk=lambda watcher: calls.append(watcher.risk), threshold=0.5)
        with guard.watch() as w:
            for _ in range(10):
                w.record(tool_call("x", {"a": 1}))
        self.assertEqual(len(calls), 1)
        self.assertGreaterEqual(calls[0], 0.5)

    def test_invalid_intervention_rejected(self):
        w = Guard().watch()
        w.record(tool_call("x"))
        with self.assertRaises(ValueError):
            w.intervene("self_destruct")

    def test_subscores_exposed(self):
        w = Guard().watch()
        w.record(tool_call("x"))
        self.assertEqual(set(w.subscores), {"loop", "tool_cascade", "budget_drift"})


class TestAggregate(unittest.TestCase):
    def test_noisy_or_compounds_weak_signals(self):
        agg = Aggregator()
        alone = agg.fuse({"loop": 0.5})
        together = agg.fuse({"loop": 0.5, "tool_cascade": 0.5, "budget_drift": 0.5})
        self.assertGreater(together, alone)

    def test_single_strong_signal_dominates(self):
        agg = Aggregator()
        self.assertGreaterEqual(agg.fuse({"loop": 1.0}), 0.99)

    def test_conformal_calibration(self):
        cal = ConformalCalibrator().fit(
            scores=[0.1, 0.2, 0.15, 0.3, 0.25, 0.9, 0.95],
            failed=[False, False, False, False, False, True, True],
        )
        # a score above every healthy run's peak -> near 1
        self.assertGreater(cal.calibrate(0.8), 0.8)
        # a score below every healthy peak -> near 0
        self.assertLess(cal.calibrate(0.05), 0.2)
        # monotone
        self.assertLessEqual(cal.calibrate(0.2), cal.calibrate(0.6))


if __name__ == "__main__":
    unittest.main()
