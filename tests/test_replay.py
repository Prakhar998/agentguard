"""Trace replay backtest: loaders, replay, summary metrics."""

import json
import os
import tempfile
import unittest

from agentguard import Guard
from agentguard.replay import demo_traces, load_traces, replay, summarize


class TestLoaders(unittest.TestCase):
    def _write(self, text):
        path = os.path.join(tempfile.mkdtemp(), "traces")
        with open(path, "w") as f:
            f.write(text)
        return path

    def test_jsonl_with_outcome_lines(self):
        lines = [
            {"run_id": "a", "kind": "tool_call", "name": "search", "content": {"q": "x"}},
            {"run_id": "b", "kind": "llm_output", "content": "hi"},
            {"run_id": "a", "kind": "llm_output", "content": "more"},
            {"run_id": "a", "outcome": "failed"},
        ]
        traces = load_traces(self._write("\n".join(json.dumps(l) for l in lines)))
        by_id = {rid: (outcome, steps) for rid, outcome, steps in traces}
        self.assertEqual(by_id["a"][0], "failed")
        self.assertEqual(len(by_id["a"][1]), 2)  # order within run preserved
        self.assertIsNone(by_id["b"][0])

    def test_json_array_of_runs(self):
        payload = [
            {"run_id": "r", "outcome": "success",
             "steps": [{"kind": "llm_output", "content": "done"}]}
        ]
        traces = load_traces(self._write(json.dumps(payload)))
        self.assertEqual(traces, [("r", "success", payload[0]["steps"])])


class TestReplay(unittest.TestCase):
    def test_backtest_on_demo_traces(self):
        guard = Guard(threshold=0.8)
        results = replay(demo_traces(n_per_type=5, seed=3), guard)
        summary = summarize(results, guard.threshold)

        self.assertGreaterEqual(summary["catch_rate"], 0.9)
        self.assertLessEqual(summary.get("false_alarm_rate", 0.0), 0.1)
        self.assertGreater(summary["mean_lead_steps"], 0)

    def test_malformed_steps_are_skipped(self):
        traces = [("bad", "failed", [
            {"kind": "not_a_kind"},
            {"kind": "tool_call", "name": "x", "content": {"a": 1}},
        ])]
        results = replay(traces, Guard())
        self.assertEqual(results[0].steps, 1)  # only the valid step recorded

    def test_flagged_at_is_first_crossing(self):
        steps = [{"kind": "tool_call", "name": "s", "content": {"q": "same"}}] * 12
        results = replay([("loop", "failed", steps)], Guard(threshold=0.5))
        r = results[0]
        self.assertIsNotNone(r.flagged_at)
        self.assertLess(r.flagged_at, 6)
        self.assertEqual(r.lead_steps, r.steps - 1 - r.flagged_at)


if __name__ == "__main__":
    unittest.main()
