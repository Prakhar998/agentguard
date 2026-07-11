"""Intervention policies + escalation notifiers."""

import unittest

from agentguard import Guard
from agentguard.adapters.raw import llm_output, tool_call
from agentguard.policy import Policy, Rule, SlackWebhook, Webhook


def loop_run(guard, n=12):
    w = guard.watch()
    for _ in range(n):
        w.record(tool_call("search", {"q": "same"}))
    return w


class TestRules(unittest.TestCase):
    def test_invalid_action_rejected(self):
        with self.assertRaises(ValueError):
            Rule(action="explode")

    def test_signal_rule_fires_and_records_intervention(self):
        policy = Policy([Rule(signal="loop", threshold=0.8, action="reset_context")])
        w = loop_run(Guard(on_step=policy))
        self.assertEqual(len(w.interventions), 1)
        self.assertEqual(w.interventions[0].action, "reset_context")

    def test_fires_once_by_default(self):
        policy = Policy([Rule(signal="loop", threshold=0.5, action="halt")])
        w = loop_run(Guard(on_step=policy), n=20)
        self.assertEqual(len(w.interventions), 1)

    def test_fused_risk_rule(self):
        policy = Policy([Rule(threshold=0.9, action="escalate")])
        w = loop_run(Guard(on_step=policy))
        self.assertEqual(w.interventions[0].action, "escalate")

    def test_sustain_requires_consecutive_steps(self):
        policy = Policy([Rule(signal="budget_drift", threshold=0.5, sustain=3,
                              action="downgrade")])
        guard = Guard(on_step=policy)
        w = guard.watch()
        for i in range(6):
            w.record(llm_output(f"baseline {i}", tokens=100))
        self.assertEqual(w.interventions, [])  # steady budget: never fires
        for i in range(8):
            w.record(llm_output(f"blowout {i}", tokens=500 + 200 * i))
        self.assertEqual(len(w.interventions), 1)

    def test_healthy_run_untouched(self):
        policy = Policy([Rule(threshold=0.8, action="halt")])
        guard = Guard(on_step=policy)
        w = guard.watch()
        for i in range(8):
            w.record(tool_call("search", {"q": f"distinct {chr(97 + i) * 3}"}))
        self.assertEqual(w.interventions, [])

    def test_state_isolated_per_watcher(self):
        policy = Policy([Rule(signal="loop", threshold=0.5, action="halt")])
        guard = Guard(on_step=policy)
        loop_run(guard)
        w2 = loop_run(guard)  # second run must fire independently
        self.assertEqual(len(w2.interventions), 1)


class TestNotifiers(unittest.TestCase):
    def _policy_with(self, notifier):
        return Policy([Rule(signal="loop", threshold=0.8, action="halt")],
                      notifiers=[notifier])

    def test_webhook_payload(self):
        sent = []
        hook = Webhook("http://example.test/x",
                       transport=lambda url, body: sent.append((url, body)))
        w = loop_run(Guard(on_step=self._policy_with(hook)))
        self.assertEqual(len(sent), 1)
        import json

        payload = json.loads(sent[0][1])
        self.assertEqual(payload["run_id"], w.run_id)
        self.assertEqual(payload["action"], "halt")
        self.assertEqual(payload["rule"]["signal"], "loop")
        self.assertIn("subscores", payload)

    def test_slack_webhook_text(self):
        sent = []
        hook = SlackWebhook("http://example.test/slack",
                            transport=lambda url, body: sent.append(body))
        loop_run(Guard(on_step=self._policy_with(hook)))
        import json

        text = json.loads(sent[0])["text"]
        self.assertIn("AgentGuard", text)
        self.assertIn("loop=", text)

    def test_broken_notifier_never_kills_the_run(self):
        def bomb(*args):
            raise RuntimeError("webhook down")

        w = loop_run(Guard(on_step=self._policy_with(bomb)))
        self.assertEqual(len(w.interventions), 1)  # intervention still recorded


if __name__ == "__main__":
    unittest.main()
