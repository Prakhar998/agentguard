"""LangChain adapter (skipped when langchain-core isn't installed)."""

import unittest
import uuid
from types import SimpleNamespace

try:
    import langchain_core  # noqa: F401

    HAVE_LC = True
except ImportError:
    HAVE_LC = False

if HAVE_LC:
    from agentguard import Guard, StepKind
    from agentguard.adapters.langchain import AgentGuardCallback, AgentGuardHalt


@unittest.skipUnless(HAVE_LC, "langchain-core not installed")
class TestAgentGuardCallback(unittest.TestCase):
    def test_tool_callbacks_become_steps(self):
        h = AgentGuardCallback(Guard())
        rid = uuid.uuid4()
        h.on_tool_start({"name": "search"}, "query text", run_id=rid)
        h.on_tool_end("some output", run_id=rid)
        h.on_tool_error(RuntimeError("boom"), run_id=uuid.uuid4())

        kinds = [s.kind for s in h.watcher.history]
        self.assertEqual(kinds, [StepKind.TOOL_CALL, StepKind.TOOL_RESULT,
                                 StepKind.TOOL_RESULT])
        self.assertTrue(h.watcher.history[2].error)
        self.assertIsNotNone(h.watcher.history[1].latency_s)

    def test_retriever_callbacks_become_retrieval_steps(self):
        h = AgentGuardCallback(Guard())
        rid = uuid.uuid4()
        docs = [SimpleNamespace(page_content="chunk one"),
                SimpleNamespace(page_content="chunk two")]
        h.on_retriever_start({}, "what is the refund policy", run_id=rid)
        h.on_retriever_end(docs, run_id=rid)

        step = h.watcher.history[-1]
        self.assertEqual(step.kind, StepKind.RETRIEVAL)
        self.assertEqual(step.content["query"], "what is the refund policy")
        self.assertEqual(step.content["chunks"], ["chunk one", "chunk two"])

    def test_auto_intervene_raises_halt(self):
        h = AgentGuardCallback(Guard(threshold=0.5), auto_intervene="halt")
        with self.assertRaises(AgentGuardHalt) as ctx:
            for _ in range(12):
                rid = uuid.uuid4()
                h.on_tool_start({"name": "search"}, "identical query", run_id=rid)
                h.on_tool_end("same result", run_id=rid)
        self.assertGreaterEqual(ctx.exception.watcher.risk, 0.5)
        self.assertEqual(ctx.exception.watcher.interventions[0].action, "halt")


if __name__ == "__main__":
    unittest.main()
