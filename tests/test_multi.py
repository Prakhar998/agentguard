"""MultiGuard cascade prediction + the LangGraph routing callback."""

import unittest
import uuid

from agentguard.adapters.raw import llm_output, tool_call
from agentguard.multi import MultiGuard

try:
    import langchain_core  # noqa: F401

    HAVE_LC = True
except ImportError:
    HAVE_LC = False

if HAVE_LC:
    from agentguard.adapters.langchain import AgentGuardHalt
    from agentguard.adapters.langgraph import MultiAgentCallback


def make_looping(mg, agent, n=10):
    for _ in range(n):
        mg.record(agent, tool_call("search", {"q": "same"}))


def make_healthy(mg, agent, n=5):
    for i in range(n):
        uid = "abcdefgh"[i % 8] * 3
        mg.record(agent, tool_call("search", {"q": f"topic {uid}"}))
        mg.record(agent, llm_output(f"progress on {uid}, next differs.", tokens=100))


class TestMultiGuard(unittest.TestCase):
    def test_isolated_agents_do_not_infect(self):
        mg = MultiGuard()
        make_looping(mg, "a")
        make_healthy(mg, "b")
        self.assertGreater(mg.risk("a"), 0.9)
        self.assertLess(mg.effective_risk("b"), 0.2)  # no edge, no contagion

    def test_contagion_flows_along_edges(self):
        mg = MultiGuard(coupling=0.5)
        mg.add_edge("a", "b")
        make_looping(mg, "a")
        make_healthy(mg, "b")
        self.assertLess(mg.risk("b"), 0.2)
        self.assertGreater(mg.effective_risk("b"), 0.4)  # >= coupling * risk(a)

    def test_contagion_is_one_hop(self):
        mg = MultiGuard(coupling=0.5)
        mg.add_edge("a", "b").add_edge("b", "c")
        make_looping(mg, "a")
        make_healthy(mg, "b")
        make_healthy(mg, "c")
        # c's upstream is b, whose OWN risk is low
        self.assertLess(mg.effective_risk("c"), 0.25)

    def test_system_risk_and_worst_agent(self):
        mg = MultiGuard()
        make_healthy(mg, "x")
        self.assertLess(mg.system_risk, 0.3)
        make_looping(mg, "y")
        self.assertGreater(mg.system_risk, 0.9)
        self.assertEqual(mg.worst_agent(), "y")

    def test_report_shape(self):
        mg = MultiGuard()
        mg.add_edge("a", "b")
        make_healthy(mg, "a", 2)
        report = mg.report()
        self.assertEqual(set(report), {"a", "b"})
        self.assertEqual(report["b"]["upstream"], ["a"])
        for key in ("risk", "effective_risk", "steps", "subscores"):
            self.assertIn(key, report["a"])


@unittest.skipUnless(HAVE_LC, "langchain-core not installed")
class TestMultiAgentCallback(unittest.TestCase):
    def test_routes_by_langgraph_node_metadata(self):
        h = MultiAgentCallback(MultiGuard())
        for node in ("researcher", "writer"):
            rid = uuid.uuid4()
            h.on_tool_start({"name": "search"}, "query", run_id=rid,
                            metadata={"langgraph_node": node})
            h.on_tool_end("result", run_id=rid)
        self.assertEqual(set(h.mg.agents), {"researcher", "writer"})
        self.assertEqual(len(h.mg.watcher("researcher").history), 2)

    def test_missing_metadata_falls_back_to_graph(self):
        h = MultiAgentCallback(MultiGuard())
        rid = uuid.uuid4()
        h.on_tool_start({"name": "search"}, "query", run_id=rid)
        h.on_tool_end("result", run_id=rid)
        self.assertEqual(h.mg.agents, ["graph"])

    def test_auto_intervene_names_the_agent(self):
        h = MultiAgentCallback(MultiGuard(threshold=0.5), auto_intervene="halt")
        with self.assertRaises(AgentGuardHalt) as ctx:
            for _ in range(12):
                rid = uuid.uuid4()
                h.on_tool_start({"name": "s"}, "same query", run_id=rid,
                                metadata={"langgraph_node": "researcher"})
                h.on_tool_end("same", run_id=rid)
        self.assertIn("researcher", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
