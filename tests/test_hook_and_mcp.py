"""Claude Code hook protocol + MCP server tools."""

import io
import json
import shutil
import unittest
from contextlib import redirect_stdout
from unittest import mock

from agentguard import hook


class TestClaudeCodeHook(unittest.TestCase):
    def setUp(self):
        self._old_dir = hook.STATE_DIR
        hook.STATE_DIR = hook.tempfile.mkdtemp(prefix="agentguard-test-hook")

    def tearDown(self):
        shutil.rmtree(hook.STATE_DIR, ignore_errors=True)
        hook.STATE_DIR = self._old_dir

    def _invoke(self, event, argv=None):
        out = io.StringIO()
        with mock.patch("sys.stdin", io.StringIO(json.dumps(event))):
            with redirect_stdout(out):
                code = hook.main(argv or [])
        return code, out.getvalue().strip()

    def _pre(self, session, tool_input):
        return {"hook_event_name": "PreToolUse", "session_id": session,
                "tool_name": "Bash", "tool_input": tool_input}

    def test_quiet_session_stays_silent(self):
        code, out = self._invoke(self._pre("s1", {"command": "ls unique-dir-abc"}))
        self.assertEqual((code, out), (0, ""))

    def test_looping_session_asks_for_confirmation(self):
        outputs = []
        for _ in range(8):
            _, out = self._invoke(self._pre("s2", {"command": "pytest tests/x.py"}))
            outputs.append(out)
        payload = json.loads(outputs[-1])
        decision = payload["hookSpecificOutput"]
        self.assertEqual(decision["permissionDecision"], "ask")
        self.assertIn("loop", decision["permissionDecisionReason"])

    def test_post_tool_use_warns_not_blocks(self):
        # realistic session: Pre + Post pairs of the same repeating command
        for _ in range(8):
            self._invoke(self._pre("s3", {"command": "pytest tests/x.py"}))
            code, out = self._invoke(
                {"hook_event_name": "PostToolUse", "session_id": "s3",
                 "tool_name": "Bash",
                 "tool_response": {"output": "1 failed", "success": False}})
        payload = json.loads(out)
        self.assertIn("systemMessage", payload)
        self.assertNotIn("hookSpecificOutput", payload)  # warns, doesn't gate

    def test_garbage_stdin_never_breaks_host(self):
        out = io.StringIO()
        with mock.patch("sys.stdin", io.StringIO("not json at all")):
            with redirect_stdout(out):
                code = hook.main([])
        self.assertEqual(code, 0)

    def test_sessions_are_isolated(self):
        for _ in range(8):
            self._invoke(self._pre("loops", {"command": "same"}))
        code, out = self._invoke(self._pre("fresh", {"command": "ls other-thing"}))
        self.assertEqual(out, "")


try:
    import mcp  # noqa: F401

    HAVE_MCP = True
except ImportError:
    HAVE_MCP = False

if HAVE_MCP:
    from agentguard import mcp_server


@unittest.skipUnless(HAVE_MCP, "mcp not installed")
class TestMCPServer(unittest.TestCase):
    def setUp(self):
        mcp_server._watchers.clear()

    def test_start_record_risk_roundtrip(self):
        mcp_server.start_run("r1", threshold=0.8)
        result = None
        for _ in range(8):
            result = mcp_server.record_step("r1", kind="tool_call",
                                            name="search", content="same query")
        self.assertTrue(result["over_threshold"])
        self.assertGreaterEqual(result["risk"], 0.8)
        self.assertEqual(mcp_server.get_risk("r1")["steps"], 8)

    def test_end_run_stores_failures_and_explains(self):
        mcp_server.start_run("bad")
        for _ in range(8):
            mcp_server.record_step("bad", kind="tool_call", name="s", content="same")
        self.assertTrue(mcp_server.end_run("bad", failed=True)["remembered"])

        mcp_server.start_run("live")
        for _ in range(8):
            mcp_server.record_step("live", kind="tool_call", name="s", content="same")
        explained = mcp_server.explain_run("live")
        self.assertEqual(explained["similar_past_failures"][0]["run_id"], "bad")

    def test_unknown_run_rejected(self):
        with self.assertRaises(ValueError):
            mcp_server.get_risk("nope")

    def test_tools_registered(self):
        import asyncio

        tools = asyncio.run(mcp_server.server.list_tools())
        names = {t.name for t in tools}
        self.assertEqual(names, {"start_run", "record_step", "get_risk",
                                 "explain_run", "end_run"})


if __name__ == "__main__":
    unittest.main()
