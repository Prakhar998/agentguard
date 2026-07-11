"""Live sidecar: native ingest, OTLP/OpenInference mapping, dashboard."""

import json
import threading
import unittest
import urllib.request

from agentguard.serve import otlp_to_steps, serve


def post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def get(url):
    with urllib.request.urlopen(url) as r:
        return r.read()


class TestServeHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = serve(port=0)  # ephemeral port
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_native_ingest_and_runs(self):
        for _ in range(8):
            post(f"{self.base}/ingest", {
                "run_id": "native_loop",
                "step": {"kind": "tool_call", "name": "search",
                         "content": {"q": "same"}}})
        data = json.loads(get(f"{self.base}/runs"))
        run = next(r for r in data["runs"] if r["run_id"] == "native_loop")
        self.assertTrue(run["over_threshold"])
        self.assertEqual(run["steps"], 8)
        self.assertEqual(run["top_signal"], "loop")
        self.assertTrue(run["trajectory"])

    def test_batch_ingest(self):
        resp = post(f"{self.base}/ingest", {
            "run_id": "batch",
            "steps": [{"kind": "llm_output", "content": f"thought {c}"}
                      for c in "abc"]})
        self.assertEqual(resp["run_id"], "batch")
        self.assertIsNotNone(resp["risk"])

    def test_otlp_endpoint(self):
        span = {
            "traceId": "otlp_run_1", "name": "search",
            "startTimeUnixNano": "1", "endTimeUnixNano": "500000001",
            "attributes": [
                {"key": "openinference.span.kind", "value": {"stringValue": "TOOL"}},
                {"key": "input.value", "value": {"stringValue": "identical query"}},
                {"key": "output.value", "value": {"stringValue": "result"}},
            ],
        }
        payload = {"resourceSpans": [{"scopeSpans": [{"spans": [span] * 6}]}]}
        resp = post(f"{self.base}/v1/traces", payload)
        self.assertEqual(resp["agentguard_steps"], 12)  # call + result per span
        data = json.loads(get(f"{self.base}/runs"))
        run = next(r for r in data["runs"] if r["run_id"] == "otlp_run_1")
        self.assertGreater(run["risk"], 0.8)

    def test_dashboard_served(self):
        html = get(f"{self.base}/").decode()
        self.assertIn("AgentGuard — live runs", html)
        self.assertIn("prefers-color-scheme: dark", html)

    def test_bad_json_rejected(self):
        req = urllib.request.Request(f"{self.base}/ingest", data=b"nope",
                                     method="POST")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)


class TestOTLPMapping(unittest.TestCase):
    def _payload(self, spans):
        return {"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}

    def test_llm_span(self):
        span = {
            "traceId": "t1", "name": "ChatOpenAI",
            "startTimeUnixNano": "0", "endTimeUnixNano": "2000000000",
            "attributes": [
                {"key": "openinference.span.kind", "value": {"stringValue": "LLM"}},
                {"key": "output.value", "value": {"stringValue": "the answer"}},
                {"key": "llm.token_count.total", "value": {"intValue": "240"}},
            ],
        }
        [(run_id, step)] = otlp_to_steps(self._payload([span]))
        self.assertEqual((run_id, step["kind"]), ("t1", "llm_output"))
        self.assertEqual(step["tokens"], 240)
        self.assertAlmostEqual(step["latency_s"], 2.0)

    def test_retriever_span(self):
        span = {
            "traceId": "t2", "name": "retriever",
            "attributes": [
                {"key": "openinference.span.kind", "value": {"stringValue": "RETRIEVER"}},
                {"key": "input.value", "value": {"stringValue": "refund policy"}},
                {"key": "retrieval.documents.0.document.content",
                 "value": {"stringValue": "chunk A"}},
                {"key": "retrieval.documents.1.document.content",
                 "value": {"stringValue": "chunk B"}},
            ],
        }
        [(_, step)] = otlp_to_steps(self._payload([span]))
        self.assertEqual(step["kind"], "retrieval")
        self.assertEqual(step["content"]["chunks"], ["chunk A", "chunk B"])

    def test_error_status_marks_error(self):
        span = {
            "traceId": "t3", "name": "fetch", "status": {"code": 2},
            "attributes": [
                {"key": "openinference.span.kind", "value": {"stringValue": "TOOL"}},
            ],
        }
        steps = [s for _, s in otlp_to_steps(self._payload([span]))]
        self.assertTrue(steps[1]["error"])  # the tool_result carries the error

    def test_unknown_span_kinds_ignored(self):
        span = {"traceId": "t4", "name": "chain", "attributes": []}
        self.assertEqual(otlp_to_steps(self._payload([span])), [])


if __name__ == "__main__":
    unittest.main()
