import importlib.util
import json
import unittest


FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed in this Python environment")
class BackendApiTests(unittest.TestCase):
    def test_health_contains_runtime_diagnostics(self):
        from backend_api import health

        payload = health()
        self.assertEqual(payload["status"], "ok")
        self.assertGreater(payload["standards_count"], 0)
        self.assertIn("rag_db_exists", payload)
        self.assertIn("categories", payload)
        self.assertIn("effluent_standard", payload["categories"])

    def test_empty_query_stream_returns_error_event(self):
        from backend_api import QueryRequest, query_stream

        events = list(query_stream(QueryRequest(query="   ")))
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].startswith("event: error"))
        self.assertIn("Query is required.", events[0])

    def test_sse_event_payload_is_json(self):
        from backend_api import sse_event

        event = sse_event("status", {"status": "done", "label": "已完成"})
        data_line = [line for line in event.splitlines() if line.startswith("data:")][0]
        payload = json.loads(data_line.replace("data:", "").strip())
        self.assertEqual(payload["status"], "done")


if __name__ == "__main__":
    unittest.main()
