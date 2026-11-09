import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from label_audit.provider import BudgetExceeded, OpenAICompatibleProvider, ProviderError
from label_audit.server import make_server

ROOT = Path(__file__).resolve().parents[1]
TEMP = ROOT / ".build/test-tmp"
TEMP.mkdir(parents=True, exist_ok=True)


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        calls = self.requests
        class FakeHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append(body)
                response = {"choices": [{"finish_reason": "stop", "message": {"content": '{"issues":[],"questions":[]}'}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
                raw = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.tmp = tempfile.TemporaryDirectory(dir=TEMP)
        self.env = patch.dict(os.environ, {"LLM_BASE_URL": f"http://127.0.0.1:{self.server.server_port}/v1", "LLM_MODEL": "fake-test-model", "LLM_API_KEY": "test-secret", "LLM_INPUT_PRICE_PER_MILLION": "2", "LLM_OUTPUT_PRICE_PER_MILLION": "4"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.tmp.cleanup()

    def test_http_adapter_cache_usage_and_no_secret_in_cache(self):
        p = OpenAICompatibleProvider(Path(self.tmp.name))
        self.assertEqual(p.ask("instruction", {"classes": []}), {"issues": [], "questions": []})
        p.ask("instruction", {"classes": []})
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(p.cache_hits, 1)
        self.assertEqual(self.requests[0]["response_format"], {"type": "json_object"})
        self.assertEqual(p.usage()["input_tokens"], 10)
        self.assertAlmostEqual(p.usage()["estimated_cost"], .00004)
        self.assertNotIn("test-secret", "".join(x.read_text() for x in Path(self.tmp.name).glob("*.json")))

    def test_budget_stops_before_second_paid_request(self):
        p = OpenAICompatibleProvider(Path(self.tmp.name), max_calls=1)
        p.ask("instruction", {"a": 1})
        with self.assertRaises(BudgetExceeded):
            p.ask("instruction", {"a": 2})
        self.assertEqual(len(self.requests), 1)

    def test_cache_invalidates_when_prompt_payload_changes(self):
        p = OpenAICompatibleProvider(Path(self.tmp.name))
        p.ask("instruction", {"version": 1})
        p.ask("instruction", {"version": 2})
        self.assertEqual(len(self.requests), 2)

    def test_ollama_uses_its_supported_token_parameter(self):
        with patch.dict(os.environ, {"LLM_BASE_URL": "http://127.0.0.1:11434/v1"}):
            p = OpenAICompatibleProvider(Path(self.tmp.name))
            self.assertEqual(p.token_parameter, "max_tokens")

    def test_missing_configuration_fails_instead_of_demo_fallback(self):
        with patch.dict(os.environ, {"LLM_MODEL": ""}):
            with self.assertRaises(ProviderError):
                OpenAICompatibleProvider(Path(self.tmp.name))


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(dir=TEMP)
        cls.root = Path(cls.tmp.name)
        (cls.root / "data").mkdir()
        for name in ["taxonomy.json", "demo.csv"]:
            (cls.root / "data" / name).write_bytes((ROOT / "data" / name).read_bytes())
        cls.server = make_server(cls.root, 0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join(); cls.tmp.cleanup()

    def request(self, path, data=None, origin=None):
        headers = {"Content-Type": "application/json"}
        if origin:
            headers["Origin"] = origin
        req = urllib.request.Request(self.url + path, data=json.dumps(data).encode() if data is not None else None, headers=headers)
        return urllib.request.urlopen(req)

    def test_end_to_end_audit_feedback_export_preserves_original(self):
        original = "id,text,label\na,Не могу войти после смены пароля.,delivery\nb,Не могу войти после смены пароля.,account\n"
        with self.request("/api/audit", {"csv": original, "mode": "demo", "budget": 2}) as r:
            result = json.load(r)
        self.assertEqual(result["findings"][0]["decision"]["action"], "change")
        with self.request("/api/feedback", {"id": "a", "label": "account"}) as r:
            result = json.load(r)
        self.assertEqual(result["input_records"][0]["label"], "delivery")
        with self.request("/api/reviewed") as r:
            exported = r.read().decode("utf-8-sig")
        self.assertIn("a,Не могу войти после смены пароля.,account", exported)
        self.assertTrue(list((self.root / "runtime/runs").glob("*/feedback.jsonl")))

    def test_rejects_foreign_origin(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.request("/api/audit", {"csv": "x"}, origin="https://example.com")
        self.assertEqual(ctx.exception.code, 403)

    def test_reports_bad_csv_as_error(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.request("/api/audit", {"csv": "broken", "mode": "demo"})
        self.assertEqual(ctx.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
