import io
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

from label_audit.provider import BudgetExceeded, InvalidModelResponse, OpenAICompatibleProvider, ProviderError, RateLimitExceeded, PROMPTS, retry_delay
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

    def test_prompt_update_sends_new_system_instruction_and_invalidates_cache(self):
        p = OpenAICompatibleProvider(Path(self.tmp.name))
        payload = {"taxonomy": {"classes": []}}
        p.ask("instruction", payload)
        updated = PROMPTS["instruction"] + "\nУточненное правило границы классов."
        with patch.dict(PROMPTS, {"instruction": updated}):
            p.ask("instruction", payload)
            p.ask("instruction", payload)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(p.cache_hits, 1)
        self.assertTrue(self.requests[1]["messages"][0]["content"].startswith(updated))
        self.assertEqual(json.loads(self.requests[1]["messages"][1]["content"]), payload)

    def test_gigachat_queue_schema_limits_ids_and_count_for_this_request(self):
        p = OpenAICompatibleProvider(Path(self.tmp.name))
        payload = {"candidates": [{"id": "a"}, {"id": "b"}], "budget": 1, "selection_size": 1}
        with patch.object(p, "is_gigachat", True), patch.object(p, "authorization", return_value="test-token"):
            p.ask("queue", payload)
            payload = {**payload, "budget": 20, "selection_size": 2}
            p.ask("queue", payload)
        self.assertEqual(len(self.requests), 2)
        for body, count in zip(self.requests, [1, 2]):
            fmt = body["response_format"]
            self.assertEqual(fmt["type"], "json_schema")
            self.assertTrue(fmt["strict"])
            selected = fmt["schema"]["properties"]["selected_ids"]
            self.assertEqual(selected["items"], {"type": "string", "enum": ["a", "b"]})
            self.assertEqual(selected["minItems"], count)
            self.assertEqual(selected["maxItems"], count)

    def test_gigachat_semantic_schema_bounds_each_generated_field(self):
        p = OpenAICompatibleProvider(Path(self.tmp.name))
        payload = {"text": "Не могу подтвердить вход.",
                   "classes": [{"id": "account"}, {"id": "technical"}], "rules": []}
        with patch.object(p, "is_gigachat", True), patch.object(p, "authorization", return_value="test-token"):
            p.ask("semantic", payload)
        fields = self.requests[0]["response_format"]["schema"]["properties"]
        self.assertEqual(fields["labels"]["items"]["enum"], ["account", "technical"])
        self.assertEqual(fields["labels"]["maxItems"], 2)
        self.assertEqual(fields["evidence"]["maxLength"], len(payload["text"]))
        self.assertEqual(fields["evidence"]["minLength"], 1)
        self.assertEqual(fields["reason"]["maxLength"], 320)
        scores = fields["confidence"]["enum"]
        self.assertTrue(all(0 <= value <= 1 for value in scores))
        self.assertIn(0.65, scores)
        self.assertIn(0, scores)
        self.assertIn(1, scores)
        self.assertNotIn("reasoning", self.requests[0])

    def test_length_error_reports_usage_without_claiming_reasoning_or_retrying(self):
        response = {"choices": [{"finish_reason": "length", "message": {"content": "private-user-text"}}],
                    "usage": {"prompt_tokens": 2048, "completion_tokens": 8192}}
        with patch.dict(os.environ, {"LLM_MAX_TOKENS": "8192"}), \
                patch("label_audit.provider.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as send:
            p = OpenAICompatibleProvider(Path(self.tmp.name))
            with self.assertRaises(ProviderError) as raised:
                p.ask("semantic", {})
        message = str(raised.exception)
        for value in ("Агент semantic", "finish_reason=length", "LLM_MAX_TOKENS=8192",
                      "входных токенов: 2048", "выходных токенов: 8192"):
            self.assertIn(value, message)
        self.assertNotIn("рассужден", message)
        self.assertNotIn("private-user-text", message)
        self.assertEqual(send.call_count, 1)
        self.assertEqual(p.calls, 1)
        self.assertEqual(p.output_tokens, 8192)
        self.assertEqual(list(Path(self.tmp.name).glob("*.json")), [])

    def test_ollama_uses_its_supported_token_parameter(self):
        with patch.dict(os.environ, {"LLM_BASE_URL": "http://127.0.0.1:11434/v1"}):
            p = OpenAICompatibleProvider(Path(self.tmp.name))
            self.assertEqual(p.token_parameter, "max_tokens")

    def test_missing_configuration_fails_instead_of_demo_fallback(self):
        with patch.dict(os.environ, {"LLM_MODEL": ""}):
            with self.assertRaises(ProviderError):
                OpenAICompatibleProvider(Path(self.tmp.name))

    def test_openrouter_reasoning_setting_is_sent_and_part_of_cache_key(self):
        response = {"model": "actual-model", "choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}]}
        requests = []

        def respond(req, **kwargs):
            requests.append(json.loads(req.data))
            return io.BytesIO(json.dumps(response).encode())

        with patch.dict(os.environ, {"LLM_BASE_URL": "https://openrouter.ai/api/v1", "LLM_REASONING_EFFORT": "none"}), \
                patch("label_audit.provider.urllib.request.urlopen", side_effect=respond):
            p = OpenAICompatibleProvider(Path(self.tmp.name))
            self.assertEqual(p.ask("instruction", {}), {"ok": True})
            self.assertEqual(requests[0]["reasoning"], {"effort": "none"})
            with patch.dict(os.environ, {"LLM_REASONING_EFFORT": "low"}):
                OpenAICompatibleProvider(Path(self.tmp.name)).ask("instruction", {})
            self.assertEqual(len(requests), 2)
        self.assertTrue(all(json.loads(f.read_text())["actual_model"] == "actual-model"
                            for f in Path(self.tmp.name).glob("*.json")))

    def test_truncated_or_filtered_responses_are_distinguished_and_not_cached(self):
        for finish, message in [("length", "исчерпан лимит"), ("content_filter", "фильтром"), ("tool_calls", "без итогового JSON")]:
            with self.subTest(finish=finish):
                response = {"choices": [{"finish_reason": finish, "message": {"content": '{"ok":true}'}}],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 20}}
                with patch("label_audit.provider.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())):
                    p = OpenAICompatibleProvider(Path(self.tmp.name))
                    with self.assertRaisesRegex(ProviderError, message):
                        p.ask("instruction", {})
                self.assertEqual(p.calls, 1)
                self.assertEqual(p.output_tokens, 20)
                self.assertEqual(list(Path(self.tmp.name).glob("*.json")), [])

    def test_invalid_model_content_has_safe_role_specific_error_and_is_not_cached(self):
        for content, message in [("", "пустой ответ"), (" \n", "пустой ответ"),
                                 (None, "пустой ответ"), ({}, "должно быть строкой"),
                                 ("[]", "требуется объект"),
                                 ('{"reason":"private-user-text",}', "некорректный JSON")]:
            with self.subTest(content=content):
                response = {"choices": [{"finish_reason": "stop", "message": {"content": content}}],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
                with patch("label_audit.provider.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as send:
                    p = OpenAICompatibleProvider(Path(self.tmp.name))
                    with self.assertRaisesRegex(InvalidModelResponse, message) as raised:
                        p.ask("queue", {})
                self.assertIn("Агент queue", str(raised.exception))
                self.assertNotIn("private-user-text", str(raised.exception))
                self.assertNotIn("test-secret", str(raised.exception))
                self.assertEqual(send.call_count, 1)
                self.assertEqual(p.output_tokens, 2)
                self.assertEqual(list(Path(self.tmp.name).glob("*.json")), [])

    def test_malformed_envelope_reports_provider_error_without_crashing(self):
        for response in ([], {}, {"choices": []}, {"choices": [None]},
                         {"choices": [{"message": None}]}, {"choices": [{}], "usage": []}):
            with self.subTest(response=response):
                with patch("label_audit.provider.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())):
                    p = OpenAICompatibleProvider(Path(self.tmp.name))
                    with self.assertRaisesRegex(InvalidModelResponse, "Агент semantic"):
                        p.ask("semantic", {})
                self.assertEqual(list(Path(self.tmp.name).glob("*.json")), [])

    def test_reasoning_setting_rejects_unsupported_provider_or_value(self):
        with patch.dict(os.environ, {"LLM_REASONING_EFFORT": "none"}):
            with self.assertRaisesRegex(ProviderError, "только для OpenRouter"):
                OpenAICompatibleProvider(Path(self.tmp.name))
        with patch.dict(os.environ, {"LLM_BASE_URL": "https://openrouter.ai/api/v1", "LLM_REASONING_EFFORT": "invalid"}):
            with self.assertRaisesRegex(ProviderError, "Некорректный LLM_REASONING_EFFORT"):
                OpenAICompatibleProvider(Path(self.tmp.name))

    def test_daily_quota_is_not_retried_and_raw_error_is_hidden(self):
        error = {"error": {"message": "test-secret", "metadata": {"limit_source": "openrouter_free_tier_daily"}}}
        failure = urllib.error.HTTPError("https://api.example", 429, "Limited", {}, io.BytesIO(json.dumps(error).encode()))
        with patch("label_audit.provider.urllib.request.urlopen", side_effect=failure) as send, \
                patch("label_audit.provider.time.sleep") as sleep:
            p = OpenAICompatibleProvider(Path(self.tmp.name))
            with self.assertRaisesRegex(RateLimitExceeded, "суточный лимит") as raised:
                p.ask("instruction", {})
        self.assertNotIn("test-secret", str(raised.exception))
        self.assertEqual(send.call_count, 1)
        sleep.assert_not_called()
        self.assertEqual(list(Path(self.tmp.name).glob("*.json")), [])

    def test_temporary_rate_limit_respects_retry_after(self):
        failure = urllib.error.HTTPError("https://api.example", 429, "Limited", {"Retry-After": "12"}, io.BytesIO(b'{}'))
        success = {"choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}]}
        with patch("label_audit.provider.urllib.request.urlopen", side_effect=[failure, io.BytesIO(json.dumps(success).encode())]), \
                patch("label_audit.provider.time.sleep") as sleep:
            p = OpenAICompatibleProvider(Path(self.tmp.name))
            self.assertEqual(p.ask("instruction", {}), {"ok": True})
        sleep.assert_called_once_with(12)
        self.assertEqual(p.calls, 2)

    def test_long_rate_limit_does_not_wait_or_retry(self):
        failure = urllib.error.HTTPError("https://api.example", 429, "Limited", {"Retry-After": "3600"}, io.BytesIO(b'not json'))
        with patch("label_audit.provider.urllib.request.urlopen", side_effect=failure) as send, \
                patch("label_audit.provider.time.sleep") as sleep:
            with self.assertRaisesRegex(RateLimitExceeded, "3600 сек"):
                OpenAICompatibleProvider(Path(self.tmp.name)).ask("instruction", {})
        self.assertEqual(send.call_count, 1)
        sleep.assert_not_called()

    def test_pacing_applies_to_requests_but_not_cache_hits(self):
        response = {"choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}]}
        with patch.dict(os.environ, {"LLM_REQUEST_INTERVAL": "3.1"}), \
                patch("label_audit.provider.time.monotonic", return_value=100), \
                patch("label_audit.provider.time.sleep") as sleep, \
                patch("label_audit.provider.urllib.request.urlopen", side_effect=lambda *a, **kw: io.BytesIO(json.dumps(response).encode())):
            p = OpenAICompatibleProvider(Path(self.tmp.name))
            p.ask("instruction", {"version": 1})
            p.ask("instruction", {"version": 1})
            sleep.assert_not_called()
            p.ask("instruction", {"version": 2})
            self.assertAlmostEqual(sleep.call_args.args[0], 3.1)
        self.assertEqual(p.calls, 2)
        self.assertEqual(p.cache_hits, 1)

    def test_retry_after_handles_date_invalid_and_nonfinite_values(self):
        with patch("label_audit.provider.time.time", return_value=0):
            self.assertEqual(retry_delay({"Retry-After": "Thu, 01 Jan 1970 00:00:20 GMT"}), 20)
        for value in ("invalid", "NaN", "inf"):
            self.assertEqual(retry_delay({"Retry-After": value}), 5)


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

    def test_rate_limit_is_reported_as_http_429(self):
        with patch("label_audit.server.run_audit", side_effect=RateLimitExceeded("Исчерпан суточный лимит")):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.request("/api/audit", {"csv": "id,text,label\na,Не могу войти,account\n", "mode": "demo"})
        self.assertEqual(caught.exception.code, 429)
        self.assertIn("суточный лимит", json.load(caught.exception)["error"])


if __name__ == "__main__":
    unittest.main()
