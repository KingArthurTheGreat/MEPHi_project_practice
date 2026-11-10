import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from label_audit.provider import OpenAICompatibleProvider, ProviderError


class GigaChatTests(unittest.TestCase):
    def test_oauth_and_chat_and_token_reuse(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            'LLM_BASE_URL': 'https://api.giga.chat/v1', 'LLM_MODEL': 'GigaChat-2',
            'GIGACHAT_AUTH_KEY': 'secret', 'LLM_REASONING_EFFORT': '',
            'LLM_TOKEN_PARAMETER': 'max_tokens', 'LLM_REQUEST_INTERVAL': '0',
        }):
            requests = []
            def respond(req, **kwargs):
                requests.append(req)
                if req.full_url.endswith('/oauth'):
                    return io.BytesIO(json.dumps({'access_token': 'token', 'expires_at': (time.time()+1800)*1000}).encode())
                return io.BytesIO(json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}], 'usage': {'prompt_tokens': 2, 'completion_tokens': 1}}).encode())
            with patch('urllib.request.urlopen', side_effect=respond):
                p = OpenAICompatibleProvider(Path(tmp))
                p.ask('instruction', {'a': 1})
                p.ask('instruction', {'a': 2})
            self.assertEqual(len(requests), 3)
            self.assertEqual(requests[0].get_header('Authorization'), 'Basic secret')
            self.assertIn(b'scope=GIGACHAT_API_PERS', requests[0].data)
            self.assertEqual(requests[1].get_header('Authorization'), 'Bearer token')
            self.assertEqual(json.loads(requests[1].data)['response_format']['type'], 'json_schema')
            self.assertNotIn('secret', ''.join(p.read_text() for p in Path(tmp).glob('*.json')))

    def test_missing_key(self):
        with patch.dict(os.environ, {'LLM_BASE_URL': 'https://api.giga.chat/v1', 'LLM_MODEL': 'GigaChat-2', 'GIGACHAT_AUTH_KEY': ''}):
            with self.assertRaisesRegex(ProviderError, 'GIGACHAT_AUTH_KEY'):
                OpenAICompatibleProvider(Path('.'))
