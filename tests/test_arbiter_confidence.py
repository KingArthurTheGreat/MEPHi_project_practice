import unittest
from types import SimpleNamespace
from label_audit.agents.arbiter import ArbiterAgent


class ArbiterConfidenceTests(unittest.TestCase):
    def test_invalid_confidence_requires_review(self):
        for value in [None, '0.9', True, -1, 2, float('nan'), float('inf')]:
            with self.subTest(value=value):
                provider = SimpleNamespace(mode='llm', ask=lambda *args: {'action': 'change', 'proposed_label': 'b', 'reason': 'test', 'confidence': value})
                with self.assertRaises(ValueError):
                    ArbiterAgent().run(SimpleNamespace(text='text', label='a'), {'labels': ['b'], 'confidence': .9}, {}, {'classes': [{'id': 'a'}, {'id': 'b'}], 'rules': []}, provider)

    def test_schema_requires_bounded_confidence(self):
        from label_audit.provider import response_schema
        for role in ('semantic', 'consistency', 'arbiter'):
            schema = response_schema(role)
            self.assertIn('confidence', schema['required'])
            self.assertEqual(schema['properties']['confidence'], {'type': 'number', 'minimum': 0, 'maximum': 1})
