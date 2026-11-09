import copy
import unittest
from pathlib import Path

from label_audit.agents.arbiter import ArbiterAgent
from label_audit.agents.consistency import ConsistencyAgent
from label_audit.agents.instruction import InstructionAgent
from label_audit.agents.queue import QueueAgent
from label_audit.agents.semantic import SemanticAgent
from label_audit.models import Record, load_taxonomy
from label_audit.pipeline import apply_feedback, run_audit
from label_audit.provider import DemoProvider
from label_audit.retrieval import NeighborIndex

ROOT = Path(__file__).resolve().parents[1]


class ScriptedProvider:
    mode = "llm"
    model = "scripted-contract-test"

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def ask(self, role, payload):
        self.calls.append((role, payload))
        value = self.responses[role]
        return copy.deepcopy(value(payload) if callable(value) else value)

    def usage(self):
        return {"model": self.model, "mode": self.mode, "calls": len(self.calls)}


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.taxonomy = load_taxonomy(ROOT / "data/taxonomy.json")
        self.demo = DemoProvider()
        self.spec = InstructionAgent().run(self.taxonomy, self.demo)

    def test_instruction_preserves_definitions_and_version(self):
        original = copy.deepcopy(self.taxonomy)
        result = InstructionAgent().run(self.taxonomy, self.demo)
        self.assertEqual(self.taxonomy, original)
        self.assertEqual(result["version"], "1.0.0-kp2")
        self.assertEqual(len(result["sha256"]), 64)

    def test_instruction_rejects_duplicate_classes(self):
        self.taxonomy["classes"].append(self.taxonomy["classes"][0])
        with self.assertRaises(ValueError):
            InstructionAgent().run(self.taxonomy, self.demo)

    def test_semantic_flags_wrong_label_without_using_it_as_gold(self):
        text = "Не могу войти после смены пароля."
        a = SemanticAgent().run(Record("a", text, "delivery"), self.spec, self.demo)
        b = SemanticAgent().run(Record("a", text, "payment"), self.spec, self.demo)
        self.assertEqual(a, b)
        self.assertEqual(a["labels"], ["account"])

    def test_semantic_abstains_for_multiple_topics(self):
        result = run_audit([Record("a", "Курьер не приехал, с карты списали деньги дважды.", "delivery")], self.taxonomy, self.demo)
        self.assertEqual(result["findings"][0]["decision"]["action"], "review")

    def test_semantic_abstains_for_unknown_topic(self):
        result = run_audit([Record("a", "Есть ли у вас вакансии?", "delivery")], self.taxonomy, self.demo)
        self.assertEqual(result["findings"][0]["decision"]["action"], "review")

    def test_semantic_routes_fabricated_quote_to_human_review(self):
        p = ScriptedProvider({"semantic": {"labels": ["account"], "confidence": .9, "evidence": "invented", "reason": "test", "needs_neighbors": False}})
        result = SemanticAgent().run(Record("a", "Не могу войти", "delivery"), self.spec, p)
        self.assertEqual(result["labels"], [])
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["evidence"], "Не могу войти")
        self.assertTrue(result["needs_neighbors"])

    def test_semantic_recovers_literal_quote_after_formatting_changes(self):
        text = "Не могу войти\nпосле смены пароля — ссылка уже недействительна."
        p = ScriptedProvider({"semantic": {"labels": ["account"], "confidence": .9,
                                          "evidence": "«не могу войти после смены пароля - ссылка уже недействительна.»",
                                          "reason": "test", "needs_neighbors": False}})
        result = SemanticAgent().run(Record("a", text, "delivery"), self.spec, p)
        self.assertEqual(result["evidence"], text)

    def test_semantic_routes_invalid_contract_to_human_review(self):
        p = ScriptedProvider({"semantic": {"labels": ["account"], "confidence": float("nan"), "evidence": "войти", "reason": "test", "needs_neighbors": False}})
        result = SemanticAgent().run(Record("a", "Не могу войти", "delivery"), self.spec, p)
        self.assertEqual(result["labels"], [])
        self.assertEqual(result["confidence"], 0.0)
        self.assertTrue(result["needs_neighbors"])

    def test_semantic_routes_wrong_field_type_to_human_review(self):
        p = ScriptedProvider({"semantic": {"labels": ["account"], "confidence": .9, "evidence": "войти", "reason": "test", "needs_neighbors": "false"}})
        result = SemanticAgent().run(Record("a", "Не могу войти", "delivery"), self.spec, p)
        self.assertEqual(result["labels"], [])
        self.assertEqual(result["confidence"], 0.0)
        self.assertTrue(result["needs_neighbors"])

    def test_consistency_excludes_self_and_finds_contradictory_duplicate(self):
        rows = [Record("a", "Не могу войти после смены пароля.", "account"), Record("b", "Не могу войти после смены пароля.", "delivery")]
        index = NeighborIndex(rows)
        result = ConsistencyAgent().run(rows[0], {"needs_neighbors": False}, index, self.spec, self.demo)
        self.assertTrue(result["conflict"])
        self.assertEqual(result["neighbor_ids"], ["b"])
        self.assertNotIn("a", [n["id"] for n in result["neighbors"]])

    def test_consistency_rejects_unretrieved_neighbor(self):
        rows = [Record("a", "Не могу войти", "account")]
        p = ScriptedProvider({"consistency": {"conflict": True, "neighbor_ids": ["invented"], "confidence": .9, "reason": "x"}})
        with self.assertRaises(ValueError):
            ConsistencyAgent().run(rows[0], {"needs_neighbors": True}, NeighborIndex(rows), self.spec, p)

    def test_arbiter_does_not_force_ambiguous_change(self):
        for action,proposal in [("change","payment"),("keep","delivery")]:
            with self.subTest(action=action):
                p = ScriptedProvider({"arbiter": {"action": action, "proposed_label": proposal, "confidence": .9, "reason": "x"}})
                result = ArbiterAgent().run(Record("a", "Оплата и доставка", "delivery"), {"labels": ["payment", "delivery"],"confidence":.9}, {}, self.spec, p)
                self.assertEqual(result["action"], "review")

    def test_arbiter_limits_recheck_loop(self):
        p = ScriptedProvider({
            "instruction": {"issues": [], "questions": []},
            "semantic": {"labels": ["account"], "confidence": .9, "evidence": "войти", "reason": "x", "needs_neighbors": False},
            "arbiter": {"action": "recheck", "proposed_label": None, "confidence": .4, "reason": "x"},
            "queue": {"selected_ids": ["a"], "reason": "x"}})
        result = run_audit([Record("a", "Не могу войти", "delivery")], self.taxonomy, p)
        self.assertEqual(sum(role == "semantic" for role, _ in p.calls), 2)
        self.assertEqual(sum(role == "arbiter" for role, _ in p.calls), 2)
        self.assertEqual(result["findings"][0]["decision"]["action"], "review")

    def test_queue_respects_budget_and_reviewed_ids(self):
        rows = [Record(str(i), "Не могу войти", "delivery") for i in range(4)]
        result = run_audit(rows, self.taxonomy, self.demo, budget=2)
        queue = QueueAgent().run(result["findings"], self.demo, budget=1, reviewed_ids=["0", "1"])
        self.assertEqual(len(queue["selected_ids"]), 1)
        self.assertTrue(set(queue["selected_ids"]).isdisjoint({"0", "1"}))

    def test_queue_replaces_hallucinated_ids_with_local_selection(self):
        result = run_audit([Record("a", "Не могу войти", "delivery")], self.taxonomy, self.demo)
        p = ScriptedProvider({"queue": {"selected_ids": ["invented"], "reason": "x"}})
        queue = QueueAgent().run(result["findings"], p)
        self.assertEqual(queue["selected_ids"], ["a"])
        self.assertIn("не прошёл проверку", queue["reason"])

    def test_feedback_rechecks_group_without_overwriting_source(self):
        rows = [Record("a", "Не могу войти после смены пароля.", "delivery"), Record("b", "Не могу войти после смены пароля.", "account")]
        result = run_audit(rows, self.taxonomy, self.demo)
        before = copy.deepcopy(result)
        updated = apply_feedback(result, "a", "account", DemoProvider())
        self.assertEqual(result, before)
        self.assertEqual(updated["input_records"][0]["label"], "delivery")
        self.assertIn("b", updated["feedback"][-1]["recheck_ids"])
        self.assertNotIn("a", updated["queue"]["selected_ids"])
        self.assertEqual(next(x for x in updated["findings"] if x["id"] == "b")["decision"]["action"], "keep")


if __name__ == "__main__":
    unittest.main()

