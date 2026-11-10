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
from label_audit.provider import DemoProvider, InvalidModelResponse, ProviderError, RateLimitExceeded
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
        self.assertEqual(result["version"], self.taxonomy["version"])
        self.assertEqual(len(result["sha256"]), 64)

    def test_instruction_model_cannot_override_taxonomy(self):
        p = ScriptedProvider({"instruction": {"issues": [], "questions": [],
                            "classes": [], "rules": [], "version": "invented", "sha256": "invented"}})
        result = InstructionAgent().run(self.taxonomy, p)
        for key in ("classes", "rules", "version", "sha256"):
            self.assertEqual(result[key], self.spec[key])

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

    def test_class_boundaries_reach_all_checking_agents_and_recheck(self):
        text = "Приложение вылетает при входе."
        p = ScriptedProvider({
            "instruction": {"issues": [], "questions": []},
            "semantic": {"labels": ["account"], "confidence": .9, "evidence": text,
                         "reason": "Сбой конкретной операции входа", "needs_neighbors": True},
            "consistency": {"conflict": False, "neighbor_ids": [], "confidence": .5, "reason": "Нет соседей"},
            "arbiter": {"action": "recheck", "proposed_label": None, "confidence": .4, "reason": "Проверка"},
            "queue": {"selected_ids": ["a"], "reason": "Проверка"}})
        run_audit([Record("a", text, "technical")], self.taxonomy, p)
        for role, payload in p.calls:
            if role in {"semantic", "consistency", "arbiter"}:
                with self.subTest(role=role):
                    self.assertEqual(payload["classes"], self.taxonomy["classes"])
                    self.assertEqual(payload["rules"], self.taxonomy["rules"])
        self.assertEqual(sum(role == "semantic" for role, _ in p.calls), 2)

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

    def test_queue_sends_explicit_size_and_accepts_complete_model_selection(self):
        rows = [Record(str(i), "Не могу войти", "delivery") for i in range(4)]
        result = run_audit(rows, self.taxonomy, self.demo)

        def select(payload):
            self.assertEqual(payload["selection_size"], 2)
            self.assertNotIn("0", [x["id"] for x in payload["candidates"]])
            return {"selected_ids": [x["id"] for x in payload["candidates"][:2]], "reason": "Выбор модели"}

        queue = QueueAgent().run(result["findings"], ScriptedProvider({"queue": select}), budget=2, reviewed_ids=["0"])
        self.assertEqual(len(queue["selected_ids"]), 2)
        self.assertEqual(queue["reason"], "Выбор модели")

    def test_queue_rejects_underfilled_answer(self):
        rows = [Record(str(i), "Не могу войти", "delivery") for i in range(3)]
        result = run_audit(rows, self.taxonomy, self.demo)
        p = ScriptedProvider({"queue": {"selected_ids": ["0"], "reason": "Неполный выбор"}})
        queue = QueueAgent().run(result["findings"], p, budget=2)
        self.assertEqual(len(queue["selected_ids"]), 2)
        self.assertIn("не прошёл проверку", queue["reason"])

    def test_queue_size_is_limited_by_available_candidates(self):
        result = run_audit([Record("a", "Не могу войти", "delivery")], self.taxonomy, self.demo)

        def select(payload):
            self.assertEqual(payload["selection_size"], 1)
            return {"selected_ids": ["a"], "reason": "Единственный кандидат"}

        queue = QueueAgent().run(result["findings"], ScriptedProvider({"queue": select}), budget=20)
        self.assertEqual(queue["reason"], "Единственный кандидат")

    def test_queue_invalid_response_preserves_findings_and_discloses_local_ordering(self):
        result = run_audit([Record("a", "Не могу войти", "delivery"),
                            Record("b", "Не прислали чек", "account")], self.taxonomy, self.demo)
        original = copy.deepcopy(result["findings"])

        def fail(payload):
            raise InvalidModelResponse("Агент queue: API вернул пустой ответ (finish_reason=stop).")

        p = ScriptedProvider({"queue": fail})
        queue = QueueAgent().run(result["findings"], p, budget=1, reviewed_ids=["a"])
        self.assertEqual(queue["selected_ids"], ["b"])
        self.assertIn("пустой ответ", queue["reason"])
        self.assertIn("локальная очередь", queue["reason"])
        self.assertEqual(result["findings"], original)

    def test_queue_does_not_hide_transport_or_rate_limit_errors(self):
        result = run_audit([Record("a", "Не могу войти", "delivery")], self.taxonomy, self.demo)
        for error in (ProviderError("API недоступен"), RateLimitExceeded("HTTP 429")):
            with self.subTest(error=type(error).__name__):
                def fail(payload):
                    raise error
                with self.assertRaises(type(error)):
                    QueueAgent().run(result["findings"], ScriptedProvider({"queue": fail}))

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

