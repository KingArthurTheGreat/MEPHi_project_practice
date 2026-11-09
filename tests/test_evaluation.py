import csv
import json
import tempfile
import unittest
from pathlib import Path

from label_audit.evaluation import evaluate_project, evaluate_ranking
from label_audit.models import load_taxonomy, parse_csv

ROOT = Path(__file__).resolve().parents[1]
TEMP = ROOT / ".build/test-tmp"
TEMP.mkdir(parents=True, exist_ok=True)


class DataAndMetricTests(unittest.TestCase):
    def test_group_split_has_no_paraphrase_leakage(self):
        gold = json.loads((ROOT / "evaluation/gold.json").read_text(encoding="utf-8"))
        dev = {x["family"] for x in gold if x["split"] == "dev"}
        test = {x["family"] for x in gold if x["split"] == "test"}
        self.assertFalse(dev & test)
        for split, count in [("dev", 72), ("test", 48)]:
            with (ROOT / f"data/{split}.csv").open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, ["id", "text", "label"])
                rows = list(reader)
                self.assertEqual(len(rows), count)
                self.assertEqual({x["id"] for x in rows}, {x["id"] for x in gold if x["split"] == split})

    def test_precision_denominator_is_review_budget_not_error_count(self):
        gold = {"a": {"needs_review": True, "is_label_error": True, "acceptable_labels": ["x"]},
                "b": {"needs_review": True, "is_label_error": False, "acceptable_labels": []},
                "c": {"needs_review": False, "is_label_error": False, "acceptable_labels": ["x"]},
                "d": {"needs_review": False, "is_label_error": False, "acceptable_labels": ["x"]}}
        actions = {"c": {"action": "change", "proposed_label": "y"}}
        result = evaluate_ranking(["a", "c", "b", "d"], gold, actions, k=2)
        self.assertEqual(result["precision_at_k_review"], .5)
        self.assertEqual(result["error_recall_at_k"], 1)
        self.assertEqual(result["false_change_rate_on_clean"], .5)

    def test_empty_error_denominator_is_null(self):
        gold = {"a": {"needs_review": False, "is_label_error": False, "acceptable_labels": ["x"]}}
        self.assertIsNone(evaluate_ranking(["a"], gold, {}, k=20)["error_recall_at_k"])

    def test_ranking_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            evaluate_ranking(["a", "a"], {"a": {}}, {}, 2)

    def test_parser_rejects_unknown_label_duplicate_and_empty(self):
        taxonomy = load_taxonomy(ROOT / "data/taxonomy.json")
        for content in ["id,text,label\na,foo,unknown", "id,text,label\na,foo,account\na,bar,account", "id,text,label", "id,text,label\na,,account"]:
            with self.subTest(content=content), self.assertRaises(ValueError):
                parse_csv(content, taxonomy)

    def test_csv_quotes_bom_newlines_and_extra_gold_column(self):
        taxonomy = load_taxonomy(ROOT / "data/taxonomy.json")
        records = parse_csv('\ufeffid,text,label,gold\na,"Не могу,\nвойти",delivery,account\n', taxonomy)
        self.assertEqual(records[0].text, "Не могу,\nвойти")
        self.assertFalse(hasattr(records[0], "gold"))

    def test_evaluation_deterministic_and_discloses_demo(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as tmp:
            a = evaluate_project(ROOT, Path(tmp) / "a")
            b = evaluate_project(ROOT, Path(tmp) / "b")
            self.assertEqual(a["methods"], b["methods"])
            self.assertEqual(a["mode"], "demo")
            self.assertEqual(a["usage"]["calls"], 0)
            self.assertEqual(a["live_single_agent_baseline"], "not_run")


if __name__ == "__main__":
    unittest.main()

