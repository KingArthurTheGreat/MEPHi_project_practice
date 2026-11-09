"""Gold is read only here, after the audit. Baselines use observable input labels."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from .models import load_taxonomy, parse_csv, write_json
from .pipeline import run_audit, save_run
from .provider import DemoProvider, OpenAICompatibleProvider


def fraction(numerator, denominator):
    return round(numerator / denominator, 6) if denominator else None


def evaluate_ranking(order, gold, actions, k=20):
    if len(order) != len(set(order)) or set(order) != set(gold):
        raise ValueError("Ранжирование должно содержать каждый ID ровно один раз.")
    k_eff = min(k, len(order))
    top = order[:k_eff]
    reviewable = {rid for rid, g in gold.items() if g["needs_review"]}
    errors = {rid for rid, g in gold.items() if g["is_label_error"]}
    clean = {rid for rid, g in gold.items() if not g["needs_review"]}
    flagged = {rid for rid, action in actions.items() if action["action"] in {"change", "review"}}
    changes = {rid: a["proposed_label"] for rid, a in actions.items() if a["action"] == "change"}
    wrong_changes = sum(rid in clean and label not in gold[rid]["acceptable_labels"] for rid, label in changes.items())
    correct_changes = sum(label in gold[rid]["acceptable_labels"] and gold[rid]["is_label_error"] for rid, label in changes.items())
    return {"n": len(gold), "k": k_eff, "reviewable_total": len(reviewable), "label_errors_total": len(errors),
            "precision_at_k_review": fraction(len(set(top) & reviewable), k_eff),
            "precision_at_k_error": fraction(len(set(top) & errors), k_eff),
            "review_cases_found_at_k": len(set(top) & reviewable), "errors_found_at_k": len(set(top) & errors),
            "error_recall_at_k": fraction(len(set(top) & errors), len(errors)),
            "flagged_error_recall": fraction(len(flagged & errors), len(errors)) if actions else None,
            "false_change_rate_on_clean": fraction(wrong_changes, len(clean)) if actions else None,
            "change_precision": fraction(correct_changes, len(changes)) if actions else None,
            "flagged_count": len(flagged) if actions else None,
            "top_ids": top}


def load_gold(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if len({x["id"] for x in data}) != len(data):
        raise ValueError("Повтор ID в gold.")
    return {x["id"]: x for x in data}


def evaluate_project(root: Path, out_dir: Path, mode="demo", split="test", k=20, max_calls=None):
    taxonomy = load_taxonomy(root / "data/taxonomy.json")
    # The audit input is physically separate and contains no gold, corruption or split fields.
    records = parse_csv((root / f"data/{split}.csv").read_text(encoding="utf-8-sig"), taxonomy)
    provider = DemoProvider() if mode == "demo" else OpenAICompatibleProvider(out_dir / "cache", max_calls=max_calls)
    result = run_audit(records, taxonomy, provider, budget=k)
    save_run(result, out_dir / "audit")
    # Read evaluation targets only once inference has completed.
    gold = load_gold(root / "evaluation/gold.json")
    gold = {r.id: gold[r.id] for r in records}
    ids = sorted(gold)
    findings = result["findings"]
    by_id = {x["id"]: x for x in findings}
    multi_order = result["queue"]["selected_ids"] + result["queue"]["remaining_ids"]
    multi_order += [rid for rid in ids if rid not in multi_order]
    actions = {x["id"]: x["decision"] for x in findings}
    semantic_actions = {}
    for item in findings:
        sem = item["semantic"]
        labels = sem["labels"]
        action = "review" if len(labels) != 1 else ("keep" if labels[0] == item["label"] else "change")
        semantic_actions[item["id"]] = {"action": action, "proposed_label": labels[0] if len(labels) == 1 else None}
    semantic_order = sorted(ids, key=lambda rid: (-{"change": 3, "review": 2, "keep": 0}[semantic_actions[rid]["action"]], rid))
    neighbor_order = sorted(ids, key=lambda rid: (-max((n["similarity"] for n in by_id[rid]["consistency"]["neighbors"] if n["label"] != by_id[rid]["label"]), default=0), rid))
    random_rows = []
    for seed in range(100):
        order = ids.copy()
        random.Random(seed).shuffle(order)
        random_rows.append(evaluate_ranking(order, gold, {}, k))
    random_summary = {key: (round(sum(row[key] for row in random_rows) / len(random_rows), 6) if isinstance(random_rows[0][key], (int, float)) else None) for key in random_rows[0] if key != "top_ids"}
    label = "five_agent_demo" if mode == "demo" else "five_agent_llm"
    single_label = "keyword_only" if mode == "demo" else "semantic_ablation"
    metrics = {"scope": "synthetic_pipeline_check", "mode": mode, "split": split, "seed": 20261105,
               "data_sha256": result["data_sha256"], "taxonomy_sha256": result["spec"]["sha256"],
               "disclosure": "Синтетический набор, созданный с помощью AI; эталоны не проверены независимыми разметчиками. Это проверка прототипа, не оценка качества на реальных данных.",
               "methods": {"random_100_seeds_mean": random_summary,
                           "neighbor_disagreement": evaluate_ranking(neighbor_order, gold, {}, k),
                           single_label: evaluate_ranking(semantic_order, gold, semantic_actions, k),
                           label: evaluate_ranking(multi_order, gold, actions, k)},
               "usage": result["usage"], "elapsed_seconds": result["elapsed_seconds"],
               "live_single_agent_baseline": "not_run" if mode == "demo" else "semantic_ablation reuses semantic outputs; independent baseline needed if rechecks occurred",
               "random_seeds": list(range(100))}
    write_json(out_dir / "metrics.json", metrics)
    return metrics

