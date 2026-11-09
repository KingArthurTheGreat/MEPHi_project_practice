from __future__ import annotations

import csv
import io
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .agents.arbiter import ArbiterAgent
from .agents.consistency import ConsistencyAgent
from .agents.instruction import InstructionAgent
from .agents.queue import QueueAgent
from .agents.semantic import SemanticAgent
from .models import Record, records_hash, write_json
from .retrieval import NeighborIndex


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def inspect_record(record, spec, provider, index, trace):
    semantic = SemanticAgent().run(record, spec, provider)
    trace.append({"agent": "semantic", "record_id": record.id, "result": semantic})
    consistency = ConsistencyAgent().run(record, semantic, index, spec, provider)
    trace.append({"agent": "consistency", "record_id": record.id, "result": consistency})
    arbiter = ArbiterAgent()
    decision = arbiter.run(record, semantic, consistency, spec, provider)
    trace.append({"agent": "arbiter", "record_id": record.id, "result": decision.copy()})
    if decision["action"] == "recheck":
        semantic = SemanticAgent().run(record, spec, provider, neighbors=consistency["neighbors"])
        trace.append({"agent": "semantic", "record_id": record.id, "recheck": True, "result": semantic})
        decision = arbiter.run(record, semantic, consistency, spec, provider, recheck_allowed=False)
        trace.append({"agent": "arbiter", "record_id": record.id, "recheck": True, "result": decision})
    return {**asdict(record), "semantic": semantic, "consistency": consistency, "decision": decision}


def run_audit(records, taxonomy, provider, budget=20):
    if not records:
        raise ValueError("Нельзя проверить пустой набор.")
    started = time.perf_counter()
    spec = InstructionAgent().run(taxonomy, provider)
    trace = [{"agent": "instruction", "result": spec}]
    index = NeighborIndex(records)
    findings = [inspect_record(r, spec, provider, index, trace) for r in records]
    queue = QueueAgent().run(findings, provider, budget=budget)
    trace.append({"agent": "queue", "result": queue})
    return {"schema_version": 1, "created_at": utc_now(), "mode": provider.mode,
            "data_sha256": records_hash(records), "taxonomy": taxonomy, "spec": spec,
            "input_records": [asdict(r) for r in records], "findings": findings, "queue": queue,
            "feedback": [], "feedback_usage": [], "trace": trace,
            "usage": provider.usage(), "elapsed_seconds": round(time.perf_counter() - started, 6),
            "disclosure": "Автономное демо: эвристики, не LLM." if provider.mode == "demo" else "Живой API; оценки confidence не откалиброваны."}


def csv_content(result, corrected=False):
    stream = io.StringIO(newline="")
    if corrected:
        writer = csv.DictWriter(stream, fieldnames=["id", "text", "label"])
        latest = {x["record_id"]: x["label"] for x in result["feedback"]}
        writer.writeheader()
        for record in result["input_records"]:
            writer.writerow({**record, "label": latest.get(record["id"], record["label"])})
    else:
        writer = csv.DictWriter(stream, fieldnames=["id", "text", "original_label", "action", "proposed_label", "reason", "evidence", "mode"])
        writer.writeheader()
        originals = {x["id"]: x["label"] for x in result["input_records"]}
        for item in result["findings"]:
            writer.writerow({"id": item["id"], "text": item["text"], "original_label": originals[item["id"]],
                             "action": item["decision"]["action"], "proposed_label": item["decision"]["proposed_label"],
                             "reason": item["decision"]["reason"], "evidence": item["semantic"]["evidence"], "mode": result["mode"]})
    return stream.getvalue()


def save_run(result, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "report.json", result)
    (out_dir / "recommendations.csv").write_text(csv_content(result), encoding="utf-8-sig")
    (out_dir / "reviewed.csv").write_text(csv_content(result, corrected=True), encoding="utf-8-sig")
    (out_dir / "trace.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in result["trace"]) + "\n", encoding="utf-8")
    (out_dir / "feedback.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in result["feedback"]), encoding="utf-8")


def apply_feedback(result, record_id, label, provider):
    if provider.mode != result["mode"] or provider.model != result["usage"]["model"]:
        raise ValueError("Повторная проверка должна использовать тот же режим и модель.")
    valid_labels = {x["id"] for x in result["taxonomy"]["classes"]}
    if label not in valid_labels:
        raise ValueError("Неизвестная метка обратной связи.")
    # Copy first: a provider error must not leave a half-applied human decision.
    updated = json.loads(json.dumps(result, ensure_ascii=False))
    by_id = {x["id"]: x for x in updated["findings"]}
    if record_id not in by_id:
        raise ValueError("Неизвестный ID строки.")
    queue_agent = QueueAgent()
    targets = queue_agent.feedback_targets(by_id[record_id], updated["findings"])
    event = {"record_id": record_id, "label": label, "previous_label": by_id[record_id]["label"],
             "at": utc_now(), "source": "human", "recheck_ids": targets}
    updated["feedback"].append(event)
    latest = {x["record_id"]: x["label"] for x in updated["feedback"]}
    records = [Record(r["id"], r["text"], latest.get(r["id"], r["label"])) for r in updated["input_records"]]
    index = NeighborIndex(records)
    for r in records:
        if r.id in targets and r.id not in latest:
            by_id[r.id] = inspect_record(r, updated["spec"], provider, index, updated["trace"])
    by_id[record_id]["label"] = label
    by_id[record_id]["human_label"] = label
    by_id[record_id]["decision"] = {"action": "keep", "proposed_label": label, "confidence": 1.0,
                                      "reason": "Метка подтверждена человеком; это не оценка уверенности модели."}
    updated["findings"] = [by_id[x["id"]] for x in updated["findings"]]
    updated["queue"] = queue_agent.run(updated["findings"], provider, budget=result["queue"]["budget"],
                                        reviewed_ids=list(latest), related_ids=targets)
    updated["trace"].extend([{"agent": "queue", "tool": "feedback", "result": event},
                              {"agent": "queue", "result": updated["queue"]}])
    updated["feedback_usage"].append(provider.usage())
    return updated

