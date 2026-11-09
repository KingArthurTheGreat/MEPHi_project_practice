from ..models import probability


class ConsistencyAgent:
    """Owner proposal: Подылов Роман. Tool: character TF-IDF neighbor search."""
    name = "consistency"

    def run(self, record, semantic, index, spec, provider):
        neighbors = index.search(record, k=4)
        candidates = [x for x in neighbors if x["similarity"] >= 0.32 and x["label"] != record.label]
        if provider.mode == "llm" and (candidates or semantic["needs_neighbors"]):
            result = provider.ask(self.name, {"record": {"id": record.id, "text": record.text, "label": record.label},
                                 "neighbors": neighbors, "classes": spec["classes"], "rules": spec["rules"]})
        else:
            close = [x for x in candidates if x["similarity"] >= 0.62]
            result = {"conflict": bool(close), "neighbor_ids": [x["id"] for x in close],
                      "confidence": max((x["similarity"] for x in close), default=0.0),
                      "reason": "У близких текстов разные метки; это сигнал для проверки, а не доказательство ошибки." if close else "Значимых противоречий среди найденных соседей нет."}
        allowed = {x["id"] for x in neighbors}
        ids = result.get("neighbor_ids")
        if not isinstance(result.get("conflict"), bool) or not isinstance(ids, list) or not all(isinstance(x, str) and x in allowed for x in ids) or len(ids) != len(set(ids)):
            raise ValueError("Агент согласованности вернул некорректных соседей.")
        if result["conflict"] and not ids:
            raise ValueError("Противоречие должно ссылаться на соседей.")
        result["confidence"] = probability(result.get("confidence"))
        if not isinstance(result.get("reason"), str):
            raise ValueError("Нет обоснования согласованности.")
        return {**result, "neighbors": neighbors}

