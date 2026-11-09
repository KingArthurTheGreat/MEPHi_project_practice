class QueueAgent:
    """Owner proposal: Курилов Максим. Tool: budgeted selection and feedback routing."""
    name = "queue"

    @staticmethod
    def score(finding):
        decision = finding["decision"]
        base = {"change": 3.0, "review": 2.0, "keep": 0.0}[decision["action"]]
        return base + 0.4 * float(finding["consistency"]["conflict"]) + 0.3 * decision["confidence"]

    def run(self, findings, provider, budget=20, reviewed_ids=None, related_ids=None):
        if not isinstance(budget, int) or not 1 <= budget <= 500:
            raise ValueError("Бюджет проверки должен быть от 1 до 500.")
        reviewed = set(reviewed_ids or [])
        related = set(related_ids or [])
        candidates = [x for x in findings if x["id"] not in reviewed and (x["decision"]["action"] != "keep" or x["id"] in related)]
        ranked = sorted(candidates, key=lambda x: (-(self.score(x) + (1 if x["id"] in related else 0)), x["id"]))

        def local_selection():
            # Greedy coverage: postpone an almost duplicate item until other groups are seen.
            selected, deferred, covered = [], [], set()
            for item in ranked:
                (deferred if item["id"] in covered else selected).append(item["id"])
                covered.update(n["id"] for n in item["consistency"]["neighbors"] if n["similarity"] >= 0.9)
            return (selected + deferred)[:budget]

        if provider.mode == "llm" and ranked:
            compact = [{"id": x["id"], "text": x["text"], "decision": x["decision"],
                        "conflict_ids": x["consistency"]["neighbor_ids"]} for x in ranked[:80]]
            result = provider.ask(self.name, {"candidates": compact, "budget": budget,
                                  "reviewed_ids": sorted(reviewed), "related_ids": sorted(related)})
            ids = result.get("selected_ids") if isinstance(result, dict) else None
            allowed = {x["id"] for x in compact}
            valid_answer = (isinstance(ids, list) and ids and len(ids) <= budget
                            and all(isinstance(x, str) and x in allowed for x in ids)
                            and len(ids) == len(set(ids)) and isinstance(result.get("reason"), str))
            if valid_answer:
                reason = result["reason"]
            else:
                ids = local_selection()
                reason = "Ответ агента очереди не прошёл проверку; применена локальная очередь из допустимых строк."
        else:
            ids = local_selection()
            reason = "Сначала вероятные ошибки и затронутые решением группы; почти дубликаты отложены ради покрытия."
        return {"selected_ids": ids, "remaining_ids": [x["id"] for x in ranked if x["id"] not in ids],
                "budget": budget, "candidate_count": len(candidates), "reason": reason,
                "related_ids": sorted(related - reviewed)}

    def feedback_targets(self, finding, findings):
        # This routes a fresh review; it never silently propagates a new label to neighbors.
        direct = {x["id"] for x in finding["consistency"]["neighbors"] if x["similarity"] >= 0.62}
        reverse = {x["id"] for x in findings if any(n["id"] == finding["id"] and n["similarity"] >= 0.62 for n in x["consistency"]["neighbors"])}
        return sorted((direct | reverse) - {finding["id"]})
