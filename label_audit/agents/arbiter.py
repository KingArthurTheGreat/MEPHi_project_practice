from ..models import probability


class ArbiterAgent:
    """Owner proposal: Хримян Артур. Tool: one bounded semantic recheck."""
    name = "arbiter"

    def run(self, record, semantic, consistency, spec, provider, recheck_allowed=True):
        labels = semantic["labels"]
        if provider.mode == "llm":
            result = provider.ask(self.name, {"text": record.text, "current_label": record.label,
                                 "semantic": semantic, "consistency": consistency,
                                 "classes": spec["classes"], "rules": spec["rules"],
                                 "recheck_allowed": recheck_allowed})
        elif len(labels) != 1 or semantic["confidence"] < 0.65:
            result = {"action": "review", "proposed_label": None, "confidence": semantic["confidence"],
                      "reason": "Несколько тем или недостаточно сведений; выбор метки передан человеку."}
        elif labels[0] != record.label:
            result = {"action": "change", "proposed_label": labels[0], "confidence": semantic["confidence"],
                      "reason": "Инструкция указывает на другой класс. Требуется подтверждение человека."}
        elif consistency["conflict"]:
            result = {"action": "review", "proposed_label": None, "confidence": consistency["confidence"],
                      "reason": "Смысл поддерживает текущую метку, но соседние метки противоречат ей."}
        else:
            result = {"action": "keep", "proposed_label": record.label, "confidence": semantic["confidence"],
                      "reason": "Текущая метка согласуется с найденными признаками."}
        valid = {c["id"] for c in spec["classes"]}
        if result.get("action") not in {"keep", "change", "review", "recheck"} or not isinstance(result.get("reason"), str):
            raise ValueError("Арбитр вернул неизвестное действие.")
        proposal = result.get("proposed_label")
        if proposal is not None and (not isinstance(proposal, str) or proposal not in valid):
            raise ValueError("Арбитр вернул неизвестную метку.")
        result["confidence"] = probability(result.get("confidence"))
        # Programmatic contracts remain authoritative even when an LLM disagrees.
        if result["action"] == "recheck" and not recheck_allowed:
            result.update(action="review", proposed_label=None, reason="Лимит повторной проверки достигнут; нужен человек.")
        if result["action"] in {"keep", "change"} and (len(labels) != 1 or semantic.get("confidence", 0) < 0.65):
            result.update(action="review", proposed_label=None, reason="Семантическая проверка неоднозначна; нужен человек.")
        if result["action"] == "keep" and labels[0] != record.label:
            result.update(action="review", proposed_label=None, reason="Арбитр и семантический агент разошлись; нужен человек.")
        if result["action"] == "change" and (len(labels) != 1 or proposal != labels[0] or proposal == record.label or result["confidence"] < 0.65):
            result.update(action="review", proposed_label=None, reason="Предложение изменить метку не прошло проверку оснований.")
        if result["action"] == "keep" and proposal not in {None, record.label}:
            raise ValueError("keep не может менять метку.")
        if result["action"] == "keep":
            result["proposed_label"] = record.label
        if result["action"] == "review":
            result["proposed_label"] = None
        return result

