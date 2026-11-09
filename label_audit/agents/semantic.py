import unicodedata

from ..models import probability


_OUTER_QUOTES = "'\"«»“”„‟"
_PUNCTUATION = str.maketrans({"–": "-", "—": "-", "‑": "-", "…": "..."})


def _comparison_text(value):
    """Return comparable text and the source position for each character.

    An LLM can preserve the wording of a quote while changing its typography or
    line wrapping.  Keeping the position map lets us return the literal source
    fragment after such a match instead of trusting the model's rendering.
    """
    chars, positions = [], []
    for position, char in enumerate(value):
        normalized = unicodedata.normalize("NFKC", char).casefold().replace("ё", "е").translate(_PUNCTUATION)
        for normalized_char in normalized:
            if normalized_char.isspace():
                if chars and chars[-1] == " ":
                    continue
                normalized_char = " "
            chars.append(normalized_char)
            positions.append(position)
    while chars and chars[0] == " ":
        chars.pop(0)
        positions.pop(0)
    while chars and chars[-1] == " ":
        chars.pop()
        positions.pop()
    return "".join(chars), positions


def source_quote(text, evidence):
    """Return a literal source quote, or ``None`` when evidence is fabricated."""
    if evidence in text:
        return evidence
    candidate = evidence.strip()
    if len(candidate) >= 2 and candidate[0] in _OUTER_QUOTES and candidate[-1] in _OUTER_QUOTES:
        candidate = candidate[1:-1].strip()
    source, positions = _comparison_text(text)
    quote, _ = _comparison_text(candidate)
    start = source.find(quote) if quote else -1
    if start < 0:
        return None
    end = start + len(quote) - 1
    return text[positions[start]:positions[end] + 1]


def human_review(record, issue):
    """Make an invalid model answer safe without aborting the whole audit."""
    return {"labels": [], "confidence": 0.0, "evidence": record.text,
            "reason": f"{issue}; нужна ручная проверка.", "needs_neighbors": True}


class SemanticAgent:
    """Owner proposal: Беседин Артем. Tool: instruction lookup / neighbor request."""
    name = "semantic"

    def run(self, record, spec, provider, neighbors=None):
        if provider.mode == "llm":
            result = provider.ask(self.name, {"text": record.text, "classes": spec["classes"],
                                  "rules": spec["rules"], "additional_evidence": neighbors or []})
        else:
            normalized = record.text.lower().replace("ё", "е")
            hits = {c["id"]: [word for word in c.get("keywords", []) if word in normalized] for c in spec["classes"]}
            labels = [label for label, words in hits.items() if words]
            result = {"labels": labels, "confidence": 0.82 if len(labels) == 1 else 0.35,
                      "evidence": record.text,
                      "reason": "Признаки: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in hits.items() if v) if labels else "Нет известных признаков; нужен человек.",
                      "needs_neighbors": len(labels) != 1}
        if not isinstance(result, dict):
            return human_review(record, "Модель вернула ответ не в формате объекта")
        valid = {c["id"] for c in spec["classes"]}
        labels = result.get("labels")
        if not isinstance(labels, list) or not all(isinstance(x, str) and x in valid for x in labels) or len(labels) != len(set(labels)):
            return human_review(record, "Модель вернула неизвестные или повторные классы")
        try:
            result["confidence"] = probability(result.get("confidence"))
        except ValueError:
            return human_review(record, "Модель вернула некорректную уверенность")
        if not isinstance(result.get("reason"), str) or not isinstance(result.get("needs_neighbors"), bool):
            return human_review(record, "Модель нарушила контракт ответа")
        evidence = result.get("evidence")
        quoted_source = source_quote(record.text, evidence) if isinstance(evidence, str) and evidence.strip() else None
        if quoted_source is None:
            return human_review(record, "Модель вернула недословную цитату")
        result["evidence"] = quoted_source
        return result

