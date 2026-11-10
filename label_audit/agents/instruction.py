from ..models import digest, validate_taxonomy


class InstructionAgent:
    """Owner proposal: Ковалев Илья. Tool: taxonomy validation and overlap search."""
    name = "instruction"

    def run(self, taxonomy, provider):
        validate_taxonomy(taxonomy)
        overlaps = []
        for i, left in enumerate(taxonomy["classes"]):
            for right in taxonomy["classes"][i + 1:]:
                common = set(left.get("keywords", [])) & set(right.get("keywords", []))
                if common:
                    overlaps.append(f"{left['id']} / {right['id']}: совпадают признаки {', '.join(sorted(common))}")
        if provider.mode == "llm":
            response = provider.ask(self.name, {"taxonomy": taxonomy, "keyword_overlaps": overlaps})
            if not all(isinstance(response.get(k), list) and all(isinstance(v, str) for v in response[k]) for k in ("issues", "questions")):
                raise ValueError("Агент инструкции нарушил контракт.")
        else:
            response = {"issues": overlaps, "questions": []}
        return {"version": taxonomy["version"], "sha256": digest(taxonomy),
                "classes": taxonomy["classes"], "rules": taxonomy["rules"],
                "issues": response["issues"], "questions": response["questions"]}

