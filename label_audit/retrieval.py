"""Character TF-IDF retrieval, fitted on input text, never on hidden gold labels."""
from __future__ import annotations

import math
import re
from collections import Counter

from .models import Record


def features(text: str) -> Counter:
    text = re.sub(r"\s+", " ", re.sub(r"[^а-яa-z0-9 ]", " ", text.lower().replace("ё", "е"))).strip()
    return Counter(text[i:i + 3] for i in range(max(0, len(text) - 2)))


class NeighborIndex:
    def __init__(self, records: list[Record]):
        self.records = records
        counts = [features(r.text) for r in records]
        df = Counter(k for c in counts for k in c)
        self.idf = {k: math.log((len(records) + 1) / (v + 1)) + 1 for k, v in df.items()}
        self.vectors = [self.vector(c) for c in counts]

    def vector(self, counter: Counter) -> dict[str, float]:
        v = {k: (1 + math.log(n)) * self.idf.get(k, 1.0) for k, n in counter.items()}
        norm = math.sqrt(sum(n * n for n in v.values())) or 1
        return {k: n / norm for k, n in v.items()}

    def search(self, record: Record, k: int = 4) -> list[dict]:
        query = self.vector(features(record.text))
        result = []
        for r, vector in zip(self.records, self.vectors):
            if r.id == record.id:
                continue
            score = sum(v * vector.get(k, 0) for k, v in query.items())
            result.append({"id": r.id, "text": r.text, "label": r.label, "similarity": round(score, 6)})
        return sorted(result, key=lambda x: (-x["similarity"], x["id"]))[:k]

