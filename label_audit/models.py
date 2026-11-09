from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Record:
    id: str
    text: str
    label: str


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_taxonomy(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    validate_taxonomy(data)
    return data


def validate_taxonomy(data: dict) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("version"), str):
        raise ValueError("У инструкции должна быть строковая версия.")
    classes = data.get("classes")
    if not isinstance(classes, list) or not 2 <= len(classes) <= 20:
        raise ValueError("Нужно от 2 до 20 классов.")
    seen = set()
    for c in classes:
        if not isinstance(c, dict) or not all(isinstance(c.get(k), str) and c[k].strip() for k in ("id", "name", "definition")):
            raise ValueError("У класса нужны id, name и definition.")
        if c["id"] in seen:
            raise ValueError("Повтор ID класса.")
        seen.add(c["id"])
        if not isinstance(c.get("keywords", []), list) or not all(isinstance(x, str) and x for x in c.get("keywords", [])):
            raise ValueError("keywords должны быть списком непустых строк.")
    if not isinstance(data.get("rules"), list) or not all(isinstance(x, str) for x in data["rules"]):
        raise ValueError("Нужен список rules.")


def parse_csv(text: str, taxonomy: dict, max_rows: int = 500) -> list[Record]:
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    if not {"id", "text", "label"}.issubset(reader.fieldnames or []):
        raise ValueError("CSV должен содержать столбцы id,text,label.")
    labels = {c["id"] for c in taxonomy["classes"]}
    records, seen = [], set()
    for number, row in enumerate(reader, 2):
        values = [row.get(k) for k in ("id", "text", "label")]
        if not all(isinstance(v, str) and v.strip() for v in values):
            raise ValueError(f"Строка {number}: пустое поле.")
        rid, body, label = (v.strip() for v in values)
        if rid in seen or len(rid) > 100:
            raise ValueError(f"Строка {number}: повтор или слишком длинный ID.")
        if len(body) > 4000:
            raise ValueError(f"Строка {number}: текст длиннее 4000 символов.")
        if label not in labels:
            raise ValueError(f"Строка {number}: неизвестный класс {label!r}.")
        seen.add(rid)
        records.append(Record(rid, body, label))
        if len(records) > max_rows:
            raise ValueError(f"Не более {max_rows} строк на один запуск.")
    if not records:
        raise ValueError("CSV пуст.")
    return records


def probability(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("confidence должен быть конечным числом от 0 до 1.")
    return float(value)


def records_hash(records: list[Record]) -> str:
    return digest([asdict(r) for r in records])

