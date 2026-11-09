"""Load single-line .env settings without third-party dependencies."""
import os
import re
from pathlib import Path


def load_env(path: Path) -> None:
    """Keep existing environment values; never execute or expand file contents."""
    if not path.is_file():
        return
    settings = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Некорректная запись .env в строке {number}.")
        if value.startswith(("'", '"')):
            end = value.find(value[0], 1)
            if end < 0 or (value[end + 1:].strip() and not value[end + 1:].lstrip().startswith("#")):
                raise ValueError(f"Некорректные кавычки .env в строке {number}.")
            value = value[1:end]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        settings[key] = value
    for key, value in settings.items():
        os.environ.setdefault(key, value)
