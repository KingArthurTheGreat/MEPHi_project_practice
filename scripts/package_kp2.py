"""Build a source-and-deliverables archive using an explicit allowlist."""
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "output/KP2_edinaya_rossiya.zip"
FILES = ["README.md", "pyproject.toml", ".gitignore", ".gitattributes", ".env.example", "START_DEMO.bat", "REPRODUCE.bat"]
DIRECTORIES = ["label_audit", "data", "evaluation", "tests", "docs", "scripts", "results", ".github"]
OUTPUTS = ["output/KP2_report.pdf", "output/KP2_presentation.pptx", "output/KP2_presentation_updated.pptx", "output/KP2_presentation.pdf", "output/START_HERE.txt"]


def package():
    selected = [ROOT / name for name in FILES + OUTPUTS]
    for name in DIRECTORIES:
        selected.extend(p for p in (ROOT / name).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"})
    selected.extend((ROOT / "output/screenshots").glob("*.png"))
    selected = sorted(set(selected))
    manifest = []
    for path in selected:
        if not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError(f"Missing or unsafe file: {path}")
        manifest.append({"file": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
    with ZipFile(ARCHIVE, "w", compression=ZIP_DEFLATED, compresslevel=9) as z:
        for path in selected:
            z.write(path, "metka-pod-voprosom/" + path.relative_to(ROOT).as_posix())
        z.writestr("metka-pod-voprosom/MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    with ZipFile(ARCHIVE) as z:
        if z.testzip() is not None:
            raise RuntimeError("ZIP integrity failed")
    print(json.dumps({"archive": str(ARCHIVE), "files": len(manifest), "bytes": ARCHIVE.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    package()
