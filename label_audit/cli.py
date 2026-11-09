import argparse
import json
from pathlib import Path

from .evaluation import evaluate_project
from .models import load_taxonomy, parse_csv
from .pipeline import apply_feedback, run_audit, save_run
from .provider import DemoProvider, OpenAICompatibleProvider


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Метка под вопросом / edinaya rossiya")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="Проверить CSV")
    audit.add_argument("--input", type=Path, default=root / "data/demo.csv")
    audit.add_argument("--taxonomy", type=Path, default=root / "data/taxonomy.json")
    audit.add_argument("--out", type=Path, default=root / "runtime/latest")
    audit.add_argument("--mode", choices=["demo", "llm"], default="demo")
    audit.add_argument("--budget", type=int, default=20)
    audit.add_argument("--limit", type=int)
    audit.add_argument("--max-calls", type=int)
    evaluate = sub.add_parser("evaluate", help="Воспроизвести метрики")
    evaluate.add_argument("--out", type=Path, default=root / "runtime/evaluation")
    evaluate.add_argument("--mode", choices=["demo", "llm"], default="demo")
    evaluate.add_argument("--split", choices=["dev", "test"], default="test")
    evaluate.add_argument("--k", type=int, default=20)
    evaluate.add_argument("--max-calls", type=int)
    feedback = sub.add_parser("feedback", help="Подтвердить метку и пересмотреть связанную группу")
    feedback.add_argument("--run", type=Path, required=True)
    feedback.add_argument("--id", required=True)
    feedback.add_argument("--label", required=True)
    serve = sub.add_parser("serve", help="Открыть локальный интерфейс")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--open", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "serve":
            from .server import serve
            serve(root, args.port, args.open)
        elif args.command == "audit":
            taxonomy = load_taxonomy(args.taxonomy)
            records = parse_csv(args.input.read_text(encoding="utf-8-sig"), taxonomy)
            if args.limit is not None:
                if args.limit < 1:
                    raise ValueError("limit должен быть положительным.")
                records = records[:args.limit]
            provider = DemoProvider() if args.mode == "demo" else OpenAICompatibleProvider(root / "runtime/cache", args.max_calls)
            result = run_audit(records, taxonomy, provider, args.budget)
            save_run(result, args.out)
            print(json.dumps({"output": str(args.out.resolve()), "rows": len(records), "queued": len(result["queue"]["selected_ids"]), "usage": result["usage"]}, ensure_ascii=False, indent=2))
        elif args.command == "evaluate":
            result = evaluate_project(root, args.out, args.mode, args.split, args.k, args.max_calls)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "feedback":
            result = json.loads((args.run / "report.json").read_text(encoding="utf-8"))
            provider = DemoProvider() if result["mode"] == "demo" else OpenAICompatibleProvider(root / "runtime/cache")
            updated = apply_feedback(result, args.id, args.label, provider)
            save_run(updated, args.run)
            print(json.dumps(updated["feedback"][-1], ensure_ascii=False, indent=2))
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")

