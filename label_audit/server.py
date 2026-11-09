from __future__ import annotations

import json
import mimetypes
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .models import load_taxonomy, parse_csv
from .pipeline import apply_feedback, csv_content, run_audit, save_run
from .provider import DemoProvider, OpenAICompatibleProvider


def make_server(root: Path, port=8765):
    taxonomy = load_taxonomy(root / "data/taxonomy.json")
    state = {"run": None, "path": None}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, data, content_type="application/json; charset=utf-8", filename=None):
            body = json.dumps(data, ensure_ascii=False).encode("utf-8") if isinstance(data, (dict, list)) else data
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/api/info":
                self.send(200, {"taxonomy": taxonomy, "team": "edinaya rossiya", "default_mode": "demo"})
            elif path == "/api/state":
                with lock:
                    self.send(200, state["run"] or {})
            elif path in {"/api/export", "/api/reviewed"}:
                with lock:
                    if not state["run"]:
                        self.send(404, {"error": "Сначала запустите проверку."})
                    else:
                        self.send(200, ("\ufeff" + csv_content(state["run"], corrected=path.endswith("reviewed"))).encode("utf-8"), "text/csv; charset=utf-8", "reviewed.csv" if path.endswith("reviewed") else "recommendations.csv")
            elif path == "/api/demo":
                self.send(200, (root / "data/demo.csv").read_bytes(), "text/csv; charset=utf-8")
            elif path in {"/", "/app.js", "/style.css"}:
                name = "index.html" if path == "/" else path[1:]
                mime = {"index.html": "text/html", "app.js": "text/javascript", "style.css": "text/css"}[name]
                self.send(200, (Path(__file__).parent / "web" / name).read_bytes(), mime + "; charset=utf-8")
            else:
                self.send(404, {"error": "Не найдено."})

        def do_POST(self):
            # Local app: only same-origin JSON requests; no cross-origin write endpoint.
            origin = self.headers.get("Origin")
            host = self.headers.get("Host", "")
            allowed_hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if host not in allowed_hosts or (origin and origin not in {"http://" + h for h in allowed_hosts}):
                self.send(403, {"error": "Разрешён только локальный интерфейс."})
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.send(415, {"error": "Ожидается JSON."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 2_000_000:
                    raise ValueError("Допустимый размер запроса: до 2 МБ.")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Ожидается объект JSON.")
                with lock:
                    if self.path == "/api/audit":
                        mode = payload.get("mode", "demo")
                        if mode not in {"demo", "llm"}:
                            raise ValueError("Неизвестный режим.")
                        text = payload.get("csv")
                        if not isinstance(text, str):
                            raise ValueError("Нужен CSV.")
                        records = parse_csv(text, taxonomy)
                        budget = payload.get("budget", 20)
                        if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= 500:
                            raise ValueError("Бюджет должен быть от 1 до 500.")
                        provider = DemoProvider() if mode == "demo" else OpenAICompatibleProvider(root / "runtime/cache")
                        result = run_audit(records, taxonomy, provider, budget)
                        run_dir = root / "runtime/runs" / uuid.uuid4().hex[:12]
                        save_run(result, run_dir)
                        state.update(run=result, path=run_dir)
                    elif self.path == "/api/feedback":
                        if state["run"] is None:
                            raise ValueError("Сначала запустите проверку.")
                        provider = DemoProvider() if state["run"]["mode"] == "demo" else OpenAICompatibleProvider(root / "runtime/cache")
                        result = apply_feedback(state["run"], payload.get("id"), payload.get("label"), provider)
                        save_run(result, state["path"])
                        state["run"] = result
                    else:
                        self.send(404, {"error": "Не найдено."})
                        return
                    self.send(200, state["run"])
            except (ValueError, RuntimeError, OSError, TypeError) as exc:
                self.send(400, {"error": str(exc)})

        def log_message(self, format, *args):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(root, port, open_browser=False):
    server = make_server(root, port)
    address = f"http://127.0.0.1:{server.server_port}"
    print(f"Метка под вопросом: {address}\nCtrl+C — остановить сервер.", flush=True)
    if open_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

