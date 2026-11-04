"""Prompt-injection review UI. Run locally: python3 src/review_app.py"""

import argparse
import csv
import io
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlparse

from comparison import FILTERS, filter_prompts, summary
from judge import JudgeStore


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "Data"
DEFAULT_DB = ROOT / "db"
STATIC = Path(__file__).resolve().parent / "review_static"
DEFAULT_CATEGORIES = [
    ("Sobrescrita de instruções", "Tenta substituir instruções ou prioridades existentes."),
    ("Falsificação de papel/autoridade", "Finge ser sistema, desenvolvedor ou outra autoridade."),
    ("Falsificação de contexto", "Insere mensagens, conversas ou documentos falsos."),
    ("Exfiltração de dados", "Pede vazamento de informações privadas ou do contexto."),
    ("Abuso de ferramentas", "Induz chamadas de ferramentas ou ações não autorizadas."),
    ("Desvio de resposta", "Muda o formato, conteúdo ou objetivo da resposta."),
    ("Codificação/ofuscação", "Esconde instruções em codificações ou texto ofuscado."),
    ("Jailbreak", "Tenta contornar restrições de segurança."),
    ("Engenharia social", "Usa pressão, persuasão ou pretextos para obter obediência."),
    ("Injeção multilíngue", "Usa mudança de idioma como parte do ataque."),
    ("Outro", "Técnica não coberta pelas demais categorias."),
]


def load_prompts(data_dir):
    """Flatten supported datasets without modifying the source files."""
    prompts = []
    for path in sorted(Path(data_dir).glob("*.json")):
        name = path.stem.casefold()
        inferred_label = 1 if name.startswith("bipia_") else 0 if name.startswith("notinject_") else None
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        if isinstance(data, list):
            for index, entry in enumerate(data):
                if isinstance(entry, dict) and isinstance(entry.get("prompt"), str):
                    prompts.append({
                        "id": f"{path.name}:{index}", "prompt": entry["prompt"],
                        "dataset": path.name, "index": index,
                        "source": entry.get("source"), "original_label": entry.get("label", inferred_label),
                        "original_category": entry.get("category"),
                        "word_list": entry.get("word_list"),
                    })
        elif isinstance(data, dict):
            for group, entries in data.items():
                if not isinstance(entries, list):
                    continue
                for index, prompt in enumerate(entries):
                    if isinstance(prompt, str):
                        prompts.append({
                            "id": f"{path.name}:{group}:{index}", "prompt": prompt,
                            "dataset": path.name, "index": index, "source": None,
                            "original_label": inferred_label, "original_category": group,
                            "word_list": None,
                        })
    return prompts


class ReviewStore:
    def __init__(self, data_dir, db_dir=None):
        self.prompts = load_prompts(data_dir)
        self.prompt_ids = {item["id"] for item in self.prompts}
        self.datasets = sorted({item["dataset"] for item in self.prompts})
        self.db_path = Path(db_dir if db_dir is not None else DEFAULT_DB) / "reviews.sqlite3"
        self.lock = Lock()
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS reviews (prompt_id TEXT PRIMARY KEY, verdict TEXT NOT NULL, categories TEXT NOT NULL, notes TEXT NOT NULL, updated_at TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS categories (name TEXT PRIMARY KEY, description TEXT NOT NULL)")
            db.executemany("INSERT OR IGNORE INTO categories VALUES (?, ?)", DEFAULT_CATEGORIES)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.db_path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def categories(self):
        with self.lock, self._connect() as db:
            return [{"name": name, "description": description} for name, description in db.execute("SELECT name, description FROM categories ORDER BY name COLLATE NOCASE")]

    def add_category(self, name):
        if not isinstance(name, str):
            raise ValueError("Nome de categoria inválido.")
        name = name.strip()
        if not name or len(name) > 80:
            raise ValueError("A categoria deve ter entre 1 e 80 caracteres.")
        with self.lock, self._connect() as db:
            try:
                db.execute("INSERT INTO categories VALUES (?, '')", (name,))
            except sqlite3.IntegrityError as exc:
                raise ValueError("Esta categoria já existe.") from exc
        return {"name": name, "description": ""}

    def reviews(self):
        with self.lock, self._connect() as db:
            rows = db.execute("SELECT prompt_id, verdict, categories, notes, updated_at FROM reviews").fetchall()
        return {row[0]: {"verdict": row[1], "categories": json.loads(row[2]), "notes": row[3], "updated_at": row[4]} for row in rows}

    def prepare_review(self, prompt_id, verdict, categories, notes):
        if prompt_id not in self.prompt_ids:
            raise ValueError("Prompt não encontrado.")
        if verdict not in ("malicious", "non-malicious", "uncertain"):
            raise ValueError("Selecione um veredito válido.")
        if not isinstance(categories, list) or not all(isinstance(c, str) for c in categories):
            raise ValueError("Categorias inválidas.")
        if not isinstance(notes, str) or len(notes) > 10000:
            raise ValueError("Notas inválidas (máximo 10.000 caracteres).")
        allowed = {item["name"] for item in self.categories()}
        if not set(categories) <= allowed:
            raise ValueError("Uma ou mais categorias não existem.")
        if verdict == "non-malicious":
            categories = []
        review = {"verdict": verdict, "categories": list(dict.fromkeys(categories)), "notes": notes,
                  "updated_at": datetime.now(timezone.utc).isoformat()}
        return review

    def save_review(self, prompt_id, verdict, categories, notes):
        review = self.prepare_review(prompt_id, verdict, categories, notes)
        with self.lock, self._connect() as db:
            db.execute("INSERT INTO reviews VALUES (?, ?, ?, ?, ?) ON CONFLICT(prompt_id) DO UPDATE SET verdict=excluded.verdict, categories=excluded.categories, notes=excluded.notes, updated_at=excluded.updated_at",
                       (prompt_id, review["verdict"], json.dumps(review["categories"], ensure_ascii=False), review["notes"], review["updated_at"]))
        return review

    def filtered(self, dataset="", status="", label="", search="", reviews=None):
        if reviews is None:
            reviews = self.reviews()
        needle = search.casefold().strip()
        items = []
        for item in self.prompts:
            review = reviews.get(item["id"])
            if dataset and item["dataset"] != dataset:
                continue
            if status == "pending" and review or status in ("malicious", "non-malicious", "uncertain") and (not review or review["verdict"] != status):
                continue
            if status == "reviewed" and not review:
                continue
            if label and str(item["original_label"]) != label:
                continue
            if needle and needle not in item["prompt"].casefold() and needle not in str(item["source"] or "").casefold():
                continue
            items.append({**item, "review": review})
        return items

    def stats(self):
        reviews = self.reviews()
        counts = {"total": len(self.prompts), "reviewed": len(reviews), "pending": len(self.prompts) - len(reviews),
                  "malicious": 0, "non-malicious": 0, "uncertain": 0}
        for review in reviews.values():
            counts[review["verdict"]] += 1
        return counts

    def export_csv(self):
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "dataset", "index", "source", "original_label", "original_category", "prompt", "verdict", "categories", "notes", "updated_at"])
        reviews = self.reviews()
        for item in self.prompts:
            review = reviews.get(item["id"])
            if review:
                writer.writerow([item["id"], item["dataset"], item["index"], item["source"], item["original_label"], item["original_category"], item["prompt"], review["verdict"], json.dumps(review["categories"], ensure_ascii=False), review["notes"], review["updated_at"]])
        return output.getvalue().encode("utf-8-sig")


class Handler(BaseHTTPRequestHandler):
    store = None
    judge_store = None

    def send_bytes(self, content, content_type="application/json; charset=utf-8", status=200, filename=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, value, status=200):
        self.send_bytes(json.dumps(value, ensure_ascii=False).encode("utf-8"), status=status)

    def do_GET(self):
        url = urlparse(self.path)
        args = parse_qs(url.query)
        if url.path == "/api/compare/prompts":
            try:
                offset = max(0, int(args.get("offset", ["0"])[0]))
                limit = min(50, max(1, int(args.get("limit", ["12"])[0])))
            except ValueError:
                return self.send_json({"error": "Paginação inválida."}, 400)
            dataset, status, label, search = (args.get(key, [""])[0] for key in ("dataset", "status", "label", "search"))
            if dataset not in ("", *self.store.datasets) or status not in FILTERS or label not in ("", "0", "1"):
                return self.send_json({"error": "Filtro inválido."}, 400)
            return self.send_json(filter_prompts(self.store.prompts, self.judge_store.reviews(), self.store.reviews(),
                                                 dataset, status, label, search, offset, limit))
        if url.path == "/api/compare/meta":
            return self.send_json({"datasets": self.store.datasets,
                                   "summary": summary(self.store.prompts, self.judge_store.reviews(), self.store.reviews())})
        if url.path == "/api/prompts":
            try:
                offset = max(0, int(args.get("offset", ["0"])[0]))
                limit = min(50, max(1, int(args.get("limit", ["12"])[0])))
            except ValueError:
                return self.send_json({"error": "Paginação inválida."}, 400)
            items = self.store.filtered(*(args.get(key, [""])[0] for key in ("dataset", "status", "label", "search")))
            return self.send_json({"total": len(items), "items": items[offset:offset + limit]})
        if url.path == "/api/meta":
            return self.send_json({"datasets": self.store.datasets, "categories": self.store.categories(), "stats": self.store.stats()})
        if url.path == "/api/export":
            return self.send_bytes(self.store.export_csv(), "text/csv; charset=utf-8", filename="revisoes_prompts.csv")
        files = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/compare": ("compare.html", "text/html; charset=utf-8"),
                 "/compare.js": ("compare.js", "application/javascript; charset=utf-8"),
                 "/compare.css": ("compare.css", "text/css; charset=utf-8"),
                 "/app.js": ("app.js", "application/javascript; charset=utf-8"),
                 "/style.css": ("style.css", "text/css; charset=utf-8")}
        if url.path in files:
            filename, mime = files[url.path]
            return self.send_bytes((STATIC / filename).read_bytes(), mime)
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/api/reviews", "/api/categories"):
            return self.send_json({"error": "Rota não encontrada."}, 404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 50000:
                raise ValueError("Tamanho da requisição inválido.")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Envie um objeto JSON.")
            if path == "/api/categories":
                result = self.store.add_category(body.get("name", ""))
            else:
                result = self.store.save_review(body.get("id"), body.get("verdict"), body.get("categories", []), body.get("notes", ""))
            self.send_json(result, 201)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, 400)


def create_stores(data_dir, db_dir, database_url=None):
    """Use shared PostgreSQL when configured; otherwise keep local SQLite."""
    if database_url:
        from postgres_store import PostgresJudgeStore, PostgresReviewStore, initialize_database

        initialize_database(database_url, DEFAULT_CATEGORIES)
        store = PostgresReviewStore(data_dir, database_url)
        return store, PostgresJudgeStore(database_url, store.prompts)
    Path(db_dir).mkdir(parents=True, exist_ok=True)
    store = ReviewStore(data_dir, db_dir)
    return store, JudgeStore(db_dir, store.prompts)


def main():
    parser = argparse.ArgumentParser(description="Interface local de revisão de prompts")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB)
    parser.add_argument("--host", default="127.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not args.data_dir.is_dir():
        parser.error(f"Pasta não encontrada: {args.data_dir}")
    Handler.store, Handler.judge_store = create_stores(args.data_dir, args.db_dir, os.environ.get("DATABASE_URL"))
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Revisando {len(Handler.store.prompts)} prompts em http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
