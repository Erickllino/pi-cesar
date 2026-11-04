"""Shared persistence for Vercel/Neon; public prompts remain in Data/*.json."""

import json

from judge import JudgeStore
from review_app import ReviewStore, load_prompts


def connect(database_url):
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("Instale psycopg[binary] para usar DATABASE_URL.") from exc
    return psycopg.connect(database_url, connect_timeout=10)


def initialize_database(database_url, default_categories):
    """Safe to call at startup on multiple instances; does not overwrite existing rows."""
    with connect(database_url) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS categories (
            name TEXT PRIMARY KEY, description TEXT NOT NULL
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS reviews (
            prompt_id TEXT PRIMARY KEY, verdict TEXT NOT NULL,
            categories TEXT NOT NULL, notes TEXT NOT NULL, updated_at TEXT NOT NULL
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS jev_reviews (
            prompt_id TEXT PRIMARY KEY, verdict TEXT NOT NULL,
            categories TEXT NOT NULL, confidence DOUBLE PRECISION NOT NULL,
            model TEXT NOT NULL, prompt_version TEXT NOT NULL, updated_at TEXT NOT NULL,
            probabilities TEXT NOT NULL, category_probabilities TEXT NOT NULL
        )""")
        with db.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO categories (name, description) VALUES (%s, %s) ON CONFLICT (name) DO NOTHING",
                default_categories,
            )


class PostgresReviewStore(ReviewStore):
    def __init__(self, data_dir, database_url):
        self.prompts = load_prompts(data_dir)
        self.prompt_ids = {item["id"] for item in self.prompts}
        self.datasets = sorted({item["dataset"] for item in self.prompts})
        self.database_url = database_url

    def categories(self):
        with connect(self.database_url) as db:
            rows = db.execute("SELECT name, description FROM categories ORDER BY lower(name), name").fetchall()
        return [{"name": name, "description": description} for name, description in rows]

    def add_category(self, name):
        if not isinstance(name, str):
            raise ValueError("Nome de categoria inválido.")
        name = name.strip()
        if not name or len(name) > 80:
            raise ValueError("A categoria deve ter entre 1 e 80 caracteres.")
        with connect(self.database_url) as db:
            row = db.execute(
                "INSERT INTO categories (name, description) VALUES (%s, '') ON CONFLICT (name) DO NOTHING RETURNING name",
                (name,),
            ).fetchone()
        if row is None:
            raise ValueError("Esta categoria já existe.")
        return {"name": name, "description": ""}

    def reviews(self):
        with connect(self.database_url) as db:
            rows = db.execute("SELECT prompt_id, verdict, categories, notes, updated_at FROM reviews").fetchall()
        return {row[0]: {"verdict": row[1], "categories": json.loads(row[2]),
                         "notes": row[3], "updated_at": row[4]} for row in rows}

    def save_review(self, prompt_id, verdict, categories, notes):
        review = self.prepare_review(prompt_id, verdict, categories, notes)
        with connect(self.database_url) as db:
            db.execute("""INSERT INTO reviews (prompt_id, verdict, categories, notes, updated_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (prompt_id) DO UPDATE SET verdict = EXCLUDED.verdict,
                categories = EXCLUDED.categories, notes = EXCLUDED.notes,
                updated_at = EXCLUDED.updated_at""",
                (prompt_id, review["verdict"], json.dumps(review["categories"], ensure_ascii=False),
                 review["notes"], review["updated_at"]),
            )
        return review


class PostgresJudgeStore(JudgeStore):
    """Read historical Jev results; the site cannot invoke the judge."""

    def __init__(self, database_url, prompts):
        self.prompts = prompts
        self.database_url = database_url

    def reviews(self):
        with connect(self.database_url) as db:
            rows = db.execute("""SELECT prompt_id, verdict, categories, confidence, model,
                prompt_version, updated_at, probabilities, category_probabilities FROM jev_reviews""").fetchall()
        return {row[0]: {"verdict": row[1], "categories": json.loads(row[2]), "confidence": row[3],
                         "model": row[4], "prompt_version": row[5], "updated_at": row[6],
                         "probabilities": json.loads(row[7]), "category_probabilities": json.loads(row[8])}
                for row in rows}

    def save(self, prompt_id, result):
        raise NotImplementedError("Resultados históricos do Jev são somente leitura.")
