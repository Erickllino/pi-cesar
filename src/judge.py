"""Persistence for historical TypeSafe Jev results used in comparisons."""

import csv
import io
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


PROMPT_VERSION = "piguard-jev-v2-binary"


class JudgeStore:
    def __init__(self, db_dir, prompts):
        self.prompts = prompts
        self.by_id = {item["id"]: item for item in prompts}
        self.db_path = Path(db_dir) / "jev_reviews.sqlite3"
        self.lock = Lock()
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS reviews (prompt_id TEXT PRIMARY KEY, verdict TEXT NOT NULL, categories TEXT NOT NULL, reason TEXT NOT NULL, confidence REAL NOT NULL, model TEXT NOT NULL, prompt_version TEXT NOT NULL, updated_at TEXT NOT NULL)")
            columns = {row[1] for row in db.execute("PRAGMA table_info(reviews)")}
            for column in ("probabilities", "category_probabilities"):
                if column not in columns:
                    db.execute(f"ALTER TABLE reviews ADD COLUMN {column} TEXT NOT NULL DEFAULT '{{}}'")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.db_path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def reviews(self):
        with self.lock, self._connect() as db:
            rows = db.execute("SELECT prompt_id, verdict, categories, confidence, model, prompt_version, updated_at, probabilities, category_probabilities FROM reviews").fetchall()
        return {row[0]: {"verdict": row[1], "categories": json.loads(row[2]), "confidence": row[3],
                         "model": row[4], "prompt_version": row[5], "updated_at": row[6],
                         "probabilities": json.loads(row[7]), "category_probabilities": json.loads(row[8])} for row in rows}

    def save(self, prompt_id, result):
        if prompt_id not in self.by_id:
            raise ValueError("Prompt não encontrado.")
        review = {**result, "prompt_version": PROMPT_VERSION,
                  "updated_at": datetime.now(timezone.utc).isoformat()}
        with self.lock, self._connect() as db:
            db.execute("INSERT INTO reviews (prompt_id, verdict, categories, reason, confidence, model, prompt_version, updated_at, probabilities, category_probabilities) VALUES (?, ?, ?, '', ?, ?, ?, ?, ?, ?) ON CONFLICT(prompt_id) DO UPDATE SET verdict=excluded.verdict, categories=excluded.categories, reason=excluded.reason, confidence=excluded.confidence, model=excluded.model, prompt_version=excluded.prompt_version, updated_at=excluded.updated_at, probabilities=excluded.probabilities, category_probabilities=excluded.category_probabilities",
                       (prompt_id, result["verdict"], json.dumps(result["categories"], ensure_ascii=False), result["confidence"], result["model"], PROMPT_VERSION, review["updated_at"], json.dumps(result["probabilities"]), json.dumps(result["category_probabilities"], ensure_ascii=False)))
        return review

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
        writer.writerow(["id", "dataset", "index", "source", "original_label", "original_category", "prompt", "verdict", "categories", "confidence", "probabilities", "category_probabilities", "model", "prompt_version", "updated_at"])
        reviews = self.reviews()
        for item in self.prompts:
            review = reviews.get(item["id"])
            if review:
                writer.writerow([item["id"], item["dataset"], item["index"], item["source"], item["original_label"], item["original_category"], item["prompt"], review["verdict"], json.dumps(review["categories"], ensure_ascii=False), review["confidence"], json.dumps(review["probabilities"]), json.dumps(review["category_probabilities"], ensure_ascii=False), review["model"], review["prompt_version"], review["updated_at"]])
        return output.getvalue().encode("utf-8-sig")
