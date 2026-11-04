"""One-time, repeatable import from local SQLite into shared PostgreSQL."""

import argparse
import os
import sqlite3
from contextlib import closing
from pathlib import Path

from postgres_store import connect, initialize_database
from review_app import DEFAULT_CATEGORIES, DEFAULT_DATA, DEFAULT_DB, load_prompts


def source(path):
    if not path.is_file():
        raise FileNotFoundError(f"Banco de origem ausente: {path}")
    db = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def transfer(cursor, rows, sql, valid_ids, batch_size=500):
    inserted = skipped = 0
    batch = []
    for row in rows:
        if row[0] not in valid_ids:
            skipped += 1
            continue
        batch.append(tuple(row))
        if len(batch) == batch_size:
            cursor.executemany(sql, batch)
            inserted += max(0, cursor.rowcount)
            batch.clear()
    if batch:
        cursor.executemany(sql, batch)
        inserted += max(0, cursor.rowcount)
    return inserted, skipped


def migrate(db_dir, data_dir, database_url):
    """Import without overwriting reviews already saved by teammates."""
    valid_ids = {item["id"] for item in load_prompts(data_dir)}
    human_path = Path(db_dir) / "reviews.sqlite3"
    jev_path = Path(db_dir) / "jev_reviews.sqlite3"
    # Fail before writing anything if either source is missing.
    for path in (human_path, jev_path):
        if not path.is_file():
            raise FileNotFoundError(f"Banco de origem ausente: {path}")
    initialize_database(database_url, DEFAULT_CATEGORIES)
    with closing(source(human_path)) as human, closing(source(jev_path)) as jev, connect(database_url) as target:
        with target.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO categories (name, description) VALUES (%s, %s) ON CONFLICT (name) DO NOTHING",
                human.execute("SELECT name, description FROM categories").fetchall(),
            )
            human_rows = human.execute(
                "SELECT prompt_id, verdict, categories, notes, updated_at FROM reviews"
            )
            human_count = transfer(cursor, human_rows, """INSERT INTO reviews
                (prompt_id, verdict, categories, notes, updated_at) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (prompt_id) DO NOTHING""", valid_ids)

            columns = {row[1] for row in jev.execute("PRAGMA table_info(reviews)")}
            probabilities = "probabilities" if "probabilities" in columns else "'{}'"
            category_probabilities = "category_probabilities" if "category_probabilities" in columns else "'{}'"
            jev_rows = jev.execute(f"""SELECT prompt_id, verdict, categories, confidence,
                model, prompt_version, updated_at, {probabilities}, {category_probabilities}
                FROM reviews""")
            jev_count = transfer(cursor, jev_rows, """INSERT INTO jev_reviews
                (prompt_id, verdict, categories, confidence, model, prompt_version,
                 updated_at, probabilities, category_probabilities)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (prompt_id) DO NOTHING""", valid_ids)
    return {"human": human_count, "jev": jev_count, "total_prompts": len(valid_ids)}


def main():
    parser = argparse.ArgumentParser(description="Importar SQLite local para PostgreSQL (sem sobrescrever linhas existentes)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB)
    parser.add_argument("--apply", action="store_true", help="Confirmar a gravação no banco remoto")
    args = parser.parse_args()
    if not args.apply:
        parser.error("Passe --apply para importar; nada foi modificado.")
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        parser.error("Defina DATABASE_URL no ambiente; nunca passe credenciais pela linha de comando.")
    counts = migrate(args.db_dir, args.data_dir, database_url)
    print(f"Importação concluída: {counts['human'][0]} revisões humanas e {counts['jev'][0]} resultados Jev inseridos; "
          f"{counts['human'][1] + counts['jev'][1]} IDs ausentes nos JSONs ignorados; "
          f"{counts['total_prompts']} prompts disponíveis. Linhas já existentes foram preservadas.")


if __name__ == "__main__":
    main()
