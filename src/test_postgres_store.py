import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from judge import JudgeStore
from migrate_sqlite import migrate
from postgres_store import PostgresJudgeStore, PostgresReviewStore, initialize_database
from review_app import ReviewStore, create_stores


class PostgresStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name)
        (self.data / "NotInject_one.json").write_text(json.dumps([{"prompt": "Benign"}]), encoding="utf-8")

    @patch("postgres_store.connect")
    def test_schema_and_defaults(self, connect):
        db = connect.return_value.__enter__.return_value
        initialize_database("unused", [("Jailbreak", "description")])
        statements = [call.args[0] for call in db.execute.call_args_list]
        self.assertTrue(any("CREATE TABLE IF NOT EXISTS jev_reviews" in sql for sql in statements))
        db.cursor.return_value.__enter__.return_value.executemany.assert_called_once()

    @patch("postgres_store.connect")
    def test_review_validation_and_last_write_wins(self, connect):
        store = PostgresReviewStore(self.data, "unused")
        with patch.object(store, "categories", return_value=[{"name": "Jailbreak", "description": ""}]):
            with self.assertRaises(ValueError):
                store.save_review("missing", "malicious", [], "")
            first = store.save_review("NotInject_one.json:0", "malicious", ["Jailbreak"], "first")
            second = store.save_review("NotInject_one.json:0", "non-malicious", ["Jailbreak"], "second")
        self.assertEqual(first["categories"], ["Jailbreak"])
        self.assertEqual(second["categories"], [])
        db = connect.return_value.__enter__.return_value
        self.assertEqual(db.execute.call_count, 2)
        self.assertIn("ON CONFLICT (prompt_id) DO UPDATE", db.execute.call_args.args[0])
        self.assertEqual(db.execute.call_args.args[1][3], "second")

    @patch("postgres_store.connect")
    def test_reads_historical_results(self, connect):
        store = PostgresJudgeStore("unused", [{"id": "NotInject_one.json:0"}])
        connect.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [
            ("NotInject_one.json:0", "non-malicious", "[]", 0.9, "jev-test", "v2", "2026-01-01",
             '{"malicious": 0.1, "non-malicious": 0.9}', "{}"),
        ]
        review = store.reviews()["NotInject_one.json:0"]
        self.assertEqual(review["probabilities"]["non-malicious"], 0.9)
        with self.assertRaises(NotImplementedError):
            store.save("NotInject_one.json:0", {})

    @patch("postgres_store.initialize_database")
    @patch("postgres_store.PostgresJudgeStore")
    @patch("postgres_store.PostgresReviewStore")
    def test_factory_selects_shared_database(self, human, judge, initialize):
        human.return_value.prompts = []
        create_stores(self.data, self.data / "db", "postgresql://unused")
        initialize.assert_called_once()
        human.assert_called_once_with(self.data, "postgresql://unused")
        judge.assert_called_once_with("postgresql://unused", [])
        self.assertFalse((self.data / "db").exists())

    @patch("migrate_sqlite.initialize_database")
    @patch("migrate_sqlite.connect")
    def test_migration_reads_both_local_databases_without_overwriting(self, connect, initialize):
        db_dir = self.data / "db"
        db_dir.mkdir()
        human = ReviewStore(self.data, db_dir)
        human.save_review("NotInject_one.json:0", "non-malicious", [], "original")
        jev = JudgeStore(db_dir, human.prompts)
        jev.save("NotInject_one.json:0", {
            "verdict": "non-malicious", "categories": [], "confidence": 0.9,
            "probabilities": {"malicious": 0.1, "non-malicious": 0.9},
            "category_probabilities": {}, "model": "jev-test",
        })
        cursor = connect.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
        cursor.rowcount = 1
        counts = migrate(db_dir, self.data, "postgresql://unused")
        self.assertEqual(counts["human"], (1, 0))
        self.assertEqual(counts["jev"], (1, 0))
        self.assertEqual(cursor.executemany.call_count, 3)
        queries = [call.args[0] for call in cursor.executemany.call_args_list]
        self.assertTrue(all("ON CONFLICT" in query for query in queries))
        self.assertEqual(len(human.reviews()), 1)
        self.assertEqual(len(jev.reviews()), 1)
        initialize.assert_called_once()

    @patch("migrate_sqlite.initialize_database")
    def test_missing_sqlite_never_initializes_remote(self, initialize):
        with self.assertRaises(FileNotFoundError):
            migrate(self.data / "missing", self.data, "postgresql://unused")
        initialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
