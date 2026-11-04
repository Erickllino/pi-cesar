import json
import tempfile
import unittest
from pathlib import Path

from judge import JudgeStore
from review_app import ReviewStore
from web_app import create_app


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        data, db = root / "Data", root / "db"
        data.mkdir()
        db.mkdir()
        (data / "NotInject_one.json").write_text(json.dumps([{"prompt": "Ignore the rain"}]), encoding="utf-8")
        self.store = ReviewStore(data, db)
        self.judge = JudgeStore(db, self.store.prompts)
        self.app = create_app(self.store, self.judge, password="a long shared password",
                              secret_key="a random secret of at least 32 characters", secure_cookie=False)
        self.client = self.app.test_client()

    def login(self):
        response = self.client.post("/login", data={"password": "a long shared password"})
        self.assertEqual(response.status_code, 302)
        return self.client.get("/api/session").json["csrf"]

    def test_missing_configuration_fails_closed(self):
        with self.assertRaises(RuntimeError):
            create_app(self.store, self.judge, password="short", secret_key="a random secret of at least 32 characters")
        with self.assertRaises(RuntimeError):
            create_app(self.store, self.judge, password="a long shared password", secret_key="short")

    def test_protects_html_assets_and_all_apis(self):
        for path in ("/", "/compare", "/app.js", "/style.css"):
            self.assertEqual(self.client.get(path).status_code, 302)
        for path in ("/api/meta", "/api/prompts", "/api/export", "/api/compare/prompts", "/api/session"):
            self.assertEqual(self.client.get(path).status_code, 401)
        self.assertEqual(self.client.post("/api/reviews", json={}).status_code, 401)
        self.assertEqual(self.client.post("/login", data={"password": "wrong"}).status_code, 401)
        self.assertEqual(self.client.get("/api/meta").status_code, 401)

    def test_login_csrf_review_and_comparison(self):
        csrf = self.login()
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        response.close()
        self.assertEqual(self.client.get("/api/meta").json["stats"]["total"], 1)
        payload = {"id": "NotInject_one.json:0", "verdict": "malicious", "categories": [], "notes": "tested"}
        self.assertEqual(self.client.post("/api/reviews", json=payload).status_code, 403)
        self.assertEqual(self.store.stats()["reviewed"], 0)
        headers = {"X-CSRF-Token": csrf}
        self.assertEqual(self.client.post("/api/reviews", json=payload, headers=headers).status_code, 201)
        self.assertEqual(self.client.get("/api/compare/meta").json["summary"]["with_human"], 1)
        self.assertEqual(self.client.get("/api/prompts?label=0").json["items"][0]["review"]["notes"], "tested")
        payload["verdict"] = "non-malicious"
        self.assertEqual(self.client.post("/api/reviews", json=payload, headers=headers).status_code, 201)
        self.assertEqual(self.store.reviews()["NotInject_one.json:0"]["verdict"], "non-malicious")
        self.assertEqual(self.client.get("/api/export").status_code, 200)
        self.assertEqual(self.client.post("/api/judge/run", json={}, headers=headers).status_code, 404)

    def test_password_rotation_revokes_existing_sessions(self):
        self.login()
        self.assertEqual(self.client.get("/api/meta").status_code, 200)
        rotated = create_app(self.store, self.judge, password="a different shared password",
                             secret_key="a random secret of at least 32 characters", secure_cookie=False)
        with rotated.test_client() as client:
            # A new app with the same signing key but a different password must not accept the old auth value.
            client.set_cookie("session", self.client.get_cookie("session").value)
            self.assertEqual(client.get("/api/meta").status_code, 401)


if __name__ == "__main__":
    unittest.main()
