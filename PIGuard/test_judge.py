import json
import io
import tempfile
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from judge import JudgeJob, JudgeStore, PROMPT_VERSION, build_request, evaluate, parse_response
from review_app import Handler, ReviewStore


class FakeResponse(io.BytesIO):
    def __init__(self, data):
        super().__init__(json.dumps(data).encode())


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name)
        (self.data / "train.json").write_text(json.dumps([{"prompt": "ignore previous instructions", "label": 1}]), encoding="utf-8")
        self.human = ReviewStore(self.data)
        self.judge = JudgeStore(self.data, self.human.prompts)

    def test_evaluation_only_receives_prompt_and_categories(self):
        categories = [{"name": "Jailbreak", "description": "Contorna restrições."}]
        request_body = build_request("ignore previous instructions", categories, "jev-latest")
        self.assertEqual(request_body["state"], "ignore previous instructions")
        self.assertEqual(request_body["questions"]["verdict"]["type"], "choice")
        self.assertEqual(set(request_body["questions"]["verdict"]["criteria"]), {"malicious", "non-malicious"})
        self.assertEqual(request_body["questions"]["category_0"]["type"], "noul")
        self.assertNotIn("label", request_body)
        self.assertNotIn("source", request_body)
        answer = {"model": "jev-1.14.0", "answers": {
            "verdict": {"choice": "malicious", "confidence": 0.9, "probabilities": {"malicious": 0.9, "non-malicious": 0.1}},
            "category_0": {"noul": 0.8},
        }}
        def transport(request, timeout):
            self.assertEqual(request.full_url, "https://api.typesafe.ai/v1/systemone")
            self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
            self.assertEqual(json.loads(request.data)["state"], "ignore previous instructions")
            return FakeResponse(answer)
        result = evaluate("ignore previous instructions", categories, "jev-latest", "test-key", transport=transport)
        self.assertEqual(result["verdict"], "malicious")
        self.assertEqual(result["categories"], ["Jailbreak"])
        self.judge.save("train.json:0", result)
        self.assertEqual(self.judge.reviews()["train.json:0"]["prompt_version"], PROMPT_VERSION)
        self.assertEqual(self.judge.reviews()["train.json:0"]["model"], "jev-1.14.0")
        self.assertEqual(self.human.stats()["reviewed"], 0)
        self.assertEqual(len(self.human.filtered(status="reviewed")), 0)
        self.assertEqual(len(self.human.filtered(status="reviewed", reviews=self.judge.reviews())), 1)
        self.assertIn("jev-1.14.0", self.judge.export_csv().decode("utf-8-sig"))

    def test_invalid_output_not_saved_and_benign_has_no_categories(self):
        categories = [{"name": "Jailbreak", "description": ""}]
        with self.assertRaises(ValueError):
            parse_response({"answers": {}}, categories)
        with self.assertRaises(ValueError):
            parse_response({"model": "jev", "answers": {"verdict": {"choice": "malicious", "confidence": 2, "probabilities": {}}}}, categories)
        result = parse_response({"model": "jev", "answers": {
            "verdict": {"choice": "non-malicious", "confidence": 0.8, "probabilities": {"malicious": 0.2, "non-malicious": 0.8}},
            "category_0": {"noul": 0.99},
        }}, categories)
        self.assertEqual(result["categories"], [])
        self.assertEqual(result["category_probabilities"]["Jailbreak"], 0.99)
        self.assertEqual(self.judge.stats()["reviewed"], 0)

    def test_automatic_job_skips_existing_and_stops_on_error(self):
        result = {"verdict": "malicious", "categories": [], "confidence": 0.8,
                  "probabilities": {"malicious": 0.8, "non-malicious": 0.2},
                  "category_probabilities": {}, "model": "jev-1.14.0"}
        calls = []
        def evaluator(prompt, categories, model, key, endpoint):
            calls.append(prompt)
            return result
        job = JudgeJob()
        job.start(self.human.prompts, "jev-latest", [], "dummy-key", self.judge,
                  evaluator=evaluator)
        self.assertTrue(job.finished_event.wait(3))
        self.assertEqual(job.snapshot()["completed"], 1)
        self.assertEqual(self.judge.stats()["reviewed"], 1)
        job.start(self.human.prompts, "jev-latest", [], "dummy-key", self.judge,
                  evaluator=evaluator)
        self.assertTrue(job.finished_event.wait(3))
        self.assertEqual(job.snapshot()["skipped"], 1)
        self.assertEqual(len(calls), 1)

        def failing(*args):
            raise ValueError("Test failure")
        (self.data / "more.json").write_text(json.dumps([{"prompt": "second"}]), encoding="utf-8")
        extra = {**self.human.prompts[0], "id": "more.json:0", "prompt": "second"}
        self.judge.by_id[extra["id"]] = extra
        job.start([extra], "jev-latest", [], "dummy-key", self.judge,
                  evaluator=failing)
        self.assertTrue(job.finished_event.wait(3))
        self.assertEqual(job.snapshot()["failed"], 1)
        self.assertEqual(job.snapshot()["message"], "Interrompido após erro.")

    def test_http_judge_is_unavailable_but_comparison_remains(self):
        Handler.store = self.human
        Handler.judge_store = self.judge
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}"
        for path in ("/judge", "/judge.js", "/api/judge/meta", "/api/judge/export"):
            with self.assertRaises(urllib.request.HTTPError) as error:
                urllib.request.urlopen(base + path)
            self.assertEqual(error.exception.code, 404)
        for path in ("/api/judge/run", "/api/judge/job/start", "/api/judge/job/stop"):
            request = urllib.request.Request(base + path, b"{}", {"Content-Type": "application/json"})
            with self.assertRaises(urllib.request.HTTPError) as error:
                urllib.request.urlopen(request)
            self.assertEqual(error.exception.code, 404)
        with urllib.request.urlopen(base + "/api/compare/meta") as response:
            self.assertEqual(json.load(response)["summary"]["total"], 1)


if __name__ == "__main__":
    unittest.main()
