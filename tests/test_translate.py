"""Checks for deterministic translation-pipeline safeguards without API calls."""

import importlib.util
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import httpx


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "translate.py"
SPEC = importlib.util.spec_from_file_location("translate", SCRIPT_PATH)
TRANSLATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = TRANSLATOR
SPEC.loader.exec_module(TRANSLATOR)


class TranslationStructureTests(unittest.TestCase):
    def test_spanish_fewshot_uses_piarena_examples(self) -> None:
        fewshot = TRANSLATOR.build_fewshot("es")
        messages = "\n".join(message["content"] for message in fewshot)

        self.assertIn("https://arlington-schools-update.info", messages)
        self.assertIn("current subscription term has expired", messages)
        self.assertNotIn("PWNED", messages)

    def test_lcc_long_literal_fields_are_not_translated(self) -> None:
        self.assertFalse(TRANSLATOR.should_translate("lcc_long", "context"))
        self.assertFalse(TRANSLATOR.should_translate("lcc_long", "target_task_answer"))
        self.assertTrue(TRANSLATOR.should_translate("lcc_long", "target_inst"))
        self.assertTrue(TRANSLATOR.should_translate("squad_v2", "context"))
        self.assertTrue(
            TRANSLATOR.should_translate("hotpotqa_rag_knowledge_corruption", "injected_task_answer")
        )

    def test_all_source_datasets_are_supported(self) -> None:
        self.assertIn("hotpotqa_rag_knowledge_corruption", TRANSLATOR.SUPPORTED_DATASETS)
        self.assertIn("msmarco_rag_knowledge_corruption", TRANSLATOR.SUPPORTED_DATASETS)
        self.assertIn("nq_rag_knowledge_corruption", TRANSLATOR.SUPPORTED_DATASETS)

    def test_url_change_is_rejected(self) -> None:
        with self.assertRaises(TRANSLATOR.UrlStructureError):
            TRANSLATOR.validate_structure(
                "Visit https://example.com",
                "Visite https://different.example",
            )

    def test_url_marker_round_trip(self) -> None:
        source = "Read https://one.example/a and https://two.example/b."
        masked, urls = TRANSLATOR.mask_urls(source)
        restored = TRANSLATOR.restore_markers(
            masked,
            TRANSLATOR.URL_MARKER_PATTERN,
            "URL",
            urls,
            TRANSLATOR.UrlStructureError,
        )
        self.assertEqual(restored, source)

    def test_many_url_markers_are_preserved_in_one_text(self) -> None:
        source = " ".join(f"https://example.org/{index}" for index in range(25))
        masked, urls = TRANSLATOR.mask_urls(source)
        self.assertEqual(len(urls), 25)
        self.assertEqual(
            TRANSLATOR.restore_markers(
                masked,
                TRANSLATOR.URL_MARKER_PATTERN,
                "URL",
                urls,
                TRANSLATOR.UrlStructureError,
            ),
            source,
        )

    def test_out_of_order_url_markers_are_rejected(self) -> None:
        with self.assertRaises(TRANSLATOR.UrlStructureError):
            TRANSLATOR.restore_markers(
                "[[URL_1]] [[URL_0]]",
                TRANSLATOR.URL_MARKER_PATTERN,
                "URL",
                ["https://one.example", "https://two.example"],
                TRANSLATOR.UrlStructureError,
            )

    def test_url_inside_code_block_is_not_masked_as_url(self) -> None:
        source = "Visit https://outside.example.\n```python\nurl = 'https://inside.example'\n```"
        code_masked, blocks = TRANSLATOR.mask_code_blocks(source)
        url_masked, urls = TRANSLATOR.mask_urls(code_masked)
        self.assertEqual(urls, ["https://outside.example."])
        self.assertIn("https://inside.example", blocks[0])
        self.assertNotIn("https://inside.example", url_masked)

    def test_code_block_round_trip(self) -> None:
        source = "Use this:\n```python\nprint('ok')\n```"
        masked, blocks = TRANSLATOR.mask_code_blocks(source)
        restored = TRANSLATOR.restore_markers(
            masked,
            TRANSLATOR.CODE_MARKER_PATTERN,
            "CODE_BLOCK",
            blocks,
            TRANSLATOR.CodeStructureError,
        )
        TRANSLATOR.validate_structure(source, restored)
        self.assertEqual(restored, source)

    def test_missing_code_marker_is_rejected(self) -> None:
        with self.assertRaises(TRANSLATOR.CodeStructureError):
            TRANSLATOR.restore_markers(
                "Before [[CODE_BLOCK_0]] after",
                TRANSLATOR.CODE_MARKER_PATTERN,
                "CODE_BLOCK",
                ["first", "second"],
                TRANSLATOR.CodeStructureError,
            )

    def test_duplicated_code_marker_is_rejected(self) -> None:
        with self.assertRaises(TRANSLATOR.CodeStructureError):
            TRANSLATOR.restore_markers(
                "[[CODE_BLOCK_0]] [[CODE_BLOCK_0]] [[CODE_BLOCK_1]]",
                TRANSLATOR.CODE_MARKER_PATTERN,
                "CODE_BLOCK",
                ["first", "second"],
                TRANSLATOR.CodeStructureError,
            )

    def test_out_of_order_code_marker_is_rejected(self) -> None:
        with self.assertRaises(TRANSLATOR.CodeStructureError):
            TRANSLATOR.restore_markers(
                "[[CODE_BLOCK_1]] [[CODE_BLOCK_0]]",
                TRANSLATOR.CODE_MARKER_PATTERN,
                "CODE_BLOCK",
                ["first", "second"],
                TRANSLATOR.CodeStructureError,
            )

    def test_new_cjk_character_is_reported(self) -> None:
        added = TRANSLATOR.unexpected_added_chars("A source mentions 户.", "Uma traducao com 新.", "pt_br")
        self.assertEqual(added, {"cjk": {"新": 1}})

    def test_existing_cjk_character_is_not_reported(self) -> None:
        added = TRANSLATOR.unexpected_added_chars("O caractere 户 aparece.", "O caractere 户 permanece.", "pt_br")
        self.assertEqual(added, {})

    def test_arabic_allows_arabic_and_latin_but_not_cjk(self) -> None:
        self.assertEqual(
            TRANSLATOR.unexpected_added_chars("Hello", "مرحبا Hello", "ar"),
            {},
        )
        self.assertEqual(
            TRANSLATOR.unexpected_added_chars("Hello", "مرحبا 新", "ar"),
            {"cjk": {"新": 1}},
        )


    def test_output_truncation_records_response_diagnostics(self) -> None:
        response = TRANSLATOR.TranslationResponse(
            text="partial output",
            finish_reason="length",
            prompt_tokens=120,
            completion_tokens=4096,
            total_tokens=4216,
            model="test-model",
        )
        with patch.object(TRANSLATOR, "translate_model_text", return_value=response) as mocked:
            result = TRANSLATOR.translate_field(
                object(),
                "test-model",
                source="x" * 600,
                language_code="pt_br",
                system_prompt="",
                fewshot=[],
                max_tokens=None,
            )

        self.assertEqual(result.fallback_type, "output_truncated")
        self.assertEqual(result.diagnostics["attempts"][0]["finish_reason"], "length")
        self.assertEqual(result.diagnostics["attempts"][0]["completion_tokens"], 4096)
        self.assertEqual(mocked.call_count, 2)

    def test_short_output_is_distinct_from_generation_limit(self) -> None:
        response = TRANSLATOR.TranslationResponse(
            text="short",
            finish_reason="stop",
            prompt_tokens=120,
            completion_tokens=5,
            total_tokens=125,
            model="test-model",
        )
        with patch.object(TRANSLATOR, "translate_model_text", return_value=response):
            result = TRANSLATOR.translate_field(
                object(),
                "test-model",
                source="x" * 600,
                language_code="pt_br",
                system_prompt="",
                fewshot=[],
                max_tokens=None,
            )

        self.assertEqual(result.fallback_type, "short_output")
        self.assertEqual(result.diagnostics["attempts"][0]["finish_reason"], "stop")

    def test_api_failures_are_classified_by_exception_type(self) -> None:
        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        self.assertEqual(
            TRANSLATOR.warning_type_for_error(TRANSLATOR.APITimeoutError(request)),
            "api_timeout",
        )
        self.assertEqual(
            TRANSLATOR.warning_type_for_error(
                TRANSLATOR.APIStatusError(
                    "server error",
                    response=httpx.Response(503, request=request),
                    body=None,
                )
            ),
            "api_status_error",
        )

    def test_retry_recovery_is_emitted_as_audit_event(self) -> None:
        first = TRANSLATOR.TranslationResponse(
            text="Visite o site.",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=4,
            total_tokens=14,
            model="test-model",
        )
        second = TRANSLATOR.TranslationResponse(
            text="Visite [[URL_0]]",
            finish_reason="stop",
            prompt_tokens=20,
            completion_tokens=6,
            total_tokens=26,
            model="test-model",
        )
        row = {field: "" for field in TRANSLATOR.TEXT_FIELDS}
        row.update({"context": "Visit https://example.org", "category": "test"})

        with patch.object(TRANSLATOR, "translate_model_text", side_effect=[first, second]):
            _, warnings = TRANSLATOR.translate_row(
                object(),
                "test-model",
                dataset="squad_v2",
                row_index=0,
                row=row,
                language_code="pt_br",
                system_prompt="",
                fewshot=[],
                max_tokens=None,
            )

        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["type"], "retry_recovered")
        self.assertIn("url_structure_fallback", warnings[0]["reason"])
        self.assertEqual(warnings[0]["diagnostics"]["attempts"][1]["outcome"], "accepted")


if __name__ == "__main__":
    unittest.main()
