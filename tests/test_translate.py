"""Checks for deterministic translation-pipeline safeguards without API calls."""

import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "translate.py"
SPEC = importlib.util.spec_from_file_location("translate", SCRIPT_PATH)
TRANSLATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
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


if __name__ == "__main__":
    unittest.main()
