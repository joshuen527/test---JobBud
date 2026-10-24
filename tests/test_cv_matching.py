"""Synthetic snippets only: never persist uploaded CVs or their contents."""
import ast
import re
import unicodedata
import unittest
from io import BytesIO
from pathlib import Path

# Make the project's src/ modules importable when tests run via unittest discovery.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from career_engine import QUESTIONNAIRE_KNOWLEDGE, QUESTIONNAIRE_SKILLS
from i18n import CV_TERM_ALIASES, cv_term_aliases


# Load the actual pure functions without executing the Streamlit UI or providers.
tree = ast.parse(Path("src/app.py").read_text())
functions = ast.Module(body=[
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name in ("cv_contains_term", "detect_cv_items")
], type_ignores=[])
namespace = {
    "re": re, "unicodedata": unicodedata, "cv_term_aliases": cv_term_aliases,
    "QUESTIONNAIRE_SKILLS": QUESTIONNAIRE_SKILLS,
    "QUESTIONNAIRE_KNOWLEDGE": QUESTIONNAIRE_KNOWLEDGE,
}
exec(compile(functions, "app.py", "exec"), namespace)
contains = namespace["cv_contains_term"]
detect = namespace["detect_cv_items"]


class CVMatchingTests(unittest.TestCase):
    def test_every_canonical_item_and_reviewed_alias(self):
        terms = set(QUESTIONNAIRE_SKILLS + QUESTIONNAIRE_KNOWLEDGE)
        self.assertEqual(set(CV_TERM_ALIASES), terms)
        for term in terms:
            for language in ("en", "ms", "zh"):
                aliases = (term,) if language == "en" else CV_TERM_ALIASES[term][language]
                for alias in aliases:
                    with self.subTest(term=term, language=language, alias=alias):
                        result = detect(f"• {alias};", [])
                        expected = {
                            "skills": [term] if term in QUESTIONNAIRE_SKILLS else [],
                            "knowledge": [term] if term in QUESTIONNAIRE_KNOWLEDGE else [],
                            "technologies": [],
                        }
                        self.assertEqual(result, expected)

    def test_equivalent_languages_produce_canonical_inputs(self):
        expected = {
            "skills": ["Critical Thinking", "Mathematics", "Time Management"],
            "knowledge": ["Mathematics", "Sales and Marketing"],
            "technologies": ["Python"],
        }
        for snippet in (
            "Critical Thinking; Mathematics; Time Management; Sales and Marketing; Python",
            "Pemikiran Kritis; Matematik; Pengurusan Masa; Jualan dan Pemasaran; Python",
            "批判性思维；数学；时间管理；销售与营销；Python",
            "Critical Thinking; Pemikiran Kritis; 批判性思维; Matematik; 时间管理; 销售与营销; Python",
        ):
            self.assertEqual(detect(snippet, ["Python", "SQL"]), expected)

    def test_normalization_and_unspaced_chinese_phrases(self):
        self.assertTrue(contains("PENGURUSAN\n\tMASA", "Time Management"))
        self.assertTrue(contains("具备时间管理和批判性思维能力。", "Time Management"))
        self.assertTrue(contains("Ｐｙｔｈｏｎ", "Python"))
        self.assertTrue(contains("C++ / C# / .NET", "C++"))

    def test_ambiguous_substrings_and_partial_subjects_are_not_skills(self):
        for snippet, term in (
            ("Matematikawan", "Mathematics"), ("Sainsbury", "Science"),
            ("copywriting", "Writing"), ("Pythonic", "Python"),
            ("MySQL", "SQL"), ("科学家在科学院", "Science"),
            ("数学家", "Mathematics"), ("英语系", "English Language"),
            ("指导方针", "Instructing"), ("不协调", "Coordination"),
            ("销售", "Sales and Marketing"), ("Perakaunan", "Economics and Accounting"),
            ("X时间管理Y", "Time Management"),
        ):
            with self.subTest(snippet=snippet, term=term):
                self.assertFalse(contains(snippet, term))

    def test_empty_and_unavailable_tools(self):
        for text in ("", " \n ", "unrelated text"):
            self.assertEqual(detect(text, ["Python"]), {
                "skills": [], "knowledge": [], "technologies": [],
            })
        self.assertFalse(contains("anything", " "))
        self.assertEqual(detect("Python; SQL", ["SQL"])["technologies"], ["SQL"])

    def test_upload_suggestions_remain_editable_and_do_not_assess_automatically(self):
        upload = BytesIO("Pemikiran Kritis; 数学; Python".encode("utf-8"))
        upload.name = "synthetic.txt"
        with patch.object(st, "file_uploader", return_value=upload):
            app = AppTest.from_file(str(Path("src/app.py").resolve()), default_timeout=30).run()
            app.radio(key="intake_method").set_value("Upload CV").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.multiselect(key="selected_skills").value,
                             ["Critical Thinking", "Mathematics"])
            self.assertIsNone(app.session_state["assessment"])
            app.multiselect(key="selected_skills").set_value(["Mathematics"]).run()
            app.slider(key="selected_skill_Mathematics").set_value(4).run()
            for language in ("ms", "zh", "en"):
                app.selectbox(key="language").set_value(language).run()
                self.assertFalse(app.exception)
                self.assertEqual(app.multiselect(key="selected_skills").value, ["Mathematics"])
                self.assertEqual(app.slider(key="selected_skill_Mathematics").value, 4)
                self.assertIsNone(app.session_state["assessment"])


if __name__ == "__main__":
    unittest.main()
