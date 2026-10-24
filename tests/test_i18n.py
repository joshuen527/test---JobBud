import ast
import json
import unittest
from pathlib import Path

# Make the project's src/ modules importable when tests run via unittest discovery.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from string import Formatter
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

import ai_explainer
import gdelt_engine
import salary_engine
import trend_engine
from career_engine import QUESTIONNAIRE_KNOWLEDGE, QUESTIONNAIRE_SKILLS
from i18n import TEXT, TERM_LABELS, SDG_RATIONALES, t, term_label


class TranslationTests(unittest.TestCase):
    def test_catalog_has_all_languages_and_matching_placeholders(self):
        formatter = Formatter()
        for key, entry in TEXT.items():
            with self.subTest(key=key):
                self.assertTrue(all(entry.get(lang) for lang in ("en", "ms", "zh")))
                fields = lambda text: {
                    name for _, name, _, _ in formatter.parse(text) if name is not None
                }
                self.assertEqual(fields(entry["en"]), fields(entry["ms"]))
                self.assertEqual(fields(entry["en"]), fields(entry["zh"]))

    def test_all_app_translation_keys_exist(self):
        tree = ast.parse(Path("src/app.py").read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in ("t", "tf") and node.args
                    and isinstance(node.args[0], ast.Constant)):
                self.assertIn(node.args[0].value, TEXT)

    def test_questionnaire_and_sdg_translations_are_complete(self):
        for value in [*QUESTIONNAIRE_SKILLS, *QUESTIONNAIRE_KNOWLEDGE]:
            self.assertIn(value, TERM_LABELS)
        mappings = json.loads(Path("config/sdg_mapping.json").read_text())
        for entries in [*mappings["careers"].values(), *mappings["skill_categories"].values()]:
            for item in entries:
                self.assertTrue(SDG_RATIONALES[item["rationale"]]["ms"])
                self.assertTrue(SDG_RATIONALES[item["rationale"]]["zh"])

    def test_local_explanations_use_requested_language(self):
        match = {
            "role": "Data Scientists", "trend_direction": "Insufficient signal",
            "strengths": [{"skill": "Mathematics", "group": "Essential", "user": 4, "target": 3}],
            "gaps": [],
        }
        for language in ("en", "ms", "zh"):
            text = ai_explainer.fallback_explanation(match, language)
            self.assertIn(term_label("Mathematics", language), text)
            self.assertIn(t("explanation_insufficient", language), text)

    def test_language_switch_preserves_answers_scores_and_avoids_new_requests(self):
        st.cache_data.clear()
        market = {
            "jobs": [], "source": "Test fixture", "fetched_at": None,
            "from_cache": True, "stale": False, "listings_analysed": 0,
        }
        with patch.object(trend_engine, "fetch_job_data", return_value=market) as fetch, \
                patch.object(gdelt_engine, "get_ai_skill_signals", return_value={}), \
                patch.object(salary_engine, "get_salary_benchmark", return_value={"available": False}), \
                patch.object(ai_explainer, "explain_recommendation", return_value={
                    "text": "Test explanation", "source": "RunPod",
                }) as explain:
            app = AppTest.from_file(str(Path("src/app.py").resolve()), default_timeout=30).run()
            self.assertFalse(app.exception)
            app.multiselect(key="selected_skills").set_value(["Mathematics", "Critical Thinking"]).run()
            app.multiselect(key="selected_knowledge").set_value(["Economics and Accounting"]).run()
            app.slider(key="selected_skill_Mathematics").set_value(4).run()
            app.button[0].click().run()
            self.assertFalse(app.exception, [e.message for e in app.exception])
            before = app.session_state["assessment"]
            scores = [(m["role"], m["base_match"], m["future_match"]) for m in before["matches"]]
            self.assertTrue(scores)
            for language in ("ms", "zh", "en"):
                app.selectbox(key="language").set_value(language).run()
                self.assertFalse(app.exception, [e.message for e in app.exception])
                self.assertIn(t("build_profile", language), [e.value for e in app.header])
                self.assertIn(t("sources", language), [e.value for e in app.header])
                self.assertEqual(app.slider(key="selected_skill_Mathematics").value, 4)
                self.assertEqual(app.multiselect(key="selected_skills").value, ["Mathematics", "Critical Thinking"])
                after = app.session_state["assessment"]
                self.assertEqual(scores, [(m["role"], m["base_match"], m["future_match"]) for m in after["matches"]])
                if language != "en":
                    self.assertTrue(any(t("explanation_insufficient", language) in e.value for e in app.markdown))
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(explain.call_count, 1)


if __name__ == "__main__":
    unittest.main()
