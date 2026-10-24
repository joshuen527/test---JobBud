"""Offline regression tests for the application-level GDELT adjustment."""
import ast
from copy import deepcopy
from pathlib import Path

# Make the project's src/ modules importable when tests run via unittest discovery.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import unittest
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

import gdelt_engine
from i18n import t


APP = Path(__file__).resolve().parents[1] / "src" / "app.py"
# Execute the production function without starting the UI or background loaders.
tree = ast.parse(APP.read_text())
namespace = {"get_role_ai_signal": gdelt_engine.get_role_ai_signal}
exec(compile(ast.Module(body=[
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "apply_gdelt_to_scenarios"
], type_ignores=[]), str(APP), "exec"), namespace)
apply_gdelt = namespace["apply_gdelt_to_scenarios"]


def signals(score):
    return {
        topic: {"score": score, "direction": "Unavailable" if score is None else "Stable",
                "recent": 50, "previous": 50, "change": 0}
        for topic in gdelt_engine.TECH_QUERIES
    }


def scenario(moderate=60.0, fast=70.0, role="Data Scientists"):
    return {
        "role": role, "base_match": 65.0, "trend_score": None,
        "moderate_ai_disruption": moderate, "fast_ai_disruption": fast,
        "moderate_weights": {"base_match": 1.0},
        "fast_weights": {"base_match": 1.0},
    }


class ScenarioAdjustmentTests(unittest.TestCase):
    def apply_without_mutation(self, rows, evidence):
        before, evidence_before = deepcopy(rows), deepcopy(evidence)
        result = apply_gdelt(rows, evidence)
        self.assertEqual(rows, before)
        self.assertEqual(evidence, evidence_before)
        self.assertIsNot(result, rows)
        self.assertEqual(len(result), len(rows))
        for original, updated in zip(rows, result):
            self.assertIsNot(original, updated)
            for key in original.keys() - {
                "moderate_ai_disruption", "fast_ai_disruption",
            }:
                self.assertEqual(updated[key], original[key])
        return result

    def test_unavailable_and_absent_signals_leave_scores_unchanged(self):
        for evidence in ({}, signals(None)):
            with self.subTest(evidence=evidence):
                row = scenario(61.234, 72.345)
                result = self.apply_without_mutation([row], evidence)[0]
                self.assertEqual(result["moderate_ai_disruption"], 61.234)
                self.assertEqual(result["fast_ai_disruption"], 72.345)
                for key in ("gdelt_attention_score", "gdelt_moderate_adjustment",
                            "gdelt_fast_adjustment"):
                    self.assertIsNone(result[key])
                self.assertEqual(result["gdelt_direction"], "Insufficient signal")
                self.assertEqual(result["gdelt_topics"], [])

    def test_neutral_extremes_fractional_rounding_and_clamps(self):
        for score, moderate, fast, expected, adjustments in (
            (50, 60, 70, (60, 70), (0, 0)),
            (0, 60, 70, (55, 60), (-5, -10)),
            (100, 60, 70, (65, 80), (5, 10)),
            (0, 2, 3, (0, 0), (-5, -10)),
            (100, 98, 99, (100, 100), (5, 10)),
            (62.5, 60, 70, (61.2, 72.5), (1.2, 2.5)),
        ):
            with self.subTest(score=score, moderate=moderate, fast=fast):
                result = self.apply_without_mutation(
                    [scenario(moderate, fast)], signals(score),
                )[0]
                self.assertEqual(
                    (result["moderate_ai_disruption"], result["fast_ai_disruption"]),
                    expected,
                )
                self.assertEqual(
                    (result["gdelt_moderate_adjustment"], result["gdelt_fast_adjustment"]),
                    adjustments,
                )
                self.assertEqual(result["gdelt_attention_score"], score)
                self.assertEqual(result["gdelt_topics"],
                                 gdelt_engine.ROLE_TOPICS["Data Scientists"])

    def test_missing_values_remain_none_or_absent_independently(self):
        for score in (None, 0, 50, 100):
            for key in ("moderate_ai_disruption", "fast_ai_disruption"):
                for absent in (False, True):
                    with self.subTest(score=score, key=key, absent=absent):
                        row = scenario()
                        if absent:
                            del row[key]
                        else:
                            row[key] = None
                        result = self.apply_without_mutation([row], signals(score))[0]
                        if absent:
                            self.assertNotIn(key, result)
                        else:
                            self.assertIsNone(result[key])
                        other = ("fast_ai_disruption" if key == "moderate_ai_disruption"
                                 else "moderate_ai_disruption")
                        delta = 0 if score is None else (score - 50) / 50
                        expected = (70 + delta * 10 if other == "fast_ai_disruption"
                                    else 60 + delta * 5)
                        self.assertEqual(result[other], expected)

    def test_empty_and_mixed_roles_preserve_order_and_independent_values(self):
        self.assertEqual(self.apply_without_mutation([], signals(100)), [])
        rows = [scenario(), scenario(20, 30, "Unknown role"), scenario(40, 50)]
        result = self.apply_without_mutation(rows, signals(100))
        self.assertEqual([r["role"] for r in result], [r["role"] for r in rows])
        self.assertEqual([r["moderate_ai_disruption"] for r in result], [65, 20, 45])
        self.assertEqual([r["fast_ai_disruption"] for r in result], [80, 30, 60])
        self.assertIsNone(result[1]["gdelt_attention_score"])


class ScenarioDisplayTests(unittest.TestCase):
    def test_assessment_renders_adjusted_metrics_and_signed_summary(self):
        # Run loaders synchronously so there are no live requests, cached provider
        # results, or background futures leaking between AppTest runs.
        for score, moderate, fast, expected in (
            (None, 60, 70, ("60.0%", "70.0%")),
            (50, 60, 70, ("60.0%", "70.0%")),
            (0, 2, 3, ("0.0%", "0.0%")),
            (100, 98, 99, ("100.0%", "100.0%")),
            (100, None, None, (t("unavailable", "en"), t("unavailable", "en"))),
        ):
            with self.subTest(score=score, moderate=moderate):
                st.cache_data.clear()
                self.addCleanup(st.cache_data.clear)
                rows = [scenario(moderate, fast), scenario(20, 30, "Unknown role")]
                original = deepcopy(rows)
                with patch("requests.sessions.Session.request",
                           side_effect=AssertionError("Network forbidden in scenario tests")), \
                        patch("ai_trend_sources.add_backup_signals", side_effect=lambda signals: signals), \
                        patch("background_data.get_background_value",
                              side_effect=lambda key, loader, fallback, **kw: (loader(), False)), \
                        patch("background_data.any_completed", return_value=False), \
                        patch("career_engine.get_top_matches", return_value=[
                            {"role": row["role"], "overall": 65, "base_match": 65,
                             "strengths": [], "gaps": []} for row in rows
                        ]), \
                        patch("scenario_engine.build_scenario_comparison", return_value=rows), \
                        patch("gdelt_engine.get_ai_skill_signals", return_value=signals(score)) as load_ai, \
                        patch("gdelt_engine.load_gdelt_cache", return_value={}), \
                        patch("trend_engine.fetch_job_data", return_value={"jobs": []}), \
                        patch("trend_engine.cached_job_data", return_value={"jobs": []}), \
                        patch("salary_engine.get_salary_benchmark", return_value={"available": False}), \
                        patch("ai_explainer.explain_recommendation",
                              return_value={"text": "Offline fixture", "source": "Rule-based fallback"}):
                    app = AppTest.from_file(str(APP), default_timeout=30).run()
                    app.multiselect(key="selected_skills").set_value(["Mathematics"]).run()
                    app.button[0].click().run()
                    self.assertFalse(app.exception, [e.message for e in app.exception])
                    self.assertIsNotNone(app.session_state["assessment"])
                    self.assertEqual(rows, original)
                    metrics = {m.label: m.value for m in app.metric}
                    self.assertEqual(metrics[t("moderate_ai", "en")], expected[0])
                    self.assertEqual(metrics[t("fast_ai", "en")], expected[1])
                    captions = "\n".join(c.value for c in app.caption)
                    if score is None:
                        self.assertNotIn(t("moderate_adjustment", "en") + ":", captions)
                        self.assertNotIn(t("fast_adjustment", "en") + ":", captions)
                        self.assertNotIn(t("gdelt_attention_short", "en") + ":", captions)
                    else:
                        self.assertIn(f'{t("gdelt_attention_short", "en")}: {score:.1f}/100', captions)
                        for key, sensitivity in (("moderate_adjustment", 5), ("fast_adjustment", 10)):
                            adjustment = (score - 50) / 50 * sensitivity
                            self.assertIn(
                                f'{t(key, "en")}: {adjustment:+.1f} {t("points_short", "en")}',
                                captions,
                            )
                    # Role selection must not retain the previous role's adjustment.
                    app.selectbox(key="selected_role").set_value("Unknown role").run()
                    self.assertFalse(app.exception, [e.message for e in app.exception])
                    metrics = {m.label: m.value for m in app.metric}
                    self.assertEqual(metrics[t("moderate_ai", "en")], "20.0%")
                    self.assertEqual(metrics[t("fast_ai", "en")], "30.0%")
                    self.assertNotIn(t("moderate_adjustment", "en") + ":",
                                     "\n".join(c.value for c in app.caption))
                    load_ai.assert_called_once()


if __name__ == "__main__":
    unittest.main()
