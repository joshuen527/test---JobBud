import time
import unittest
from pathlib import Path

# Make the project's src/ modules importable when tests run via unittest discovery.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from threading import Event
from unittest.mock import Mock, patch

import streamlit as st
from streamlit.testing.v1 import AppTest
import ai_explainer
import background_data
import gdelt_engine
import salary_engine
import trend_engine


class BackgroundLoadingTests(unittest.TestCase):
    def tearDown(self):
        with background_data._lock:
            futures = [entry["future"] for entry in background_data._entries.values()]
        for future in futures:
            try:
                future.result(timeout=5)
            except Exception:
                pass
        with background_data._lock:
            background_data._entries.clear()
        st.cache_data.clear()

    def test_slow_request_returns_immediately_and_is_not_duplicated(self):
        release = Event()
        loader = Mock(side_effect=lambda: (release.wait(5), {"value": 42})[1])
        try:
            start = time.monotonic()
            for _ in range(5):
                value, pending = background_data.get_background_value("slow-test", loader, {})
                self.assertTrue(pending)
                self.assertEqual(value, {})
            self.assertLess(time.monotonic() - start, 0.5)
        finally:
            release.set()
        background_data._entries["slow-test"]["future"].result(timeout=2)
        self.assertEqual(background_data.get_background_value("slow-test", loader, {}), ({"value": 42}, False))
        self.assertEqual(loader.call_count, 1)

    def test_provider_failure_is_cached_not_retried_on_every_render(self):
        loader = Mock(side_effect=RuntimeError("test outage"))
        background_data.get_background_value("failed-test", loader, {"available": False})
        try:
            background_data._entries["failed-test"]["future"].result(timeout=2)
        except RuntimeError:
            pass
        for _ in range(5):
            self.assertEqual(
                background_data.get_background_value("failed-test", loader, {"available": False}),
                ({"available": False}, False),
            )
        self.assertEqual(loader.call_count, 1)

    def test_matches_render_while_external_services_are_blocked_then_refresh(self):
        st.cache_data.clear()
        release = Event()
        market = {
            "jobs": [], "source": "Refreshed test source", "from_cache": False,
            "stale": False, "listings_analysed": 0,
        }

        def slow_market():
            release.wait(30)
            return market

        def slow_gdelt():
            release.wait(30)
            return {}

        with patch.object(trend_engine, "fetch_job_data", side_effect=slow_market) as fetch, \
                patch.object(trend_engine, "cached_job_data", return_value={**market, "source": "Cached test source"}), \
                patch.object(gdelt_engine, "get_ai_skill_signals", side_effect=slow_gdelt) as gdelt_fetch, \
                patch.object(gdelt_engine, "load_gdelt_cache", return_value={}), \
                patch.object(salary_engine, "get_salary_benchmark", return_value={"available": False}), \
                patch.object(ai_explainer, "explain_recommendation", return_value={
                    "text": "Completed explanation", "source": "RunPod",
                }) as explain:
            try:
                app = AppTest.from_file(str(Path("src/app.py").resolve()), default_timeout=8).run()
                app.multiselect(key="selected_skills").set_value(["Mathematics", "Critical Thinking"]).run()
                start = time.monotonic()
                app.button[0].click().run()
                elapsed = time.monotonic() - start
                self.assertFalse(app.exception, [e.message for e in app.exception])
                self.assertTrue(app.session_state["assessment"]["matches"])
                self.assertIn("market", app.session_state["assessment"]["pending_keys"])
                self.assertEqual(explain.call_count, 0)
                self.assertLess(elapsed, 4)
                self.assertTrue(any("Your Job Buddy. Every career starts somewhere" in e.value for e in app.caption))
                print(f"\nInitial matches visible in {elapsed:.2f}s with external services blocked.")
                original_scores = [(m["role"], m["base_match"]) for m in app.session_state["assessment"]["matches"]]
            finally:
                release.set()
            for _ in range(10):
                app.run()
                if not app.session_state["assessment"]["pending_keys"]:
                    break
            self.assertFalse(app.exception, [e.message for e in app.exception])
            self.assertFalse(app.session_state["assessment"]["pending_keys"])
            self.assertEqual(app.session_state["assessment"]["market"]["source"], "Refreshed test source")
            self.assertEqual(original_scores, [(m["role"], m["base_match"]) for m in app.session_state["assessment"]["matches"]])
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(explain.call_count, 1)
            # Simulate cooldown expiry without waiting five minutes. A page
            # refresh/fragment tick must recover without resubmitting the form.
            self.assertIsNotNone(app.session_state["assessment"]["ai_retry_at"])
            gdelt_fetch.side_effect = lambda: {
                topic: {"recent": 60, "previous": 40, "change": 50,
                        "score": 75, "direction": "Rising"}
                for topic in gdelt_engine.TECH_QUERIES
            }
            with background_data._lock:
                background_data._entries["gdelt"]["expires"] = 0
            app.session_state["assessment"]["ai_retry_at"] = 0
            for _ in range(10):
                app.run()
                if app.session_state["assessment"]["gdelt_signals"]:
                    break
            self.assertFalse(app.exception, [e.message for e in app.exception])
            self.assertEqual(gdelt_fetch.call_count, 2)
            self.assertEqual(len(app.session_state["assessment"]["gdelt_signals"]), 6)
            self.assertIsNone(app.session_state["assessment"]["ai_retry_at"])


if __name__ == "__main__":
    unittest.main()
