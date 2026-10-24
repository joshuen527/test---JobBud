"""Offline checks for rate limits, corrupted caches and recovery."""
import json
import tempfile
import time
import unittest
from pathlib import Path

# Make the project's src/ modules importable when tests run via unittest discovery.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from unittest.mock import Mock, patch

import requests
import gdelt_engine as gdelt


class AITrendRecoveryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "signals.json"
        for target, value in [
            ("CACHE_FILE", str(self.path)), ("_retry_after", 0),
        ]:
            p = patch.object(gdelt, target, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(gdelt.time, "sleep")
        self.sleep = p.start()
        self.addCleanup(p.stop)
        p = patch.object(gdelt.requests, "get", side_effect=AssertionError("No live requests in tests"))
        self.request = p.start()
        self.addCleanup(p.stop)

    def record(self, **overrides):
        return dict(recent=60, previous=40, change=50, score=75,
                    direction="Rising", **overrides)

    def test_rate_limit_stops_whole_batch_without_retry_sleep(self):
        self.request.side_effect = None
        self.request.return_value = Mock(status_code=429)
        result = gdelt.get_ai_skill_signals()
        self.assertEqual(len(result), len(gdelt.TECH_QUERIES))
        self.assertTrue(all(s["score"] is None for s in result.values()))
        self.assertEqual(self.request.call_count, 1)
        self.sleep.assert_not_called()
        gdelt.get_ai_skill_signals()
        self.assertEqual(self.request.call_count, 1)

    def test_timeout_is_bounded_and_does_not_retry_every_topic(self):
        self.request.side_effect = requests.Timeout("test timeout")
        result = gdelt.get_ai_skill_signals()
        self.assertTrue(all(s["direction"] == "Unavailable" for s in result.values()))
        self.assertEqual(self.request.call_count, 1)
        self.assertEqual(self.request.call_args.kwargs["timeout"], 10)
        self.sleep.assert_not_called()

    def test_recovery_after_cooldown_preserves_valid_cache(self):
        good = self.record(fetched_at=time.time())
        self.path.write_text(json.dumps({"Artificial Intelligence": good}))
        self.request.side_effect = None
        self.request.return_value = Mock(status_code=429)
        first = gdelt.get_ai_skill_signals()
        self.assertEqual(first["Artificial Intelligence"]["score"], 75)
        gdelt._retry_after = 0
        with patch.object(gdelt, "analyse_topic", side_effect=lambda name, query: self.record()) as analyse:
            second = gdelt.get_ai_skill_signals()
        self.assertEqual(analyse.call_count, len(gdelt.TECH_QUERIES) - 1)
        self.assertTrue(all(s["score"] == 75 for s in second.values()))

    def test_corrupt_and_non_object_caches_are_ignored(self):
        for text in ("{broken", "[]", "null", '"text"', '{"Artificial Intelligence": 4}'):
            with self.subTest(text=text):
                self.path.write_text(text)
                self.assertFalse(gdelt.load_gdelt_cache())

    def test_invalid_scores_never_affect_role_scores(self):
        for bad in (True, "75", float("nan"), float("inf"), -1, 101):
            with self.subTest(bad=bad):
                invalid = {**self.record(), "score": bad}
                result = gdelt.get_role_ai_signal("Data Scientists", {
                    "Artificial Intelligence": invalid,
                    "Machine Learning": self.record(),
                })
                self.assertEqual(result["score"], 75)
                self.assertEqual(result["topics"], ["Machine Learning"])
        missing = {**self.record(), "direction": "Unavailable"}
        self.assertIsNone(gdelt.get_role_ai_signal(
            "Data Scientists", {"Artificial Intelligence": missing},
        )["score"])

    def test_partial_refresh_does_not_make_old_topic_fresh(self):
        gdelt.save_gdelt_cache({
            "Artificial Intelligence": self.record(fetched_at=time.time() - 7 * 3600),
            "Machine Learning": self.record(fetched_at=time.time()),
        })
        self.assertEqual(set(gdelt.load_gdelt_cache()), {"Machine Learning"})

    def test_invalid_observations_are_ignored_without_crashing(self):
        bad = [None, {}, {"date": []}, {"date": "2026-10-01", "value": True}]
        bad += [{"date": "2026-10-01", "value": v} for v in (None, "oops", -1, "NaN", "Infinity")]
        with patch.object(gdelt, "fetch_timeline", return_value=bad):
            self.assertIsNone(gdelt.analyse_topic("Artificial Intelligence", "q")["score"])

    def test_valid_signal_keeps_existing_formula(self):
        rows = [{"date": "20261001T000000Z", "value": 40},
                {"date": "20261020T000000Z", "value": 60}]
        with patch.object(gdelt, "fetch_timeline", return_value=rows):
            result = gdelt.analyse_topic("Artificial Intelligence", "q")
        self.assertEqual(result["score"], 75)
        self.assertEqual(result["change"], 50)


if __name__ == "__main__":
    unittest.main()
