"""Offline regression tests for GDELT retries, scoring and cache behavior."""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Make the project's src/ modules importable when tests run via unittest discovery.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from unittest.mock import Mock, call, patch

import requests

import gdelt_engine as gdelt


class GdeltTestCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cache_file = Path(directory.name) / "gdelt.json"
        self.now = datetime(2026, 10, 23, 12, tzinfo=timezone.utc)
        self.get = self.start_patch(
            "gdelt_engine.requests.get",
            side_effect=AssertionError("Unexpected GDELT request"),
        )
        self.sleep = self.start_patch("gdelt_engine.time.sleep")
        self.start_patch("gdelt_engine._retry_after", 0)
        self.clock = self.start_patch("gdelt_engine.time.monotonic", return_value=1000)
        self.start_patch("gdelt_engine.CACHE_FILE", self.cache_file)
        clock = self.start_patch("gdelt_engine.datetime")
        clock.now.return_value = self.now
        # Protect against accidental network access outside the mocked entrypoint.
        network = self.start_patch(
            "requests.sessions.Session.request",
            side_effect=AssertionError("Unexpected live network request"),
        )
        self.addCleanup(network.assert_not_called)

    def start_patch(self, target, *args, **kwargs):
        patcher = patch(target, *args, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def response(self, payload=None, status=200):
        response = Mock(status_code=status)
        response.json.return_value = payload
        if status >= 400:
            response.raise_for_status.side_effect = requests.HTTPError(str(status))
        return response

    def payload(self, rows):
        return {"timeline": [{"data": rows}]}

    def serve(self, payload):
        self.get.side_effect = None
        self.get.return_value = self.response(payload)

    def row(self, days_ago, value):
        return {
            "date": (self.now - timedelta(days=days_ago)).isoformat(),
            "value": value,
        }

    def signal(self, direction="Stable", score=50):
        return {
            "skill": "Artificial Intelligence",
            "recent": 10, "previous": 10, "change": 0 if score is not None else None,
            "direction": direction, "score": score,
        }

    def write_cache(self, contents, age=0):
        self.cache_file.write_text(contents, encoding="utf-8")
        timestamp = self.now.timestamp() - age
        os.utime(self.cache_file, (timestamp, timestamp))

    def assert_no_signal(self, result, direction):
        self.assertEqual(result["direction"], direction)
        self.assertIsNone(result["score"])
        self.assertIsNone(result["change"])


class TimelineFetchTests(GdeltTestCase):
    def test_success_uses_expected_api_contract_without_sleep(self):
        rows = [self.row(0, 25)]
        self.serve(self.payload(rows))
        self.assertEqual(gdelt.fetch_timeline('"machine learning"'), rows)
        self.get.assert_called_once_with(
            gdelt.GDELT_URL,
            params={"query": '"machine learning"', "mode": "timelinevolraw",
                    "format": "json", "timespan": "28d"},
            timeout=10,
            headers={"User-Agent": "AI-Career-Navigator-University-Hackathon"},
        )
        self.sleep.assert_not_called()

    def test_rate_limit_recovers_after_cooldown(self):
        rows = [self.row(0, 25)]
        self.get.side_effect = [
            self.response(status=429),
            self.response(self.payload(rows)),
        ]
        with self.assertRaisesRegex(RuntimeError, "rate-limited"):
            gdelt.fetch_timeline("query")
        self.assertEqual(gdelt._retry_after, 1300)
        self.clock.return_value = 1299
        with self.assertRaisesRegex(RuntimeError, "cooldown"):
            gdelt.fetch_timeline("query")
        self.assertEqual(self.get.call_count, 1)
        self.clock.return_value = 1300
        self.assertEqual(gdelt.fetch_timeline("query"), rows)
        self.assertEqual(self.get.call_count, 2)
        self.sleep.assert_not_called()
        self.assertEqual(self.get.call_args_list, [self.get.call_args] * 2)

    def test_persistent_rate_limits_renew_cooldown_without_retry_loops(self):
        self.get.side_effect = [self.response(status=429) for _ in range(3)]
        for now in (1000, 1300, 1600):
            self.clock.return_value = now
            with self.assertRaisesRegex(RuntimeError, "rate-limited"):
                gdelt.fetch_timeline("query")
            self.assertEqual(gdelt._retry_after, now + 300)
        self.assertEqual(self.get.call_count, 3)
        self.sleep.assert_not_called()

    def test_request_errors_recover_only_after_cooldown(self):
        self.get.side_effect = [
            requests.Timeout("timeout"), requests.ConnectionError("offline"),
            self.response(self.payload([])),
        ]
        for now, exception in ((1000, requests.Timeout), (1300, requests.ConnectionError)):
            self.clock.return_value = now
            with self.assertRaises(exception):
                gdelt.fetch_timeline("query")
            with self.assertRaisesRegex(RuntimeError, "cooldown"):
                gdelt.fetch_timeline("query")
        self.clock.return_value = 1600
        self.assertEqual(gdelt.fetch_timeline("query"), [])
        self.assertEqual(self.get.call_count, 3)
        self.sleep.assert_not_called()

    def test_request_error_is_preserved_and_repeated_calls_are_blocked(self):
        last = requests.Timeout("last timeout")
        self.get.side_effect = last
        with self.assertRaises(requests.Timeout) as raised:
            gdelt.fetch_timeline("query")
        self.assertIs(raised.exception, last)
        with self.assertRaisesRegex(RuntimeError, "cooldown"):
            gdelt.fetch_timeline("query")
        self.assertEqual(self.get.call_count, 1)
        self.sleep.assert_not_called()

    def test_http_errors_become_unavailable_and_start_cooldown(self):
        self.get.side_effect = [self.response(status=503) for _ in range(3)]
        result = gdelt.analyse_topic("AI", "query")
        self.assert_no_signal(result, "Unavailable")
        self.assertIn("503", result["error"])
        self.assertEqual(self.get.call_count, 1)
        self.assertEqual(gdelt._retry_after, 1300)
        self.sleep.assert_not_called()

    def test_invalid_json_becomes_unavailable_and_starts_cooldown(self):
        response = self.response()
        response.json.side_effect = requests.exceptions.JSONDecodeError(
            "Invalid JSON", "<html>", 0,
        )
        self.get.side_effect = [response] * 3
        result = gdelt.analyse_topic("AI", "query")
        self.assert_no_signal(result, "Unavailable")
        self.assertIn("Invalid JSON", result["error"])
        self.assertEqual(self.get.call_count, 1)
        self.assertEqual(gdelt._retry_after, 1300)
        self.sleep.assert_not_called()

    def test_empty_timelines_are_insufficient_not_zero_scored(self):
        for payload in ({}, {"timeline": []}, {"timeline": None},
                        {"timeline": [{}]}, self.payload([])):
            with self.subTest(payload=payload):
                self.serve(payload)
                self.assert_no_signal(
                    gdelt.analyse_topic("AI", "query"), "Insufficient signal",
                )
        self.sleep.assert_not_called()

    def test_malformed_timelines_are_unavailable_without_a_score(self):
        for payload in (
            None, [], {"timeline": "bad"}, {"timeline": [None]},
            self.payload("bad"), self.payload([None]),
            self.payload([self.row(0, "not a number")]),
            self.payload([self.row(0, None)]),
            self.payload([self.row(0, float("nan"))]),
            self.payload([self.row(0, float("inf"))]),
            self.payload([self.row(0, -1)]),
            self.payload([self.row(0, True)]),
            self.payload([self.row(0, 100), self.row(20, "bad")]),
            self.payload(None), self.payload({}),
        ):
            with self.subTest(payload=payload):
                gdelt._retry_after = 0
                self.get.reset_mock()
                self.serve(payload)
                result = gdelt.analyse_topic("AI", "query")
                self.assert_no_signal(result, "Unavailable")
                self.assertTrue(result["error"])
                self.get.assert_called_once()
        self.sleep.assert_not_called()


class TopicAnalysisTests(GdeltTestCase):
    def analyse(self, rows):
        self.serve(self.payload(rows))
        return gdelt.analyse_topic("AI", "query")

    def test_invalid_dates_are_skipped(self):
        result = self.analyse([
            {"date": "invalid", "value": 999},
            {"value": 999}, {"date": "invalid"},
        ])
        self.assert_no_signal(result, "Insufficient signal")
        self.assertEqual((result["recent"], result["previous"]), (0, 0))
        result = self.analyse([{"date": "invalid", "value": 999}])
        self.assert_no_signal(result, "Insufficient signal")

    def test_missing_count_invalidates_mixed_timeline_and_is_excluded_from_role(self):
        for missing_days, valid_days in ((0, 20), (20, 0)):
            with self.subTest(missing_days=missing_days):
                missing = self.row(missing_days, 0)
                del missing["value"]
                result = self.analyse([missing, self.row(valid_days, 30)])
                self.assert_no_signal(result, "Unavailable")
                self.assertIn("missing its count", result["error"])
                role = gdelt.get_role_ai_signal("Data Scientists", {
                    "Artificial Intelligence": result,
                    "Generative AI": self.signal(score=80),
                })
                self.assertEqual(role, {
                    "score": 80, "direction": "Rising",
                    "topics": ["Generative AI"],
                })

    def test_window_boundaries_are_anchored_to_latest_observation(self):
        # Shift observations away from wall-clock time and supply them unsorted.
        rows = [self.row(104, 12), self.row(118, 999), self.row(90, 30),
                self.row(103, 10), self.row(117, 8)]
        result = self.analyse(rows)
        self.assertEqual(result, {
            "skill": "AI", "recent": 40, "previous": 20,
            "change": 100.0, "direction": "Rising", "score": 100,
        })

    def test_minimum_signal_threshold(self):
        self.assertEqual(gdelt.MIN_SIGNAL, 20)
        self.assert_no_signal(
            self.analyse([self.row(0, 9), self.row(20, 10)]),
            "Insufficient signal",
        )
        self.assertEqual(
            self.analyse([self.row(0, 10), self.row(20, 10)])["score"], 50,
        )

    def test_numeric_strings_and_rounding_keep_existing_behavior(self):
        result = self.analyse([self.row(0, "14"), self.row(20, "12")])
        self.assertEqual(result, {
            "skill": "AI", "recent": 14, "previous": 12,
            "change": 16.7, "direction": "Stable", "score": 58.3,
        })

    def test_scoring_and_direction_boundaries_remain_unchanged(self):
        for recent, previous, change, direction, score in (
            (121, 100, 21, "Rising", 60.5),
            (120, 100, 20, "Stable", 60),
            (80, 100, -20, "Stable", 40),
            (79, 100, -21, "Declining", 39.5),
            (100, 100, 0, "Stable", 50),
            (0, 20, -100, "Declining", 0),
            (20, 0, 2000, "Rising", 100),
        ):
            with self.subTest(recent=recent, previous=previous):
                result = self.analyse([self.row(0, recent), self.row(20, previous)])
                self.assertEqual(
                    (result["change"], result["direction"], result["score"]),
                    (change, direction, score),
                )

    def test_unavailable_topic_is_not_used_in_role_average(self):
        self.get.side_effect = [self.response(status=429) for _ in range(3)]
        unavailable = gdelt.analyse_topic("Artificial Intelligence", "query")
        self.assert_no_signal(unavailable, "Unavailable")
        result = gdelt.get_role_ai_signal("Data Scientists", {
            "Artificial Intelligence": unavailable,
            "Generative AI": self.signal(score=80),
            "Machine Learning": self.signal("Insufficient signal", None),
            "Automation": self.signal(score=0),  # Not a topic for this role.
        })
        self.assertEqual(result, {
            "score": 80, "direction": "Rising", "topics": ["Generative AI"],
        })

    def test_no_available_scores_and_unknown_roles_are_insufficient(self):
        for role in ("Data Scientists", "Unknown role"):
            with self.subTest(role=role):
                result = gdelt.get_role_ai_signal(role, {
                    "Artificial Intelligence": self.signal("Unavailable", None),
                    "Generative AI": self.signal("Insufficient signal", None),
                })
                self.assertEqual(result, {
                    "score": None, "direction": "Insufficient signal", "topics": [],
                })

    def test_role_average_keeps_zero_and_uses_existing_direction_boundaries(self):
        for scores, expected, direction in (
            ((0, 100), 50, "Stable"), ((60, 60), 60, "Stable"),
            ((40, 40), 40, "Stable"), ((60, 61), 60.5, "Rising"),
            ((39, 40), 39.5, "Declining"),
        ):
            with self.subTest(scores=scores):
                result = gdelt.get_role_ai_signal("Data Scientists", {
                    "Artificial Intelligence": self.signal(score=scores[0]),
                    "Generative AI": self.signal(score=scores[1]),
                })
                self.assertEqual(result["score"], expected)
                self.assertEqual(result["direction"], direction)
                self.assertEqual(result["topics"],
                                 ["Artificial Intelligence", "Generative AI"])


class CacheTests(GdeltTestCase):
    def test_missing_corrupt_and_unreadable_cache_return_none(self):
        self.assertIsNone(gdelt.load_gdelt_cache())
        self.write_cache("not JSON")
        self.assertIsNone(gdelt.load_gdelt_cache())
        self.write_cache("{}")
        with patch("builtins.open", side_effect=PermissionError("denied")):
            self.assertIsNone(gdelt.load_gdelt_cache())
        with patch("gdelt_engine.os.path.getmtime", side_effect=OSError("missing")):
            self.assertIsNone(gdelt.load_gdelt_cache())

    def test_cache_freshness_includes_exact_six_hour_boundary(self):
        self.assertEqual(gdelt.CACHE_HOURS, 6)
        cached = {"Artificial Intelligence": self.signal()}
        for age in (0, 6 * 3600 - 1, 6 * 3600, 6 * 3600 + 1):
            with self.subTest(age=age):
                self.write_cache(json.dumps(cached), age)
                self.assertEqual(
                    gdelt.load_gdelt_cache(),
                    {"Artificial Intelligence": {
                        **cached["Artificial Intelligence"],
                        "fetched_at": self.now.timestamp() - age,
                    }} if age <= 21600 else None,
                )
        self.get.assert_not_called()
        self.sleep.assert_not_called()

    def test_save_round_trip_uses_temporary_cache_only(self):
        cached = {"Artificial Intelligence": self.signal()}
        gdelt.save_gdelt_cache(cached)
        self.assertEqual(json.loads(self.cache_file.read_text()), cached)
        os.utime(self.cache_file, (self.now.timestamp(), self.now.timestamp()))
        self.assertEqual(gdelt.load_gdelt_cache(), {
            "Artificial Intelligence": {
                **cached["Artificial Intelligence"], "fetched_at": self.now.timestamp(),
            },
        })

    def test_fresh_cache_reuses_good_and_insufficient_topics_only(self):
        cached = {
            "good": self.signal(),
            "low": self.signal("Insufficient signal", None),
            "failed": self.signal("Unavailable", None),
            "invalid": {"score": None},
        }
        queries = {name: f"query-{name}" for name in (*cached, "missing")}
        self.write_cache(json.dumps(cached))
        with patch.object(gdelt, "TECH_QUERIES", queries), \
                patch.object(gdelt, "analyse_topic", return_value=self.signal()) as analyse:
            result = gdelt.get_ai_skill_signals()
        self.assertEqual(analyse.call_args_list, [
            call(name, queries[name]) for name in ("failed", "invalid", "missing")
        ])
        self.assertEqual(result["good"], {**cached["good"], "fetched_at": self.now.timestamp()})
        self.assertEqual(result["low"], {**cached["low"], "fetched_at": self.now.timestamp()})
        self.assertEqual(self.sleep.call_args_list, [call(8)] * 2)
        self.assertEqual(json.loads(self.cache_file.read_text()), result)
        self.get.assert_not_called()

    def test_expired_cache_refreshes_all_topics(self):
        self.write_cache(json.dumps({"AI": self.signal()}), age=21601)
        with patch.object(gdelt, "TECH_QUERIES", {"AI": "query"}), \
                patch.object(gdelt, "analyse_topic", return_value=self.signal()) as analyse:
            result = gdelt.get_ai_skill_signals()
        analyse.assert_called_once_with("AI", "query")
        self.sleep.assert_not_called()
        self.assertEqual(json.loads(self.cache_file.read_text()), result)

    def test_unavailable_provider_preserves_cached_topics_and_skips_missing_topics(self):
        self.write_cache(json.dumps({"good": self.signal()}))
        self.get.side_effect = [self.response(status=429)]
        with patch.object(gdelt, "TECH_QUERIES", {"failed": "q1", "good": "q2", "missing": "q3"}):
            result = gdelt.get_ai_skill_signals()
        self.assert_no_signal(result["failed"], "Unavailable")
        self.assert_no_signal(result["missing"], "Unavailable")
        self.assertEqual(result["good"]["score"], 50)
        self.assertEqual(self.get.call_count, 1)
        self.sleep.assert_not_called()
        self.assertEqual(json.loads(self.cache_file.read_text()), result)


if __name__ == "__main__":
    unittest.main()
