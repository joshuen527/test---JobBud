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

from ai_explainer import explain_recommendation, fallback_explanation
from career_engine import (
    CANDIDATE_ROLES,
    QUESTIONNAIRE_KNOWLEDGE,
    QUESTIONNAIRE_SKILLS,
    calculate_match,
    get_available_technologies,
    get_top_matches,
)
from reskill_engine import build_learning_path
from scenario_engine import build_scenario_comparison
from sdg_engine import get_match_sdg_impact, load_sdg_mapping
from trend_engine import (
    CACHE_HOURS,
    HIMALAYAS_SEARCH_QUERIES,
    HIMALAYAS_SEARCH_URL,
    REMOTIVE_API_URL,
    analyse_role_trends,
    analyse_skill_radar,
    apply_trends_to_matches,
    fetch_job_data,
    get_time_bucket,
)


class CareerEngineTests(unittest.TestCase):
    def make_profile(self, level):
        return {
            "skills": {skill: level for skill in QUESTIONNAIRE_SKILLS},
            "knowledge": {topic: level for topic in QUESTIONNAIRE_KNOWLEDGE},
            "technologies": [],
        }

    def test_all_candidate_roles_score_from_supplied_onet_data(self):
        matches = get_top_matches(self.make_profile(3), top_n=11)
        self.assertEqual(len(matches), len(CANDIDATE_ROLES))
        self.assertEqual({match["role"] for match in matches}, set(CANDIDATE_ROLES))
        self.assertTrue(
            all(0 <= match["overall"] <= 100 for match in matches)
        )
        self.assertTrue(
            all(0 <= match["confidence"] <= 100 for match in matches)
        )

    def test_unassessed_components_are_omitted_not_scored_as_zero(self):
        role = "Data Scientists"
        profile = {"skills": {skill: 5 for skill in QUESTIONNAIRE_SKILLS}}
        result = calculate_match(role, profile)
        self.assertIsNone(result["knowledge_alignment"])
        self.assertIsNone(result["technology_alignment"])
        self.assertAlmostEqual(result["overall"], 100.0)

    def test_profile_answers_change_the_match_score(self):
        role = "Data Scientists"
        high = calculate_match(role, self.make_profile(5))
        low = calculate_match(role, self.make_profile(1))
        self.assertNotEqual(high["overall"], low["overall"])
        self.assertNotEqual(
            high["component_scores"]["essential"],
            low["component_scores"]["essential"],
        )

    def test_technology_catalog_is_based_on_supplied_onet_data(self):
        catalog = set(get_available_technologies())
        self.assertTrue(catalog)
        for known_example in (
            "Python",
            "Microsoft Excel",
            "Microsoft Power BI",
            "Tableau",
        ):
            self.assertIn(known_example, catalog)


class TrendEngineTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 21, 12, tzinfo=timezone.utc)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cache_file = Path(directory.name) / "cache.json"
        # Any accidental use of the default session must fail, not reach the web.
        network = patch(
            "requests.sessions.Session.request",
            side_effect=AssertionError("Unexpected live network request"),
        )
        self.network = network.start()
        self.addCleanup(network.stop)
        self.addCleanup(self.network.assert_not_called)

    def make_session(self, himalayas=None, remotive=None):
        def get(url, **kwargs):
            if url == HIMALAYAS_SEARCH_URL:
                payload = himalayas
            elif url == REMOTIVE_API_URL:
                payload = remotive
            else:
                raise AssertionError(f"Unexpected API: {url}")
            if isinstance(payload, Exception):
                raise payload
            response = Mock()
            response.json.return_value = payload if payload is not None else {"jobs": []}
            return response

        return Mock(get=Mock(side_effect=get))

    def fetch(self, session, **kwargs):
        return fetch_job_data(
            session=session, cache_file=self.cache_file,
            now=kwargs.pop("now", self.now), **kwargs,
        )

    def himalayas_job(self):
        return {
            "guid": "himalayas-1",
            "applicationLink": "https://example.com/jobs/1",
            "title": "Data Scientist",
            "companyName": "Example",
            "pubDate": int((self.now - timedelta(days=2)).timestamp()),
            "categories": ["Data Science"],
            "parentCategories": ["Technology"],
            "excerpt": "Python",
            "description": "SQL",
        }

    def make_jobs(self, count, days_ago, title="Data Scientist"):
        publication_date = (
            self.now - timedelta(days=days_ago)
        ).isoformat()
        return [
            {
                "id": f"{title}-{days_ago}-{index}",
                "title": title,
                "description": "",
                "publication_date": publication_date,
            }
            for index in range(count)
        ]

    def test_low_sample_is_insufficient_and_does_not_change_match(self):
        trends = analyse_role_trends(self.make_jobs(1, 2), now=self.now)
        signal = trends["Data Scientists"]
        self.assertEqual(signal["direction"], "Insufficient signal")
        self.assertIsNone(signal["trend_score"])
        adjusted = apply_trends_to_matches(
            [{"role": "Data Scientists", "overall": 72.0}],
            trends,
        )[0]
        self.assertEqual(adjusted["future_match"], 72.0)
        self.assertFalse(adjusted["trend_adjusted"])

    def test_sufficient_sample_produces_real_counted_signal(self):
        jobs = self.make_jobs(6, 2) + self.make_jobs(2, 20)
        trend = analyse_role_trends(jobs, now=self.now)["Data Scientists"]
        self.assertEqual(trend["recent_signal"], 6)
        self.assertEqual(trend["previous_signal"], 2)
        self.assertEqual(trend["sample_count"], 8)
        self.assertEqual(trend["direction"], "Strong current signal")
        self.assertEqual(trend["trend_score"], 100)
        self.assertIsNone(trend["momentum"])

    def test_previous_evidence_does_not_satisfy_current_threshold(self):
        for recent in (0, 5, 6):
            with self.subTest(recent=recent):
                jobs = self.make_jobs(recent, 2) + self.make_jobs(30, 20)
                trend = analyse_role_trends(jobs, now=self.now)["Data Scientists"]
                self.assertEqual(trend["sample_count"], recent + 30)
                self.assertIsNone(trend["momentum"])
                self.assertEqual(
                    trend["direction"],
                    "Strong current signal" if recent == 6 else "Insufficient signal",
                )
                self.assertEqual(trend["trend_score"], 100 if recent == 6 else None)

    def test_current_demand_is_relative_and_adjusts_matches_without_mutation(self):
        jobs = (
            self.make_jobs(20, 2)
            + self.make_jobs(12, 2, "Accountant")
            + self.make_jobs(6, 2, "Actuary")
            + self.make_jobs(6, 2, "Economist")
        )
        # 6 / 21 is below the moderate threshold; 6 / 20 is exactly 30%.
        for leader_count, expected in ((20, "Moderate current signal"),
                                       (21, "Limited current signal")):
            with self.subTest(leader_count=leader_count):
                sample = jobs + self.make_jobs(leader_count - 20, 3)
                trends = analyse_role_trends(sample, now=self.now)
                self.assertEqual(trends["Actuaries"]["direction"], expected)
                self.assertEqual(
                    trends["Accountants and Auditors"]["direction"],
                    "Strong current signal" if leader_count == 20 else "Moderate current signal",
                )
                base = [{"role": "Actuaries", "overall": 80.0},
                        {"role": "Data Scientists", "overall": 75.0}]
                adjusted = apply_trends_to_matches(base, trends)
                self.assertEqual(adjusted[0]["role"], "Data Scientists")
                self.assertAlmostEqual(adjusted[1]["future_match"], 64 + (6 / leader_count * 100) * .2)
                self.assertTrue(all(row["trend_adjusted"] for row in adjusted))
                self.assertNotIn("future_match", base[0])

    def test_empty_and_unmatched_samples_leave_scores_unchanged(self):
        for jobs in ([], self.make_jobs(10, 2, "Unrelated role")):
            with self.subTest(jobs=jobs):
                trends = analyse_role_trends(jobs, now=self.now)
                self.assertTrue(all(t["trend_score"] is None for t in trends.values()))
                self.assertTrue(all(t["demand_index"] == 0 for t in trends.values()))
                adjusted = apply_trends_to_matches(
                    [{"role": "Data Scientists", "overall": 72}], trends,
                )[0]
                self.assertEqual(adjusted["future_match"], 72)
                self.assertFalse(adjusted["trend_adjusted"])

    def test_skill_radar_uses_current_evidence_not_temporal_growth(self):
        jobs = self.make_jobs(6, 2, "Python SQL") + self.make_jobs(20, 20, "Python")
        jobs += self.make_jobs(5, 2, "Tableau")
        radar = {row["skill"]: row for row in analyse_skill_radar(jobs, now=self.now)}
        self.assertEqual(radar["Python"]["recent"], 6)
        self.assertEqual(radar["Python"]["previous"], 20)
        self.assertEqual(radar["Python"]["demand_index"], 100)
        self.assertEqual(radar["Python"]["direction"], "Strong current signal")
        self.assertEqual(radar["Tableau"]["direction"], "Insufficient signal")
        self.assertTrue(all(row["change"] is None for row in radar.values()))

    def test_time_bucket_boundaries_and_invalid_dates(self):
        for days, expected in ((0, "recent"), (13, "recent"), (14, "previous"),
                               (27, "previous"), (28, "older"), (-1, None)):
            with self.subTest(days=days):
                self.assertEqual(
                    get_time_bucket((self.now - timedelta(days=days)).isoformat(), self.now),
                    expected,
                )
        for value in (None, "", "not-a-date"):
            self.assertIsNone(get_time_bucket(value, self.now))

    def test_fetch_queries_both_sources_normalizes_deduplicates_and_caches(self):
        session = self.make_session(
            {"jobs": [self.himalayas_job()]},
            {"jobs": self.make_jobs(1, 3)},
        )
        first = self.fetch(session)
        expected_calls = [
            call(HIMALAYAS_SEARCH_URL, params={"q": query, "sort": "recent", "page": 1},
                 timeout=20, headers={"User-Agent": "AI-Career-Navigator/1.0"})
            for query in HIMALAYAS_SEARCH_QUERIES
        ] + [call(REMOTIVE_API_URL, timeout=20,
                  headers={"User-Agent": "AI-Career-Navigator/1.0"})]
        self.assertEqual(session.get.call_args_list, expected_calls)
        self.assertEqual(len(first["jobs"]), 2)
        job = first["jobs"][0]
        self.assertEqual(job["id"], "himalayas-1")
        self.assertEqual(job["url"], "https://example.com/jobs/1")
        self.assertEqual(job["company_name"], "Example")
        self.assertEqual(job["publication_date"], (self.now - timedelta(days=2)).isoformat())
        self.assertEqual(job["category"], "Data Science Technology")
        self.assertEqual(job["description"], "Python SQL")
        self.assertEqual(job["source"], "Himalayas Remote Jobs API")
        self.assertEqual(first["jobs"][1]["source"], "Remotive Public Jobs API")
        self.assertEqual(first["source"], "Himalayas Remote Jobs API + Remotive Public Jobs API")
        self.assertFalse(first["from_cache"])
        self.assertFalse(first["stale"])
        self.assertIsNone(first["error"])
        self.assertEqual(first["listings_analysed"], 2)
        self.assertEqual(first["api_calls_today"], len(expected_calls))
        self.assertEqual(first["next_refresh_at"], (self.now + timedelta(hours=24)).isoformat())
        saved = json.loads(self.cache_file.read_text())
        self.assertEqual(saved["jobs"], first["jobs"])
        second = self.fetch(session, now=self.now + timedelta(hours=23, minutes=59))
        self.assertTrue(second["from_cache"])
        self.assertFalse(second["stale"])
        self.assertEqual(second["jobs"], first["jobs"])
        self.assertEqual(session.get.call_args_list, expected_calls)

    def test_expired_cache_and_force_refresh_fetch_again(self):
        self.assertEqual(CACHE_HOURS, 24)
        for force, hours in ((False, 24), (True, 1)):
            with self.subTest(force=force):
                session = self.make_session(remotive={"jobs": self.make_jobs(1, 2)})
                self.fetch(session, force_refresh=True)
                session.get.reset_mock()
                result = self.fetch(session, force_refresh=force,
                                    now=self.now + timedelta(hours=hours))
                self.assertFalse(result["from_cache"])
                self.assertEqual(session.get.call_count, len(HIMALAYAS_SEARCH_QUERIES) + 1)

    def test_remotive_fallback_when_himalayas_fails(self):
        session = self.make_session(
            requests.Timeout("offline"), {"jobs": self.make_jobs(1, 2)},
        )
        result = self.fetch(session)
        self.assertEqual(result["source"], "Remotive Public Jobs API")
        self.assertEqual(len(result["jobs"]), 1)
        self.assertFalse(result["stale"])
        self.assertIsNone(result["error"])

    def test_himalayas_success_survives_remotive_failure(self):
        result = self.fetch(self.make_session(
            {"jobs": [self.himalayas_job()]}, requests.ConnectionError("offline"),
        ))
        self.assertEqual(result["source"], "Himalayas Remote Jobs API")
        self.assertFalse(result["stale"])
        self.assertEqual(len(result["jobs"]), 1)

    def test_partial_himalayas_failure_reports_warning_and_keeps_successes(self):
        session = self.make_session({"jobs": [self.himalayas_job()]})
        normal_get = session.get.side_effect

        def get(url, **kwargs):
            response = normal_get(url, **kwargs)
            if kwargs.get("params", {}).get("q") == HIMALAYAS_SEARCH_QUERIES[0]:
                response.raise_for_status.side_effect = requests.HTTPError("503")
            return response

        session.get.side_effect = get
        result = self.fetch(session)
        self.assertEqual(len(result["jobs"]), 1)
        self.assertFalse(result["stale"])
        self.assertIn("Some Himalayas role-family searches were unavailable", result["error"])
        saved = json.loads(self.cache_file.read_text())
        self.assertEqual(saved["himalayas_query_errors"],
                         [f"{HIMALAYAS_SEARCH_QUERIES[0]}: HTTPError"])

    def test_unavailable_sources_return_explicit_empty_result_without_cache(self):
        for payload in (requests.Timeout("offline"), {"jobs": []},
                        {"jobs": "invalid"}, ["invalid root"]):
            with self.subTest(payload=payload):
                result = self.fetch(self.make_session(payload, payload))
                self.assertEqual(result["jobs"], [])
                self.assertEqual(result["listings_analysed"], 0)
                self.assertFalse(result["from_cache"])
                self.assertTrue(result["stale"])
                self.assertIsNotNone(result["error"])
                self.assertIsNone(result["fetched_at"])
                self.assertIsNone(result["next_refresh_at"])
                self.assertFalse(self.cache_file.exists())

    def test_failed_refresh_preserves_stale_cache(self):
        first = self.fetch(self.make_session(remotive={"jobs": self.make_jobs(1, 2)}))
        original_cache = self.cache_file.read_bytes()
        result = self.fetch(
            self.make_session(requests.Timeout(), requests.Timeout()),
            now=self.now + timedelta(hours=25),
        )
        self.assertTrue(result["from_cache"])
        self.assertTrue(result["stale"])
        self.assertEqual(result["jobs"], first["jobs"])
        self.assertEqual(result["fetched_at"], first["fetched_at"])
        self.assertIn("using the latest cached sample", result["error"])
        self.assertEqual(self.cache_file.read_bytes(), original_cache)

    def test_corrupt_or_invalid_cache_is_refetched(self):
        for content in ("not json", "[]", '{"jobs": {}}'):
            with self.subTest(content=content):
                self.cache_file.write_text(content)
                result = self.fetch(self.make_session(remotive={"jobs": self.make_jobs(1, 2)}))
                self.assertFalse(result["from_cache"])
                self.assertEqual(json.loads(self.cache_file.read_text())["jobs"], result["jobs"])

    def test_old_listings_are_filtered_and_unknown_dates_preserved(self):
        rows = self.make_jobs(1, 36) + self.make_jobs(1, 35) + self.make_jobs(1, 2)
        rows.append({"id": "undated", "title": "Data Scientist"})
        result = self.fetch(self.make_session(remotive={"jobs": rows + [None]}))
        self.assertEqual([job["id"] for job in result["jobs"]],
                         [rows[2]["id"], rows[1]["id"], "undated"])


class RecommendationsTests(unittest.TestCase):
    def test_scenarios_renormalize_when_trend_is_missing(self):
        rows = build_scenario_comparison(
            [
                {
                    "role": "Data Scientists",
                    "overall": 80.0,
                    "base_match": 80.0,
                    "trend_score": None,
                    "trend_adjusted": False,
                    "ai_technology_readiness": None,
                }
            ]
        )
        self.assertEqual(rows[0]["moderate_ai_disruption"], 80.0)
        self.assertEqual(rows[0]["fast_ai_disruption"], 80.0)

    def test_scenario_uses_available_ai_readiness_without_missing_trend_zero(self):
        rows = build_scenario_comparison(
            [
                {
                    "role": "Data Scientists",
                    "overall": 80.0,
                    "base_match": 80.0,
                    "trend_score": None,
                    "trend_adjusted": False,
                    "ai_technology_readiness": 100.0,
                }
            ]
        )
        self.assertGreater(rows[0]["fast_ai_disruption"], 80.0)

    def test_learning_path_contains_only_actual_gaps_and_real_mappings(self):
        match = {
            "role": "Data Scientists",
            "gaps": [
                {
                    "skill": "Python",
                    "group": "Technology",
                    "user": 0,
                    "target": 1,
                    "gap": 1,
                    "importance": 2,
                    "priority": 2,
                },
                {
                    "skill": "Unmapped O*NET skill",
                    "group": "Essential",
                    "user": 1,
                    "target": 3,
                    "gap": 2,
                    "importance": 4,
                    "priority": 8,
                },
            ],
        }
        path = build_learning_path(match)
        self.assertEqual(len(path), 2)
        self.assertEqual(path[0]["skill"], "Unmapped O*NET skill")
        self.assertEqual(path[0]["resource_title"], "Learning resource not yet mapped")
        self.assertEqual(path[1]["skill"], "Python")
        self.assertIsNotNone(path[1]["source_url"])

    def test_every_role_and_major_skill_group_has_an_sdg_mapping(self):
        mapping = load_sdg_mapping()
        self.assertEqual(set(mapping["careers"]), set(CANDIDATE_ROLES))
        self.assertTrue(
            all(
                mapping["careers"][role]
                for role in CANDIDATE_ROLES
            )
        )
        for category in ("essential", "transferable", "knowledge", "technology"):
            self.assertTrue(mapping["skill_categories"][category])
        self.assertTrue(
            get_match_sdg_impact({"role": "Data Scientists"})["career_mappings"]
        )

    def test_runpod_missing_configuration_uses_rule_based_fallback(self):
        match = {
            "role": "Data Scientists",
            "overall": 70.0,
            "trend_direction": "Insufficient signal",
            "strengths": [],
            "gaps": [],
        }
        with patch.dict(os.environ, {}, clear=True):
            result = explain_recommendation({}, match)
        self.assertEqual(result["source"], "Rule-based fallback")
        self.assertIn("Insufficient signal", result["text"])

    def test_technology_self_report_is_not_formatted_as_a_skill_rating(self):
        match = {
            "role": "Data Scientists",
            "trend_direction": "Insufficient signal",
            "strengths": [],
            "gaps": [
                {
                    "skill": "Python",
                    "group": "Technology",
                    "user": 0,
                    "target": 1,
                    "gap": 1,
                }
            ],
        }
        text = fallback_explanation(match)
        self.assertIn("No O*NET-listed workplace software examples were selected", text)
        self.assertNotIn("Python (you 0.0/5", text)


if __name__ == "__main__":
    unittest.main()
