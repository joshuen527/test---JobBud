"""Explainable O*NET-based matching for the AI Career Navigator."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

CANDIDATE_ROLES = [
    "Business Intelligence Analysts",
    "Data Scientists",
    "Statisticians",
    "Actuaries",
    "Economists",
    "Market Research Analysts and Marketing Specialists",
    "Accountants and Auditors",
    "Management Analysts",
    "Financial Quantitative Analysts",
    "Loan Officers",
    "Financial Examiners",
]

QUESTIONNAIRE_SKILLS = [
    "Reading Comprehension",
    "Active Listening",
    "Speaking",
    "Critical Thinking",
    "Active Learning",
    "Writing",
    "Mathematics",
    "Monitoring",
    "Learning Strategies",
    "Science",
    "Complex Problem Solving",
    "Judgment and Decision Making",
    "Systems Analysis",
    "Systems Evaluation",
    "Time Management",
    "Coordination",
    "Social Perceptiveness",
    "Instructing",
]

QUESTIONNAIRE_KNOWLEDGE = [
    "Mathematics",
    "Computers and Electronics",
    "Economics and Accounting",
    "Administration and Management",
    "Sales and Marketing",
    "English Language",
    "Customer and Personal Service",
]

# The weights are a prototype team-defined formula, not an O*NET formula.
COMPONENT_WEIGHTS = {
    "essential": 0.35,
    "transferable": 0.25,
    "knowledge": 0.25,
    "technology": 0.15,
}


# Context is used only as a small supporting signal.
# Skills / knowledge / technology remain the main drivers of the match.
BACKGROUND_ROLE_AFFINITY: dict[str, dict[str, float]] = {
    "Accounting": {
        "Accountants and Auditors": 1.0,
        "Financial Examiners": 0.7,
        "Management Analysts": 0.4,
        "Loan Officers": 0.4,
    },
    "Actuarial Science": {
        "Actuaries": 1.0,
        "Financial Quantitative Analysts": 0.8,
        "Statisticians": 0.7,
        "Data Scientists": 0.4,
        "Financial Examiners": 0.5,
    },
    "Business": {
        "Management Analysts": 1.0,
        "Market Research Analysts and Marketing Specialists": 0.8,
        "Business Intelligence Analysts": 0.6,
        "Loan Officers": 0.4,
    },
    "Economics": {
        "Economists": 1.0,
        "Market Research Analysts and Marketing Specialists": 0.7,
        "Financial Quantitative Analysts": 0.5,
        "Management Analysts": 0.5,
    },
    "Finance": {
        "Financial Quantitative Analysts": 1.0,
        "Financial Examiners": 0.9,
        "Loan Officers": 0.8,
        "Accountants and Auditors": 0.6,
        "Actuaries": 0.5,
    },
    "Financial Technology": {
        "Financial Quantitative Analysts": 1.0,
        "Business Intelligence Analysts": 0.8,
        "Data Scientists": 0.7,
        "Financial Examiners": 0.6,
    },
    "Marketing": {
        "Market Research Analysts and Marketing Specialists": 1.0,
        "Management Analysts": 0.6,
        "Business Intelligence Analysts": 0.4,
    },
    "Mathematics": {
        "Statisticians": 0.9,
        "Actuaries": 0.9,
        "Data Scientists": 0.8,
        "Financial Quantitative Analysts": 0.8,
        "Economists": 0.4,
    },
    "Statistics": {
        "Statisticians": 1.0,
        "Data Scientists": 0.9,
        "Actuaries": 0.7,
        "Market Research Analysts and Marketing Specialists": 0.5,
        "Business Intelligence Analysts": 0.5,
    },
    "Other": {},
}

INTEREST_ROLE_AFFINITY: dict[str, dict[str, float]] = {
    "Analytics": {
        "Business Intelligence Analysts": 1.0,
        "Data Scientists": 1.0,
        "Statisticians": 0.9,
        "Management Analysts": 0.7,
        "Market Research Analysts and Marketing Specialists": 0.7,
        "Financial Quantitative Analysts": 0.7,
        "Actuaries": 0.6,
    },
    "Banking": {
        "Loan Officers": 1.0,
        "Financial Examiners": 1.0,
        "Financial Quantitative Analysts": 0.8,
        "Accountants and Auditors": 0.5,
    },
    "Business": {
        "Management Analysts": 1.0,
        "Business Intelligence Analysts": 0.7,
        "Market Research Analysts and Marketing Specialists": 0.7,
        "Accountants and Auditors": 0.5,
    },
    "Data": {
        "Data Scientists": 1.0,
        "Business Intelligence Analysts": 1.0,
        "Statisticians": 0.8,
        "Market Research Analysts and Marketing Specialists": 0.5,
    },
    "Finance": {
        "Financial Quantitative Analysts": 1.0,
        "Financial Examiners": 1.0,
        "Loan Officers": 0.8,
        "Actuaries": 0.7,
        "Accountants and Auditors": 0.6,
        "Economists": 0.5,
    },
    "Marketing": {
        "Market Research Analysts and Marketing Specialists": 1.0,
        "Management Analysts": 0.5,
        "Business Intelligence Analysts": 0.4,
    },
    "Risk": {
        "Actuaries": 1.0,
        "Financial Examiners": 1.0,
        "Financial Quantitative Analysts": 0.8,
        "Loan Officers": 0.6,
    },
    "Technology": {
        "Data Scientists": 1.0,
        "Business Intelligence Analysts": 0.9,
        "Financial Quantitative Analysts": 0.6,
    },
}

BACKGROUND_BONUS_MAX = 4.0
INTEREST_BONUS_MAX = 1.0
LOW_CONFIDENCE_PENALTY_MAX = 6.0

DATA_FILES = {
    "occupations": "occupation_data.csv",
    "essential": "essential_skills.csv",
    "transferable": "transferable_skills.csv",
    "knowledge": "knowledge.csv",
    "software": "software_skills.csv",
}


@lru_cache(maxsize=1)
def load_data() -> dict[str, pd.DataFrame]:
    """Load the public O*NET tables once; absent files remain explicit/empty."""
    tables: dict[str, pd.DataFrame] = {}
    for name, filename in DATA_FILES.items():
        path = DATA_DIR / filename
        tables[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return tables


def get_data_health() -> list[dict[str, Any]]:
    """Return dataset availability and row counts for the UI."""
    tables = load_data()
    return [
        {
            "name": filename,
            "available": not tables[name].empty,
            "rows": len(tables[name]),
        }
        for name, filename in DATA_FILES.items()
    ]


@lru_cache(maxsize=1)
def get_onet_source_date() -> str:
    """Return the newest source-data date recorded in the supplied O*NET tables."""
    dates = []
    for table in load_data().values():
        if table.empty or "Date" not in table.columns:
            continue
        for value in table["Date"].dropna().astype(str).unique():
            parsed = pd.to_datetime(value, format="%m/%Y", errors="coerce")
            if pd.isna(parsed):
                parsed = pd.to_datetime(value, errors="coerce")
            if not pd.isna(parsed):
                dates.append(parsed)
    return max(dates).strftime("%b %Y") if dates else "Not listed in the supplied files"


def _role_rows(dataset: pd.DataFrame, role: str) -> pd.DataFrame:
    if dataset.empty or "Title" not in dataset.columns:
        return pd.DataFrame()

    role_occupations = load_data()["occupations"]
    if (
        "O*NET-SOC Code" in dataset.columns
        and not role_occupations.empty
        and {"Title", "O*NET-SOC Code"}.issubset(role_occupations.columns)
    ):
        codes = role_occupations.loc[
            role_occupations["Title"].eq(role), "O*NET-SOC Code"
        ].dropna().unique()
        if len(codes):
            return dataset[dataset["O*NET-SOC Code"].isin(codes)].copy()

    return dataset[dataset["Title"].eq(role)].copy()


def get_requirements(dataset: pd.DataFrame, role: str) -> pd.DataFrame:
    """Join importance and level by the O*NET element name for one occupation."""
    rows = _role_rows(dataset, role)
    required = {"Element Name", "Scale Name", "Data Value"}
    if rows.empty or not required.issubset(rows.columns):
        return pd.DataFrame(columns=["Element Name", "Importance", "Level"])

    values = rows[["Element Name", "Scale Name", "Data Value"]].copy()
    values["Data Value"] = pd.to_numeric(values["Data Value"], errors="coerce")
    values = values.dropna(subset=["Element Name", "Data Value"])

    importance = (
        values[values["Scale Name"].eq("Importance")]
        .groupby("Element Name", as_index=False)["Data Value"]
        .mean()
        .rename(columns={"Data Value": "Importance"})
    )
    levels = (
        values[values["Scale Name"].eq("Level")]
        .groupby("Element Name", as_index=False)["Data Value"]
        .mean()
        .rename(columns={"Data Value": "Level"})
    )
    return importance.merge(levels, on="Element Name", how="inner")


@lru_cache(maxsize=128)
def get_role_technology_requirements(role: str) -> list[dict[str, Any]]:
    """Return software examples and O*NET technology flags for a role."""
    rows = _role_rows(load_data()["software"], role)
    if rows.empty or "Workplace Example" not in rows.columns:
        return []

    for flag in ("Hot Technology", "In Demand"):
        if flag not in rows.columns:
            rows[flag] = "N"

    rows["Workplace Example"] = rows["Workplace Example"].astype(str).str.strip()
    rows = rows[rows["Workplace Example"].ne("") & rows["Workplace Example"].ne("nan")]
    grouped = rows.groupby("Workplace Example", as_index=False).agg(
        hot=("Hot Technology", lambda values: values.astype(str).str.upper().eq("Y").any()),
        in_demand=("In Demand", lambda values: values.astype(str).str.upper().eq("Y").any()),
    )
    requirements = []
    for row in grouped.to_dict("records"):
        weight = 1.0 + 0.5 * row["hot"] + 0.5 * row["in_demand"]
        requirements.append(
            {
                "skill": row["Workplace Example"],
                "importance": weight,
                "hot_technology": bool(row["hot"]),
                "in_demand": bool(row["in_demand"]),
            }
        )
    return requirements


@lru_cache(maxsize=1)
def get_available_technologies() -> tuple[str, ...]:
    """Technology choices drawn only from the candidate roles' O*NET rows."""
    software = load_data()["software"]
    if software.empty or "Workplace Example" not in software.columns:
        return ()

    candidate_rows = software[software["Title"].isin(CANDIDATE_ROLES)].copy()
    if candidate_rows.empty:
        return ()

    candidate_rows["Workplace Example"] = (
        candidate_rows["Workplace Example"].astype(str).str.strip()
    )
    candidate_rows = candidate_rows[
        candidate_rows["Workplace Example"].ne("")
        & candidate_rows["Workplace Example"].ne("nan")
    ]
    candidate_rows["Hot Technology"] = (
        candidate_rows.get("Hot Technology", "N").astype(str).str.upper().eq("Y")
    )
    candidate_rows["In Demand"] = (
        candidate_rows.get("In Demand", "N").astype(str).str.upper().eq("Y")
    )

    catalog = (
        candidate_rows.groupby("Workplace Example", as_index=False)
        .agg(
            role_count=("Title", "nunique"),
            hot=("Hot Technology", "max"),
            in_demand=("In Demand", "max"),
        )
        .sort_values(
            ["role_count", "hot", "in_demand", "Workplace Example"],
            ascending=[False, False, False, True],
        )
    )
    return tuple(catalog["Workplace Example"].tolist())


def _score_requirement_group(
    dataset: pd.DataFrame,
    role: str,
    answers: dict[str, Any],
    group_name: str,
) -> dict[str, Any]:
    requirements = get_requirements(dataset, role)
    if requirements.empty:
        return {
            "alignment": None,
            "readiness": None,
            "coverage": 0.0,
            "strengths": [],
            "gaps": [],
        }

    requirements["Importance"] = pd.to_numeric(
        requirements["Importance"], errors="coerce"
    )
    requirements["Level"] = pd.to_numeric(requirements["Level"], errors="coerce")
    requirements = requirements.dropna(subset=["Importance", "Level"])
    requirements = requirements[requirements["Importance"].gt(0)]
    if requirements.empty:
        return {
            "alignment": None,
            "readiness": None,
            "coverage": 0.0,
            "strengths": [],
            "gaps": [],
        }

    total_importance = float(requirements["Importance"].sum())
    assessed_importance = 0.0
    alignment_numerator = 0.0
    alignment_denominator = 0.0
    readiness_numerator = 0.0
    strengths: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []

    for row in requirements.to_dict("records"):
        skill = str(row["Element Name"])
        raw_answer = answers.get(skill)
        try:
            user_level = float(raw_answer)
        except (TypeError, ValueError):
            continue
        if not pd.notna(user_level):
            continue

        importance = float(row["Importance"])
        required_level = max(0.0, float(row["Level"]) / 7.0 * 5.0)
        assessed_importance += importance

        # Preserve the existing O*NET importance × target-level scoring method.
        alignment_weight = importance * max(required_level, 0.1)
        alignment_numerator += min(max(user_level, 0.0), 5.0) / 5.0 * alignment_weight
        alignment_denominator += alignment_weight

        readiness = (
            min(max(user_level, 0.0) / required_level, 1.0)
            if required_level > 0
            else 1.0
        )
        readiness_numerator += readiness * importance

        record = {
            "skill": skill,
            "user": user_level,
            "target": required_level,
            "importance": importance,
            "group": group_name,
        }
        if user_level >= required_level:
            strengths.append(record)
        elif required_level - user_level >= 0.1:
            record["gap"] = required_level - user_level
            record["priority"] = record["gap"] * importance
            gaps.append(record)

    coverage = (
        assessed_importance / total_importance * 100.0
        if total_importance > 0
        else 0.0
    )
    alignment = (
        alignment_numerator / alignment_denominator * 100.0
        if alignment_denominator > 0
        else None
    )
    readiness = (
        readiness_numerator / assessed_importance * 100.0
        if assessed_importance > 0
        else None
    )

    return {
        "alignment": alignment,
        "readiness": readiness,
        "coverage": coverage,
        "strengths": sorted(
            strengths, key=lambda item: item["importance"], reverse=True
        ),
        "gaps": sorted(
            gaps, key=lambda item: item["priority"], reverse=True
        ),
    }


def _score_technology_group(
    role: str,
    technologies: set[str],
    answered: bool,
) -> dict[str, Any]:
    requirements = get_role_technology_requirements(role)
    if not requirements or not answered:
        return {
            "alignment": None,
            "readiness": None,
            "coverage": 0.0,
            "strengths": [],
            "gaps": [],
        }

    total_weight = sum(item["importance"] for item in requirements)
    selected_weight = sum(
        item["importance"]
        for item in requirements
        if item["skill"] in technologies
    )
    strengths = []
    gaps = []
    for item in requirements:
        record = {
            "skill": item["skill"],
            "user": 1.0 if item["skill"] in technologies else 0.0,
            "target": 1.0,
            "importance": item["importance"],
            "group": "Technology",
        }
        if item["skill"] in technologies:
            strengths.append(record)
        else:
            record["gap"] = 1.0
            record["priority"] = item["importance"]
            gaps.append(record)

    score = selected_weight / total_weight * 100.0 if total_weight else None
    return {
        "alignment": score,
        "readiness": score,
        # The user explicitly confirms all unselected options as not yet used.
        "coverage": 100.0,
        "strengths": sorted(
            strengths, key=lambda item: item["importance"], reverse=True
        ),
        "gaps": sorted(gaps, key=lambda item: item["priority"], reverse=True),
    }


def _score_ai_technology_readiness(
    role: str,
    technologies: set[str],
    answered: bool,
) -> float | None:
    if not answered:
        return None
    markers = ("tensorflow", "sagemaker", "machine learning", "artificial intelligence")
    requirements = [
        item
        for item in get_role_technology_requirements(role)
        if any(marker in item["skill"].lower() for marker in markers)
    ]
    if not requirements:
        return None
    total = sum(item["importance"] for item in requirements)
    achieved = sum(
        item["importance"] for item in requirements if item["skill"] in technologies
    )
    return achieved / total * 100.0 if total else None


def _normalise_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """
    Normalise the full profile while preserving user context.

    Earlier versions discarded background, interests, experience and user type.
    Those values are now kept so they can be used as small supporting signals.
    """
    if any(key in profile for key in ("skills", "knowledge", "technologies")):
        return {
            "skills": profile.get("skills") or {},
            "knowledge": profile.get("knowledge") or {},
            "technologies": profile.get("technologies"),
            "background": str(profile.get("background") or "").strip(),
            "interests": list(profile.get("interests") or []),
            "experience": str(profile.get("experience") or "").strip(),
            "user_type": str(profile.get("user_type") or "").strip(),
        }

    # Backwards-compatible input: the old API accepted a plain skills mapping.
    return {
        "skills": profile,
        "knowledge": {},
        "technologies": None,
        "background": "",
        "interests": [],
        "experience": "",
        "user_type": "",
    }


def _context_bonus(role: str, profile: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """
    Return a small, transparent context bonus.

    Background contributes at most 4 points and interests at most 1 point.
    This helps distinguish otherwise-similar sparse CV profiles without
    allowing context to overpower assessed O*NET evidence.
    """
    background = profile.get("background", "")
    interests = profile.get("interests") or []

    background_affinity = BACKGROUND_ROLE_AFFINITY.get(
        background,
        {},
    ).get(role, 0.0)

    interest_affinities = [
        INTEREST_ROLE_AFFINITY.get(interest, {}).get(role, 0.0)
        for interest in interests
    ]

    # Use the strongest selected interest rather than stacking many interests.
    interest_affinity = max(interest_affinities, default=0.0)

    background_bonus = background_affinity * BACKGROUND_BONUS_MAX
    interest_bonus = interest_affinity * INTEREST_BONUS_MAX

    total = min(
        BACKGROUND_BONUS_MAX + INTEREST_BONUS_MAX,
        background_bonus + interest_bonus,
    )

    return total, {
        "background_bonus": background_bonus,
        "interest_bonus": interest_bonus,
    }


def _confidence_penalty(confidence: float) -> float:
    """
    Penalise very sparse assessments without treating low confidence as failure.

    Confidence >= 50% receives no penalty. Below 50%, the penalty increases
    gradually up to 6 points at 0% confidence.
    """
    confidence = max(0.0, min(100.0, float(confidence)))

    if confidence >= 50.0:
        return 0.0

    return (
        (50.0 - confidence)
        / 50.0
        * LOW_CONFIDENCE_PENALTY_MAX
    )


def calculate_match(role: str, user_skills: dict[str, Any]) -> dict[str, Any]:
    """Score one role; accepts the legacy skills mapping or a full profile."""
    profile = _normalise_profile(user_skills)
    tables = load_data()

    results = {
        "essential": _score_requirement_group(
            tables["essential"], role, profile["skills"], "Essential"
        ),
        "transferable": _score_requirement_group(
            tables["transferable"], role, profile["skills"], "Transferable"
        ),
        "knowledge": _score_requirement_group(
            tables["knowledge"], role, profile["knowledge"], "Knowledge"
        ),
        "technology": _score_technology_group(
            role,
            set(profile["technologies"] or []),
            profile["technologies"] is not None,
        ),
    }

    ai_technology_readiness = _score_ai_technology_readiness(
        role,
        set(profile["technologies"] or []),
        profile["technologies"] is not None,
    )

    available = [
        name
        for name, result in results.items()
        if result["alignment"] is not None
    ]

    available_weight = sum(
        COMPONENT_WEIGHTS[name]
        for name in available
    )

    raw_overall = (
        sum(
            float(results[name]["alignment"])
            * COMPONENT_WEIGHTS[name]
            for name in available
        )
        / available_weight
        if available_weight
        else None
    )

    confidence = (
        sum(
            results[name]["coverage"]
            * COMPONENT_WEIGHTS[name]
            for name in available
        )
        / available_weight
        if available_weight
        else 0.0
    )

    context_bonus, context_parts = _context_bonus(
        role,
        profile,
    )

    penalty = _confidence_penalty(
        confidence
    )

    overall = (
        max(
            0.0,
            min(
                100.0,
                float(raw_overall)
                + context_bonus
                - penalty,
            ),
        )
        if raw_overall is not None
        else None
    )

    strengths = [
        item
        for result in results.values()
        for item in result["strengths"]
    ]

    gaps = [
        item
        for result in results.values()
        for item in result["gaps"]
    ]

    strengths.sort(
        key=lambda item: item["importance"],
        reverse=True,
    )

    gaps.sort(
        key=lambda item: item["priority"],
        reverse=True,
    )

    occupation_rows = tables["occupations"]
    description = ""

    if not occupation_rows.empty:
        rows = occupation_rows[
            occupation_rows["Title"].eq(role)
        ]

        if (
            not rows.empty
            and "Description" in rows.columns
        ):
            description = str(
                rows.iloc[0]["Description"]
            )

    return {
        "role": role,
        "description": description,

        # overall is the evidence-aware score used for ranking.
        "overall": overall,

        # raw_overall is retained for transparency/debugging.
        "raw_overall": raw_overall,

        "confidence": confidence,

        "context_bonus": context_bonus,
        "background_bonus": context_parts["background_bonus"],
        "interest_bonus": context_parts["interest_bonus"],
        "confidence_penalty": penalty,

        "essential_alignment": results["essential"]["alignment"],
        "transferable_alignment": results["transferable"]["alignment"],
        "knowledge_alignment": results["knowledge"]["alignment"],
        "technology_alignment": results["technology"]["alignment"],

        "essential_readiness": results["essential"]["readiness"],
        "transferable_readiness": results["transferable"]["readiness"],
        "knowledge_readiness": results["knowledge"]["readiness"],
        "technology_readiness": results["technology"]["readiness"],

        "ai_technology_readiness": ai_technology_readiness,

        "essential_coverage": results["essential"]["coverage"],
        "transferable_coverage": results["transferable"]["coverage"],
        "knowledge_coverage": results["knowledge"]["coverage"],
        "technology_coverage": results["technology"]["coverage"],

        "component_scores": {
            name: result["alignment"]
            for name, result in results.items()
        },

        "component_coverages": {
            name: result["coverage"]
            for name, result in results.items()
        },

        "strengths": strengths,
        "gaps": gaps,
    }


def get_top_matches(
    user_skills: dict[str, Any],
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """
    Return the top matches.

    Ranking is primarily driven by the evidence-aware overall score.
    Confidence is used only as a tie-breaker, followed by the raw O*NET match.
    """
    results = [
        calculate_match(role, user_skills)
        for role in CANDIDATE_ROLES
    ]

    results.sort(
        key=lambda item: (
            item["overall"] is not None,
            item["overall"] if item["overall"] is not None else -1,
            item.get("confidence", 0.0),
            item.get("raw_overall")
            if item.get("raw_overall") is not None
            else -1,
        ),
        reverse=True,
    )

    return results[:top_n]


