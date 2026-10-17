import pytest

from job_agent.jd_parser import normalize_jd, normalize_list, normalize_skill, parse_jd
from job_agent.schemas import JobDescription, Seniority


def test_normalize_skill_lowercases_and_maps_aliases():
    assert normalize_skill("  Scikit  Learn ") == "scikit-learn"
    assert normalize_skill("GenAI") == "generative ai"
    assert normalize_skill("Python") == "python"


def test_normalize_list_removes_duplicates_keeps_order():
    assert normalize_list(["SQL", "sql", "Python", " "]) == ["sql", "python"]


def test_normalize_jd_required_wins_over_preferred():
    jd = JobDescription(
        job_title="Intern",
        seniority=Seniority.INTERN,
        required_skills=["Python", "SQL"],
        preferred_skills=["python", "Docker"],
        responsibilities=[],
        keywords=["ML"],
        summary="x",
    )
    out = normalize_jd(jd)
    assert out.preferred_skills == ["docker"]
    assert out.keywords == ["machine learning"]


def test_parse_jd_rejects_empty_text():
    with pytest.raises(ValueError):
        parse_jd("   ")
