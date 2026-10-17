import pytest

from job_agent import embeddings, gap_advice, matching
from job_agent.chunking import embedding_text, profile_to_chunks
from job_agent.matching import (
    SkillStatus,
    ats_keyword_coverage,
    fit_notes_for,
    match_resume_to_jd,
    overall_score,
    skill_in_known,
    verdict_for,
)
from job_agent.resume_schema import Education, Experience, Project, ResumeProfile
from job_agent.schemas import JobDescription, Seniority
from job_agent.vector_store import VectorStore


def make_profile() -> ResumeProfile:
    return ResumeProfile(
        skills=["Python", "SQL", "Power BI", "Machine Learning Fundamentals", "Git"],
        experience=[Experience(organization="Acme", role="Data Intern",
                               bullets=["Cleaned datasets with pandas"])],
        projects=[
            Project(name="Flight Fare Prediction", tech_stack=["Python", "scikit-learn", "Pandas"],
                    bullets=["Reduced RMSE by 18% with feature engineering"]),
            Project(name="Cricket Dashboard", tech_stack=["Power BI", "SQL"],
                    bullets=["Built an interactive Power BI dashboard for batting stats"]),
        ],
        education=[Education(institution="Example Institute", degree="B.Tech CSE")],
        certifications=["Python for Data Science"],
    )


def make_store(profile: ResumeProfile) -> VectorStore:
    chunks = profile_to_chunks(profile)
    vectors = embeddings.embed_passages([embedding_text(c) for c in chunks])
    return VectorStore(chunks, vectors, embeddings.embedding_id())


def make_jd(**overrides) -> JobDescription:
    base = dict(
        job_title="Data Science Intern", company="Acme", seniority=Seniority.INTERN,
        required_skills=["python", "sql", "machine learning"],
        preferred_skills=["tableau", "docker"],
        responsibilities=[], keywords=["python", "tableau", "docker", "dashboards"], summary="x",
    )
    base.update(overrides)
    return JobDescription(**base)


# ---- skill comparison -------------------------------------------------------
def test_skill_in_known_rules():
    known = ["python", "machine learning fundamentals", "javascript", "vector database"]
    assert skill_in_known("Python", known)
    assert skill_in_known("machine learning", known)       # phrase of a longer skill
    assert skill_in_known("vector databases", known)       # plural
    assert not skill_in_known("java", known)               # never match javascript
    assert not skill_in_known("sql", ["mysql"])            # single words match exactly
    assert not skill_in_known("", known)


# ---- per-skill status -------------------------------------------------------
def test_status_explicit_related_missing():
    profile = make_profile()
    report = match_resume_to_jd(profile, make_store(profile), make_jd(), related_threshold=0.99)
    status = {m.skill: m.status for m in report.matches}
    assert status["python"] == SkillStatus.EXPLICIT
    assert status["sql"] == SkillStatus.EXPLICIT
    assert status["machine learning"] == SkillStatus.EXPLICIT  # via 'machine learning fundamentals'
    assert status["tableau"] == SkillStatus.RELATED            # via taxonomy: power bi
    assert status["docker"] == SkillStatus.MISSING


def test_related_via_taxonomy_and_evidence():
    profile = make_profile()
    report = match_resume_to_jd(profile, make_store(profile), make_jd(), related_threshold=0.99)
    tableau = next(m for m in report.matches if m.skill == "tableau")
    assert tableau.related_via == "power bi"
    assert tableau.evidence and "Power BI" in tableau.evidence[0].text + tableau.evidence[0].source


def test_missing_skill_has_no_evidence():
    profile = make_profile()
    report = match_resume_to_jd(profile, make_store(profile), make_jd(), related_threshold=0.99)
    docker = next(m for m in report.matches if m.skill == "docker")
    assert docker.evidence == [] and docker.related_via is None


def test_explicit_evidence_prefers_project_bullet_that_mentions_skill():
    profile = make_profile()
    report = match_resume_to_jd(profile, make_store(profile), make_jd(), related_threshold=0.99)
    python = next(m for m in report.matches if m.skill == "python")
    assert python.evidence[0].section in {"project", "experience"}
    assert "python" in (python.evidence[0].source + python.evidence[0].text).lower()


def test_semantic_fallback_only_uses_bullet_sections():
    profile = make_profile()
    jd = make_jd(required_skills=["kubernetes"], preferred_skills=[])
    report = match_resume_to_jd(profile, make_store(profile), jd, related_threshold=-1.0)
    match = report.matches[0]
    assert match.status == SkillStatus.RELATED and match.related_via is None
    assert match.evidence
    assert all(e.section in {"experience", "project", "certification"} for e in match.evidence)


# ---- scoring ----------------------------------------------------------------
def test_overall_score_weights():
    assert overall_score(1.0, 0.0) == 70
    assert overall_score(1.0, 1.0) == 100
    assert overall_score(0.5, 0.5) == 50
    assert overall_score(0.8, None) == 80   # only required skills
    assert overall_score(None, 0.4) == 40   # only preferred skills
    assert overall_score(None, None) == 0


def test_verdict_thresholds():
    assert [verdict_for(s) for s in (90, 75, 60, 55, 40, 35, 10)] == [
        "strong", "strong", "good", "good", "stretch", "stretch", "weak"]


def test_report_scores_match_statuses():
    profile = make_profile()
    report = match_resume_to_jd(profile, make_store(profile), make_jd(), related_threshold=0.99)
    assert report.required_coverage == 1.0                 # 3/3 explicit
    assert report.preferred_coverage == pytest.approx(0.25)  # tableau related (0.5) + docker missing (0) / 2
    assert report.overall_score == round(100 * (0.7 * 1.0 + 0.3 * 0.25))
    assert report.strengths == ["python", "sql", "machine learning"]
    assert report.related_skills == ["tableau"] and report.missing_preferred == ["docker"]


# ---- ATS keywords + fit notes ----------------------------------------------
def test_ats_keyword_coverage():
    hits, missing = ats_keyword_coverage(make_profile(), make_jd())
    assert "python" in hits and "dashboards" in hits   # 'dashboard' appears in a bullet
    assert "docker" in missing and "tableau" in missing


def test_fit_notes():
    assert fit_notes_for(make_jd()) == []
    senior = make_jd(seniority=Seniority.SENIOR, min_years_experience=5)
    assert len(fit_notes_for(senior)) == 2
    assert fit_notes_for(make_jd(min_years_experience=1)) == []


# ---- gap advice ---------------------------------------------------------------
def test_no_gaps_means_no_api_call(monkeypatch):
    profile = make_profile()
    jd = make_jd(required_skills=["python"], preferred_skills=[])
    report = match_resume_to_jd(profile, make_store(profile), jd, related_threshold=0.99)
    monkeypatch.setattr(gap_advice, "invoke_with_fallback",
                        lambda *a, **k: pytest.fail("LLM must not be called"))
    assert gap_advice.advise_gaps(report, jd, profile.known_skills()).items == []


def test_gap_advice_keeps_only_requested_skills_and_sets_priority(monkeypatch):
    profile = make_profile()
    jd = make_jd(required_skills=["python", "kubernetes"], preferred_skills=["docker"])
    report = match_resume_to_jd(profile, make_store(profile), jd, related_threshold=0.99)

    llm_plan = gap_advice._LLMGapPlan(items=[
        gap_advice._LLMGapAdvice(skill="Docker", how_to_close="a", project_idea="b"),
        gap_advice._LLMGapAdvice(skill="Kubernetes", how_to_close="c", project_idea="d"),
        gap_advice._LLMGapAdvice(skill="Rust", how_to_close="e", project_idea="f"),  # not requested
    ])
    monkeypatch.setattr(gap_advice, "invoke_with_fallback", lambda *a, **k: llm_plan)

    plan = gap_advice.advise_gaps(report, jd, profile.known_skills())
    assert [i.skill for i in plan.items] == ["kubernetes", "docker"]  # our order, no 'rust'
    assert [i.priority for i in plan.items] == ["high", "medium"]
