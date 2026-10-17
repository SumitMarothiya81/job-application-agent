from job_agent.demo import build_store, load_demo_profile
from job_agent.matching import SkillStatus, match_resume_to_jd
from job_agent.schemas import JobDescription, Seniority


def test_demo_profile_loads_and_indexes():
    profile = load_demo_profile()
    store = build_store(profile)
    assert len(profile.projects) == 4 and len(store.chunks) == 11
    assert {"python", "sql", "power bi"} <= set(profile.known_skills())


def test_demo_profile_gives_expected_statuses():
    profile = load_demo_profile()
    jd = JobDescription(
        job_title="Intern", seniority=Seniority.INTERN, required_skills=["python", "pandas"],
        preferred_skills=["tableau", "docker"], responsibilities=[], keywords=[], summary="x",
    )
    report = match_resume_to_jd(profile, build_store(profile), jd, related_threshold=0.99)
    status = {m.skill: m.status for m in report.matches}
    assert status == {"python": SkillStatus.EXPLICIT, "pandas": SkillStatus.EXPLICIT,
                      "tableau": SkillStatus.RELATED, "docker": SkillStatus.MISSING}
    tableau = next(m for m in report.matches if m.skill == "tableau")
    assert "Power BI" in tableau.evidence[0].text  # the dashboard bullet, not the SQL one
