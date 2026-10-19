import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from job_agent import agent, llm
from job_agent.agent import (AgentDeps, EmailDraft, JudgedBullet, JudgeOutput, TailorOutput, TailoredBullet,
                             build_graph, package_markdown, pending_interrupt, select_source_bullets,
                             template_email)
from job_agent.demo import build_store, load_demo_profile
from job_agent.schemas import JobDescription, Seniority

JD = JobDescription(
    job_title="Data Science Intern", company="Acme", seniority=Seniority.INTERN,
    required_skills=["python", "sql", "pandas", "scikit-learn"], preferred_skills=["docker", "tableau"],
    responsibilities=[], keywords=["python"], summary="Build ML features and dashboards.")
JD_TEXT = "Data Science Intern at Acme. Python, SQL, pandas, scikit-learn. 6 month internship."

GOOD_EMAIL = EmailDraft(subject="Application: Data Science Intern",
                        body="Hi [Recruiter name],\n\nI'm applying for the Data Science Intern role. "
                             "I have worked with Python, SQL and pandas.\n\nThanks,\n[Your name]")


@pytest.fixture(scope="module")
def resume():
    profile = load_demo_profile()
    return profile, build_store(profile)


class Fake:
    """Scriptable stand-ins for the model calls, with call counters."""

    def __init__(self, rewrite=None, judge=None, emails=None):
        self.calls = {"parse": 0, "tailor": 0, "judge": 0, "email": 0}
        self.tailor_payloads, self.email_payloads = [], []
        self._rewrite = rewrite or (lambda bullet, call: "Key result: " + bullet["original"])
        self._judge = judge or (lambda item: JudgedBullet(id=item["id"], supported=True))
        self._emails = list(emails or [GOOD_EMAIL])

    def deps(self) -> AgentDeps:
        def parse(text):
            self.calls["parse"] += 1
            return JD

        def tailor(payload):
            self.calls["tailor"] += 1
            self.tailor_payloads.append(payload)
            return TailorOutput(items=[TailoredBullet(id=b["id"], rewritten=self._rewrite(b, self.calls["tailor"]))
                                       for b in payload["bullets"]])

        def judge(payload):
            self.calls["judge"] += 1
            return JudgeOutput(items=[v for v in map(self._judge, payload["items"]) if v is not None])

        def email(payload):
            self.calls["email"] += 1
            self.email_payloads.append(payload)
            return self._emails[min(self.calls["email"], len(self._emails)) - 1]

        return AgentDeps(parse_jd=parse, tailor=tailor, judge=judge, email=email)


def start(resume, fake, thread="t1"):
    profile, store = resume
    graph = build_graph(profile, store, fake.deps(), InMemorySaver())
    config = {"configurable": {"thread_id": thread}}
    graph.invoke({"jd_text": JD_TEXT, "log": []}, config)
    return graph, config


def resume_with(graph, config, value):
    graph.invoke(Command(resume=value), config)
    return pending_interrupt(graph, config)


# ---- selecting bullets -------------------------------------------------------------------
def test_select_source_bullets_picks_relevant_bullets_with_limits(resume):
    profile, store = resume
    sources = select_source_bullets(profile, store, JD)
    assert 1 <= len(sources) <= agent.MAX_SOURCE_BULLETS
    assert all(s["section"] in {"project", "experience"} for s in sources)
    assert max(sum(s["source"] == o["source"] for o in sources) for s in sources) <= agent.MAX_PER_SOURCE
    flight = [s for s in sources if "Flight Fare" in s["source"]]
    assert flight and {"python", "pandas", "scikit-learn"} <= set(flight[0]["mirror_skills"]) | {"pandas", "python", "scikit-learn"}
    assert all(m in JD.required_skills + JD.preferred_skills for s in sources for m in s["mirror_skills"])
    assert "docker" not in {m for s in sources for m in s["mirror_skills"]}      # never offered a skill you lack


# ---- the full happy path ---------------------------------------------------------------------
def test_full_flow_with_two_human_checkpoints(resume):
    fake = Fake()
    graph, config = start(resume, fake)

    review = pending_interrupt(graph, config)
    assert review["type"] == "review_bullets" and all(i["status"] == "pass" for i in review["items"])
    assert fake.calls == {"parse": 1, "tailor": 1, "judge": 1, "email": 0}

    ids = [i["id"] for i in review["items"]]
    email_step = resume_with(graph, config, {"accept": ids[:2]})
    assert email_step["type"] == "review_email" and email_step["origin"] == "ai" and email_step["issues"] == []
    assert fake.calls["parse"] == 1 and fake.calls["tailor"] == 1          # nothing before the pause re-ran

    resume_with(graph, config, {"approved": True, "subject": "Hello", "body": "Hi [Recruiter name], I use Python.\n[Your name]"})
    assert pending_interrupt(graph, config) is None
    package = graph.get_state(config).values["package"]
    assert [b["id"] for b in package["bullets"]] == ids[:2]
    assert package["email"]["subject"] == "Hello" and package["match_score"] > 0
    assert "Nothing here has been sent" in package["markdown"] and package["resume_version"].startswith("Tailored: Acme")
    assert fake.calls == {"parse": 1, "tailor": 1, "judge": 1, "email": 1}


def test_user_cannot_accept_unknown_failed_or_unchanged_rewrites(resume):
    def rewrite(bullet, call):
        return bullet["original"] if bullet["id"].endswith("b2") else "Key result: " + bullet["original"]

    graph, config = start(resume, Fake(rewrite=rewrite))
    items = pending_interrupt(graph, config)["items"]
    unchanged = [i["id"] for i in items if not i["changed"]]
    changed = [i["id"] for i in items if i["changed"]]
    resume_with(graph, config, {"accept": unchanged + changed[:1] + ["made-up-id"]})
    resume_with(graph, config, {"approved": False})
    package = graph.get_state(config).values["package"]
    assert [b["id"] for b in package["bullets"]] == changed[:1]
    assert package["email"] is None                                       # the email was skipped


# ---- honesty: retry, drop, judge -----------------------------------------------------------------
def test_invented_number_is_rejected_then_fixed_on_retry(resume):
    def rewrite(bullet, call):
        if bullet["id"] == "proj-1-b1" and call == 1:
            return "Built a regression model, reducing RMSE by 25%"          # invented metric
        return "Key result: " + bullet["original"]

    fake = Fake(rewrite=rewrite)
    graph, config = start(resume, fake)
    assert fake.calls["tailor"] == 2
    second = fake.tailor_payloads[1]["bullets"]
    assert [b["id"] for b in second] == ["proj-1-b1"]                        # only the failed bullet is redone
    assert any("adds numbers" in text for text in second[0]["feedback"])      # and it is told what was wrong
    assert fake.calls["judge"] == 2
    assert all(i["status"] == "pass" for i in pending_interrupt(graph, config)["items"])


def test_bullet_that_keeps_failing_is_dropped_and_cannot_be_accepted(resume):
    def rewrite(bullet, call):
        return "Built a Docker pipeline" if bullet["id"] == "proj-1-b1" else "Key result: " + bullet["original"]

    fake = Fake(rewrite=rewrite)
    graph, config = start(resume, fake)
    assert fake.calls["tailor"] == agent.MAX_TAILOR_ATTEMPTS
    items = {i["id"]: i for i in pending_interrupt(graph, config)["items"]}
    assert items["proj-1-b1"]["status"] == "fail"
    assert any("docker" in issue for issue in items["proj-1-b1"]["issues"])
    resume_with(graph, config, {"accept": list(items)})                       # tries to accept everything
    resume_with(graph, config, {"approved": False})
    assert "proj-1-b1" not in [b["id"] for b in graph.get_state(config).values["package"]["bullets"]]


def test_ai_reviewer_can_reject_and_missing_verdicts_fail_closed(resume):
    def judge(item):
        if item["id"] == "proj-1-b1":
            return JudgedBullet(id=item["id"], supported=False, issue="claims you led the project")
        if item["id"] == "proj-1-b2":
            return None                                                       # reviewer 'forgot' this one
        return JudgedBullet(id=item["id"], supported=True)

    graph, config = start(resume, Fake(judge=judge))
    items = {i["id"]: i for i in pending_interrupt(graph, config)["items"]}
    assert items["proj-1-b1"]["status"] == "fail" and "reviewer: claims you led the project" in items["proj-1-b1"]["issues"]
    assert items["proj-1-b2"]["status"] == "fail" and "rejected to be safe" in items["proj-1-b2"]["issues"][0]


def test_missing_rewrite_counts_as_failure(resume):
    def rewrite(bullet, call):
        return "" if bullet["id"] == "proj-1-b1" else "Key result: " + bullet["original"]

    graph, config = start(resume, Fake(rewrite=rewrite))
    item = next(i for i in pending_interrupt(graph, config)["items"] if i["id"] == "proj-1-b1")
    assert item["status"] == "fail" and item["issues"] == ["no rewrite was returned"]


def test_when_every_rewrite_fails_the_graph_goes_straight_to_the_email(resume):
    fake = Fake(rewrite=lambda bullet, call: "Used Kubernetes and Docker for this")
    graph, config = start(resume, fake)
    step = pending_interrupt(graph, config)
    assert step["type"] == "review_email"                                    # no bullet review was needed
    assert fake.calls["judge"] == 0                                           # nothing reached the AI reviewer
    resume_with(graph, config, {"approved": True})
    assert graph.get_state(config).values["package"]["bullets"] == []


# ---- the email -------------------------------------------------------------------------------------
def test_email_that_invents_skills_falls_back_to_the_safe_template(resume):
    bad = EmailDraft(subject="Application", body="Hi [Recruiter name], I built Docker and Kubernetes systems, "
                                                 "improving speed by 90%.\n[Your name]")
    fake = Fake(emails=[bad, bad])
    graph, config = start(resume, fake)
    step = resume_with(graph, config, {"accept": []})
    assert fake.calls["email"] == agent.MAX_EMAIL_ATTEMPTS
    assert fake.email_payloads[1]["feedback"] and any("docker" in text for text in fake.email_payloads[1]["feedback"])
    assert step["origin"] == "template" and step["issues"]
    assert "Docker" not in step["body"] and "[Recruiter name]" in step["body"] and "[Your name]" in step["body"]


def test_email_retry_can_succeed_and_edits_are_checked_but_not_blocked(resume):
    bad = EmailDraft(subject="Hi", body="I know Docker well.\n[Your name]")
    fake = Fake(emails=[bad, GOOD_EMAIL])
    graph, config = start(resume, fake)
    step = resume_with(graph, config, {"accept": []})
    assert step["origin"] == "ai" and step["issues"] == [] and fake.calls["email"] == 2

    resume_with(graph, config, {"approved": True, "subject": "S", "body": "I also use Kubernetes in 12 projects."})
    package = graph.get_state(config).values["package"]
    assert package["email"]["body"].endswith("12 projects.")                  # the user's words are kept
    assert any("kubernetes" in w.lower() for w in package["warnings"])        # but flagged
    assert any("12" in w for w in package["warnings"])


def test_email_fields_are_length_capped(resume):
    graph, config = start(resume, Fake())
    resume_with(graph, config, {"accept": []})
    resume_with(graph, config, {"approved": True, "subject": "S" * 500, "body": "b" * 10000})
    email = graph.get_state(config).values["package"]["email"]
    assert len(email["subject"]) == 150 and len(email["body"]) == 3000


# ---- plumbing ------------------------------------------------------------------------------------------
def test_visitor_api_key_reaches_the_model_calls_inside_the_graph(resume):
    seen = []

    def rewrite(bullet, call):
        seen.append(llm._api_key_override.get())
        return "Key result: " + bullet["original"]

    profile, store = resume
    graph = build_graph(profile, store, Fake(rewrite=rewrite).deps(), InMemorySaver())
    with llm.use_api_key("visitor-key"):
        graph.invoke({"jd_text": JD_TEXT, "log": []}, {"configurable": {"thread_id": "k"}})
    assert seen and set(seen) == {"visitor-key"}


def test_two_runs_do_not_mix(resume):
    profile, store = resume
    saver = InMemorySaver()
    graphs = {}
    for name in ("a", "b"):
        graphs[name] = build_graph(profile, store, Fake().deps(), saver)
        graphs[name].invoke({"jd_text": JD_TEXT + name, "log": []}, {"configurable": {"thread_id": name}})
    graphs["a"].invoke(Command(resume={"accept": []}), {"configurable": {"thread_id": "a"}})
    assert pending_interrupt(graphs["a"], {"configurable": {"thread_id": "a"}})["type"] == "review_email"
    assert pending_interrupt(graphs["b"], {"configurable": {"thread_id": "b"}})["type"] == "review_bullets"


def test_template_email_and_markdown_helpers():
    email = template_email({"job_title": "Intern", "company": None}, ["python", "sql"], ["Built a model."])
    assert "your company" in email["body"] and "python, sql" in email["body"] and email["origin"] == "template"
    package = {"job_title": "Intern", "company": "Acme", "match_score": 80, "verdict": "strong", "warnings": ["careful"],
               "bullets": [{"source": "Project: X", "original": "old", "rewritten": "new"}],
               "email": {"subject": "S", "body": "B"}}
    text = package_markdown(package)
    assert "- new" in text and "_was (Project: X):_ old" in text and "**Subject:** S" in text and "- careful" in text
    package.update(bullets=[], email=None)
    assert "No rewrites were accepted" in package_markdown(package) and "No email was kept" in package_markdown(package)
