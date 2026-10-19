"""The apply assistant: a LangGraph workflow with two human checkpoints.

    parse job -> match -> pick bullets -> tailor -> verify --(failed? retry once)--> review (YOU)
                                                                                         |
                                  finalize <- review email (YOU) <- draft email <--------+

Design rules worth knowing:
  * Nothing is ever submitted or sent. The output is text you copy.
  * A rewrite is shown to you only if it passed the deterministic checks (verify.py) AND an AI reviewer.
  * Nodes that call interrupt() are re-run from their first line when you resume, so they contain NO
    model calls. All Gemini calls live in nodes that run exactly once per step.
  * State holds plain dicts (not objects) so it checkpoints cleanly. Your resume profile and the
    vector store are closed over by the graph, not stored in it.
"""

import json
import operator
import re
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Callable, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from job_agent import embeddings
from job_agent.agent_prompts import EMAIL_SYSTEM, JUDGE_SYSTEM, TAILOR_SYSTEM
from job_agent.jd_parser import normalize_skill, parse_jd
from job_agent.llm import invoke_with_fallback
from job_agent.matching import _profile_text, _tokens, match_resume_to_jd
from job_agent.resume_schema import ResumeProfile
from job_agent.schemas import JobDescription
from job_agent.vector_store import VectorStore
from job_agent.verify import build_vocab, check_email, check_rewrite

MAX_TAILOR_ATTEMPTS = 2     # first try + one retry, to keep the number of Gemini calls down
MAX_EMAIL_ATTEMPTS = 2
MAX_SOURCE_BULLETS = 5
MAX_PER_SOURCE = 2


# --------------------------------------------------------------------------
# Model outputs and the pluggable model calls
# --------------------------------------------------------------------------
class TailoredBullet(BaseModel):
    id: str
    rewritten: str
    keywords_used: list[str] = Field(default_factory=list)


class TailorOutput(BaseModel):
    items: list[TailoredBullet]


class JudgedBullet(BaseModel):
    id: str
    supported: bool
    issue: str = ""


class JudgeOutput(BaseModel):
    items: list[JudgedBullet]


class EmailDraft(BaseModel):
    subject: str
    body: str


@dataclass
class AgentDeps:
    """The model calls, injected so tests can run the whole graph without Gemini."""
    parse_jd: Callable[[str], JobDescription]
    tailor: Callable[[dict], TailorOutput]
    judge: Callable[[dict], JudgeOutput] | None
    email: Callable[[dict], EmailDraft]


def _ask(system: str, payload: dict, schema):
    return invoke_with_fallback(
        [SystemMessage(content=system), HumanMessage(content=json.dumps(payload, indent=2, default=str))],
        schema=schema,
    )


def default_deps() -> AgentDeps:
    return AgentDeps(
        parse_jd=parse_jd,
        tailor=lambda payload: _ask(TAILOR_SYSTEM, payload, TailorOutput),
        judge=lambda payload: _ask(JUDGE_SYSTEM, payload, JudgeOutput),
        email=lambda payload: _ask(EMAIL_SYSTEM, payload, EmailDraft),
    )


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    jd_text: str
    jd: dict
    report: dict
    sources: list[dict]
    proposals: list[dict]
    attempt: int
    accepted: list[dict]
    email: dict | None
    email_issues: list[str]
    email_warnings: list[str]
    package: dict
    log: Annotated[list[str], operator.add]


# --------------------------------------------------------------------------
# Picking which of YOUR bullets to tailor (deterministic: no model involved)
# --------------------------------------------------------------------------
def _tech_for_chunk(profile: ResumeProfile, chunk_id: str) -> list[str]:
    match = re.match(r"^(exp|proj)-(\d+)-", chunk_id)
    if not match or match.group(1) == "exp":
        return []
    index = int(match.group(2)) - 1
    return list(profile.projects[index].tech_stack) if 0 <= index < len(profile.projects) else []


def select_source_bullets(profile: ResumeProfile, store: VectorStore, jd: JobDescription,
                          limit: int = MAX_SOURCE_BULLETS, per_source: int = MAX_PER_SOURCE) -> list[dict]:
    """The resume bullets most relevant to this job: skills they literally mention, plus semantic closeness."""
    weights = {**{s: 1.0 for s in jd.preferred_skills}, **{s: 2.0 for s in jd.required_skills}}
    similarity = {}
    if store.chunks:
        query = embeddings.embed_query(f"{jd.job_title}. Skills: {', '.join(jd.required_skills)}")
        similarity = dict(zip((c.id for c in store.chunks), (store.vectors @ query).tolist()))

    scored = []
    for chunk in store.chunks:
        if chunk.section not in ("experience", "project") or chunk.id.endswith("-tech"):
            continue
        tech = _tech_for_chunk(profile, chunk.id)
        tokens = _tokens(f"{chunk.source} {chunk.text}")
        skill_score = sum(w for skill, w in weights.items() if (t := _tokens(skill)) and t <= tokens)
        scored.append((skill_score + 2.0 * similarity.get(chunk.id, 0.0), chunk, tech))

    chosen, per_origin = [], {}
    for _, chunk, tech in sorted(scored, key=lambda item: item[0], reverse=True):
        if per_origin.get(chunk.source, 0) >= per_source:
            continue
        per_origin[chunk.source] = per_origin.get(chunk.source, 0) + 1
        bullet_tokens = _tokens(chunk.text) | {tok for term in tech for tok in _tokens(term)}
        mirror = [s for s in weights if (t := _tokens(s)) and t <= bullet_tokens]
        chosen.append({"id": chunk.id, "source": chunk.source.split(" (tech:")[0], "section": chunk.section,
                       "text": chunk.text, "tech_stack": [normalize_skill(t) for t in tech],
                       "mirror_skills": mirror})
        if len(chosen) >= limit:
            break
    return chosen


# --------------------------------------------------------------------------
# Deterministic fallback email (used when the model can't produce an honest one)
# --------------------------------------------------------------------------
def template_email(jd: dict, strengths: list[str], bullets: list[str]) -> dict:
    role, company = jd.get("job_title", "the role"), jd.get("company") or "your company"
    skills = ", ".join(strengths[:4])
    lines = ["Hi [Recruiter name],", "",
             f"I'm writing to apply for the {role} position at {company}."]
    if skills:
        lines.append(f"My background includes hands-on work with {skills}.")
    if bullets:
        lines.append(f"For example: {bullets[0].rstrip('.')}.")
    lines += ["", "I've attached my resume and would be glad to talk about how I can contribute.", "",
              "Thank you for your time,", "[Your name]"]
    return {"subject": f"Application for {role}", "body": "\n".join(lines), "origin": "template"}


def package_markdown(package: dict, today: date | None = None) -> str:
    today = today or date.today()
    lines = [f"# Application kit: {package['job_title']} @ {package.get('company') or 'unknown company'}",
             f"_Prepared {today:%d %b %Y}. Nothing here has been sent or submitted; you stay in control._", "",
             f"**Match score:** {package['match_score']}/100 ({package['verdict']})", "",
             "## Tailored bullets", ""]
    if package["bullets"]:
        for item in package["bullets"]:
            lines += [f"- {item['rewritten']}", f"  - _was ({item['source']}):_ {item['original']}"]
    else:
        lines.append("_No rewrites were accepted. Your original bullets stay as they are._")
    lines += ["", "## Email draft", ""]
    if package["email"]:
        lines += [f"**Subject:** {package['email']['subject']}", "", package["email"]["body"]]
    else:
        lines.append("_No email was kept._")
    if package["warnings"]:
        lines += ["", "## Things to double-check", ""] + [f"- {w}" for w in package["warnings"]]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# The graph
# --------------------------------------------------------------------------
def build_graph(profile: ResumeProfile, store: VectorStore, deps: AgentDeps, checkpointer):
    known = profile.known_skills()
    profile_text = _profile_text(profile)

    def parse_node(state: AgentState) -> dict:
        if state.get("jd"):
            return {"log": ["Used the job details already parsed."]}
        jd = deps.parse_jd(state["jd_text"])
        return {"jd": jd.model_dump(mode="json"),
                "log": [f"Read the job: {jd.job_title} at {jd.company or 'unknown company'}."]}

    def match_node(state: AgentState) -> dict:
        report = match_resume_to_jd(profile, store, JobDescription(**state["jd"]))
        return {"report": report.model_dump(mode="json"),
                "log": [f"Match score {report.overall_score}/100 ({report.verdict})."]}

    def select_node(state: AgentState) -> dict:
        sources = select_source_bullets(profile, store, JobDescription(**state["jd"]))
        return {"sources": sources, "proposals": [], "attempt": 0,
                "log": [f"Picked {len(sources)} of your bullets that are most relevant to this job."]}

    def tailor_node(state: AgentState) -> dict:
        proposals = {p["id"]: p for p in state.get("proposals", [])}
        targets = [s for s in state["sources"] if proposals.get(s["id"], {}).get("status") != "pass"]
        if not targets:
            return {"attempt": state.get("attempt", 0) + 1}
        jd = state["jd"]
        payload = {
            "job": {"title": jd["job_title"], "summary": jd["summary"],
                    "required_skills": jd["required_skills"], "preferred_skills": jd["preferred_skills"]},
            "bullets": [{"id": s["id"], "from": s["source"], "original": s["text"],
                         "mirror_skills": s["mirror_skills"],
                         "feedback": proposals.get(s["id"], {}).get("issues") or None} for s in targets],
        }
        returned = {item.id: item for item in deps.tailor(payload).items}
        for s in targets:
            item = returned.get(s["id"])
            proposals[s["id"]] = {"id": s["id"], "source": s["source"], "original": s["text"],
                                  "rewritten": item.rewritten.strip() if item else "",
                                  "status": "pending", "issues": [], "warnings": []}
        attempt = state.get("attempt", 0) + 1
        return {"proposals": [proposals[s["id"]] for s in state["sources"] if s["id"] in proposals],
                "attempt": attempt, "log": [f"Drafted rewrites for {len(targets)} bullet(s) (attempt {attempt})."]}

    def verify_node(state: AgentState) -> dict:
        jd = JobDescription(**state["jd"])
        vocab = build_vocab(profile, jd)
        by_id = {s["id"]: s for s in state["sources"]}
        proposals = [dict(p) for p in state["proposals"]]

        pending = [p for p in proposals if p["status"] == "pending"]
        for p in pending:
            source = by_id[p["id"]]
            if not p["rewritten"]:
                p["issues"] = ["no rewrite was returned"]
                continue
            check = check_rewrite(source["text"], source["tech_stack"], p["rewritten"], vocab)
            p["issues"], p["warnings"] = list(check.issues), list(check.warnings)

        # Second layer: an AI fact-checker for claims the word checks can't see (only for rewrites that
        # passed the cheap checks and actually changed something).
        to_judge = [p for p in pending if not p["issues"] and p["rewritten"] != p["original"]]
        if deps.judge and to_judge:
            verdicts = {v.id: v for v in deps.judge({"items": [
                {"id": p["id"], "original": p["original"], "rewrite": p["rewritten"]} for p in to_judge]}).items}
            for p in to_judge:
                verdict = verdicts.get(p["id"])
                if verdict is None:
                    p["issues"].append("the reviewer did not return a verdict (rejected to be safe)")
                elif not verdict.supported:
                    p["issues"].append("reviewer: " + (verdict.issue or "the rewrite adds claims not in the original"))

        for p in pending:
            p["status"] = "fail" if p["issues"] else "pass"
        passed = sum(p["status"] == "pass" for p in proposals)
        return {"proposals": proposals,
                "log": [f"Fact-checked the rewrites: {passed} passed, {len(proposals) - passed} rejected so far."]}

    def route_after_verify(state: AgentState) -> str:
        failed = any(p["status"] == "fail" for p in state["proposals"])
        return "tailor" if failed and state.get("attempt", 0) < MAX_TAILOR_ATTEMPTS else "review"

    def review_node(state: AgentState) -> dict:
        # NO model calls here: this node re-runs from its first line when you resume.
        proposals = state["proposals"]
        passing = [p for p in proposals if p["status"] == "pass"]
        if not any(p["rewritten"] != p["original"] for p in passing):
            return {"accepted": [], "log": ["No rewrite passed the fact-check; your original bullets stay as they are."]}
        reply = interrupt({"type": "review_bullets", "items": [
            {**p, "changed": p["rewritten"] != p["original"]} for p in proposals]})
        wanted = set((reply or {}).get("accept", []))
        accepted = [{"id": p["id"], "source": p["source"], "original": p["original"], "rewritten": p["rewritten"]}
                    for p in passing if p["id"] in wanted and p["rewritten"] != p["original"]]
        return {"accepted": accepted, "log": [f"You accepted {len(accepted)} rewrite(s)."]}

    def email_node(state: AgentState) -> dict:
        jd = state["jd"]
        accepted = state.get("accepted", [])
        facts_bullets = [a["rewritten"] for a in accepted] or [s["text"] for s in state["sources"][:2]]
        strengths = state["report"]["strengths"]
        payload = {"job": {"title": jd["job_title"], "company": jd.get("company")},
                   "facts": {"skills_that_match": strengths, "bullets": facts_bullets}}
        allowed_text = f"{profile_text} {state['jd_text']} {' '.join(facts_bullets)}"
        vocab = build_vocab(profile, JobDescription(**jd))

        issues: list[str] = []
        for _ in range(MAX_EMAIL_ATTEMPTS):
            draft = deps.email({**payload, "feedback": issues or None})
            issues = check_email(draft.subject, draft.body, allowed_text, known, vocab)
            if not issues:
                return {"email": {"subject": draft.subject.strip(), "body": draft.body.strip(), "origin": "ai"},
                        "email_issues": [], "log": ["Drafted a recruiter email and fact-checked it."]}
        return {"email": template_email(jd, strengths, facts_bullets), "email_issues": issues,
                "log": ["The AI email kept making claims not in your resume, so a safe template was used."]}

    def email_review_node(state: AgentState) -> dict:
        email = state["email"]
        reply = interrupt({"type": "review_email", "subject": email["subject"], "body": email["body"],
                           "origin": email["origin"], "issues": state.get("email_issues", [])})
        reply = reply or {}
        if not reply.get("approved"):
            return {"email": None, "email_warnings": [], "log": ["You skipped the email."]}
        subject, body = str(reply.get("subject", email["subject"]))[:150], str(reply.get("body", email["body"]))[:3000]
        jd = JobDescription(**state["jd"])
        warnings = check_email(subject, body, f"{profile_text} {state['jd_text']}", known, build_vocab(profile, jd))
        return {"email": {"subject": subject, "body": body, "origin": email["origin"]},
                "email_warnings": warnings, "log": ["You approved the email."]}

    def finalize_node(state: AgentState) -> dict:
        jd, report = state["jd"], state["report"]
        warnings = [f"Email: {w}" for w in state.get("email_warnings", [])]
        warnings += [f"{p['source']}: {w}" for p in state.get("proposals", []) for w in p.get("warnings", [])
                     if any(a["id"] == p["id"] for a in state.get("accepted", []))]
        package = {
            "job_title": jd["job_title"], "company": jd.get("company"),
            "match_score": report["overall_score"], "verdict": report["verdict"],
            "bullets": state.get("accepted", []), "email": state.get("email"), "warnings": warnings,
            "resume_version": f"Tailored: {jd.get('company') or 'job'} - {jd['job_title']}"[:60],
        }
        package["markdown"] = package_markdown(package)
        return {"package": package, "log": ["Prepared your application kit. Nothing was sent."]}

    graph = StateGraph(AgentState)
    for name, node in [("parse", parse_node), ("match", match_node), ("select", select_node),
                       ("tailor", tailor_node), ("verify", verify_node), ("review", review_node),
                       ("email", email_node), ("email_review", email_review_node), ("finalize", finalize_node)]:
        graph.add_node(name, node)
    graph.add_edge(START, "parse")
    for a, b in [("parse", "match"), ("match", "select"), ("select", "tailor"), ("tailor", "verify"),
                 ("review", "email"), ("email", "email_review"), ("email_review", "finalize"),
                 ("finalize", END)]:
        graph.add_edge(a, b)
    graph.add_conditional_edges("verify", route_after_verify, {"tailor": "tailor", "review": "review"})
    return graph.compile(checkpointer=checkpointer)


def pending_interrupt(graph, config: dict) -> dict | None:
    """The question the graph is currently waiting on (None when it has finished)."""
    snapshot = graph.get_state(config)
    if snapshot.next and snapshot.tasks and snapshot.tasks[0].interrupts:
        return snapshot.tasks[0].interrupts[0].value
    return None
