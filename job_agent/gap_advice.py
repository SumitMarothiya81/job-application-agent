"""Honest skill-gap advice: how to close gaps, never pretend they're closed."""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from job_agent.jd_parser import normalize_skill
from job_agent.llm import invoke_with_fallback
from job_agent.matching import MatchReport
from job_agent.schemas import JobDescription

GAP_ADVICE_SYSTEM = """You are a career coach for a fresher applying to a job.
For each skill in 'skills to cover', give practical, honest advice.

Rules:
- Never say or imply the candidate already has these skills.
- Cover ONLY the skills listed under 'skills to cover'. Do not add others.
- how_to_close: 1-2 sentences on the fastest credible way to learn it.
- project_idea: one small, concrete project (1-2 sentences) that would show the skill
  and could extend one of the candidate's existing projects.
- Free resources and a realistic scope are better than ambitious plans.
"""


class _LLMGapAdvice(BaseModel):
    skill: str
    how_to_close: str = Field(description="1-2 sentences")
    project_idea: str = Field(description="1-2 sentences")


class _LLMGapPlan(BaseModel):
    items: list[_LLMGapAdvice]


class GapAdvice(_LLMGapAdvice):
    priority: str = "medium"  # set in code: required -> high, preferred -> medium


class GapPlan(BaseModel):
    items: list[GapAdvice] = Field(default_factory=list)


def advise_gaps(report: MatchReport, jd: JobDescription, known_skills: list[str],
                max_items: int = 5) -> GapPlan:
    targets = (report.missing_required + report.missing_preferred)[:max_items]
    if not targets:
        return GapPlan()  # nothing to advise: no API call

    request = (
        f"Role: {jd.job_title} ({jd.summary})\n"
        f"Candidate's current skills: {', '.join(known_skills) or 'none listed'}\n"
        f"Skills to cover: {', '.join(targets)}"
    )
    plan = invoke_with_fallback(
        [SystemMessage(content=GAP_ADVICE_SYSTEM), HumanMessage(content=request)],
        schema=_LLMGapPlan,
    )

    # Guard: keep only skills we asked about, one entry each, in our order.
    by_skill = {normalize_skill(item.skill): item for item in plan.items}
    items: list[GapAdvice] = []
    for skill in targets:
        item = by_skill.get(normalize_skill(skill))
        if item:
            priority = "high" if skill in report.missing_required else "medium"
            items.append(GapAdvice(skill=skill, how_to_close=item.how_to_close,
                                   project_idea=item.project_idea, priority=priority))
    return GapPlan(items=items)
