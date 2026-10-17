"""Use Gemini to turn (redacted) resume text into a ResumeProfile."""

from langchain_core.messages import HumanMessage, SystemMessage

from job_agent.jd_parser import normalize_list
from job_agent.llm import invoke_with_fallback
from job_agent.prompts import RESUME_EXTRACTION_SYSTEM
from job_agent.resume_schema import ResumeProfile


def parse_resume(resume_text: str) -> ResumeProfile:
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is empty.")
    profile = invoke_with_fallback(
        [
            SystemMessage(content=RESUME_EXTRACTION_SYSTEM),
            HumanMessage(content=f"Resume:\n\n{resume_text}"),
        ],
        schema=ResumeProfile,
    )
    # Keep original wording of bullets; only clean the skill lists.
    for project in profile.projects:
        project.tech_stack = normalize_list(project.tech_stack)
    return profile.model_copy(update={"skills": normalize_list(profile.skills)})
