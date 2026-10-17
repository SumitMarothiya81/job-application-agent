"""Turn raw job-description text into a validated JobDescription."""

from langchain_core.messages import HumanMessage, SystemMessage

from job_agent.llm import invoke_with_fallback
from job_agent.prompts import JD_EXTRACTION_SYSTEM
from job_agent.schemas import JobDescription

# Different spellings of the same skill. We'll grow this over time.
SKILL_ALIASES = {
    "js": "javascript",
    "node": "node.js",
    "nodejs": "node.js",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "ml": "machine learning",
    "genai": "generative ai",
    "gen ai": "generative ai",
    "postgres": "postgresql",
}


def normalize_skill(skill: str) -> str:
    """lowercase, collapse spaces, map aliases to one canonical name."""
    cleaned = " ".join(skill.lower().strip().split())
    return SKILL_ALIASES.get(cleaned, cleaned)


def normalize_list(items: list[str]) -> list[str]:
    """Normalize every item, drop empties and duplicates, keep order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        norm = normalize_skill(item)
        if norm and norm not in seen:
            seen.add(norm)
            result.append(norm)
    return result


def normalize_jd(jd: JobDescription) -> JobDescription:
    """Clean the LLM output so later steps can compare skills reliably."""
    required = normalize_list(jd.required_skills)
    # A skill can't be both required and preferred: required wins.
    preferred = [s for s in normalize_list(jd.preferred_skills) if s not in required]
    return jd.model_copy(
        update={
            "required_skills": required,
            "preferred_skills": preferred,
            "keywords": normalize_list(jd.keywords),
        }
    )


def parse_jd(jd_text: str) -> JobDescription:
    """Extract structured data from a job description."""
    if not jd_text or not jd_text.strip():
        raise ValueError("Job description text is empty.")

    result = invoke_with_fallback(
        [
            SystemMessage(content=JD_EXTRACTION_SYSTEM),
            HumanMessage(content=f"Job description:\n\n{jd_text}"),
        ],
        schema=JobDescription,
    )
    return normalize_jd(result)
