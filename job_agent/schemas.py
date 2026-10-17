"""The shape of the data we want from the LLM."""

from enum import Enum

from pydantic import BaseModel, Field


class Seniority(str, Enum):
    INTERN = "intern"
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    UNKNOWN = "unknown"


class JobDescription(BaseModel):
    """Structured version of a job description."""

    job_title: str = Field(description="Title of the role, e.g. 'Data Scientist Intern'")
    company: str | None = Field(default=None, description="Company name if stated")
    location: str | None = Field(default=None, description="City/country or 'Remote' if stated")
    seniority: Seniority = Field(description="Seniority level inferred from the text")
    min_years_experience: float | None = Field(
        default=None, description="Minimum years of experience required, if stated"
    )
    required_skills: list[str] = Field(
        description="Skills/tools/technologies the JD says are required or must-have"
    )
    preferred_skills: list[str] = Field(
        default_factory=list,
        description="Skills described as nice-to-have, preferred or a plus",
    )
    responsibilities: list[str] = Field(
        description="Main day-to-day responsibilities, short phrases"
    )
    keywords: list[str] = Field(
        description="Important ATS keywords from the JD (technologies, methods, domains)"
    )
    summary: str = Field(description="One-sentence summary of the role")
