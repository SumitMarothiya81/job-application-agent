"""Structured resume: the 'fact base' our agent is allowed to use."""

from pydantic import BaseModel, Field

from job_agent.jd_parser import normalize_list


class Experience(BaseModel):
    organization: str = Field(description="Company or organization name")
    role: str = Field(description="Job title / role")
    duration: str | None = Field(default=None, description="Dates or duration as written")
    bullets: list[str] = Field(
        default_factory=list, description="Achievement/responsibility bullets, copied verbatim"
    )


class Project(BaseModel):
    name: str = Field(description="Project name")
    tech_stack: list[str] = Field(
        default_factory=list,
        description="Technologies explicitly mentioned for THIS project only",
    )
    bullets: list[str] = Field(
        default_factory=list, description="Project description bullets, copied verbatim"
    )


class Education(BaseModel):
    institution: str
    degree: str | None = Field(default=None, description="Degree and field, as written")
    duration: str | None = None
    details: list[str] = Field(
        default_factory=list, description="CGPA, coursework, honours, as written"
    )


class ResumeProfile(BaseModel):
    summary: str | None = Field(default=None, description="Professional summary if present")
    skills: list[str] = Field(
        default_factory=list,
        description="Skills explicitly listed in the resume's skills section(s)",
    )
    experience: list[Experience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)

    def known_skills(self) -> list[str]:
        """Every skill the candidate can truthfully claim (normalized).

        Used later to block the tailoring step from inventing skills.
        """
        skills = list(self.skills)
        for project in self.projects:
            skills.extend(project.tech_stack)
        return normalize_list(skills)


class EvidenceChunk(BaseModel):
    """One retrievable fact from the resume (usually a single bullet)."""

    id: str
    section: str  # "experience" | "project" | "education" | "skills" | "certification"
    source: str  # e.g. "Project: Travel Agent" or "Intern at Acme"
    text: str  # what we show the user / may quote later
