"""Turn a ResumeProfile into evidence chunks (one per bullet, with context).

Why chunk by bullet instead of by fixed size? A bullet is one claim.
Retrieval then returns exactly the claim that supports a JD requirement,
and later the verifier can check a tailored sentence against that claim.
Each chunk's embedding text includes its source (project/company) so a
bullet like "Reduced RMSE by 18%" still carries the context it needs.
"""

from job_agent.resume_schema import EvidenceChunk, ResumeProfile


def profile_to_chunks(profile: ResumeProfile) -> list[EvidenceChunk]:
    chunks: list[EvidenceChunk] = []

    for i, exp in enumerate(profile.experience, 1):
        source = f"{exp.role} at {exp.organization}"
        for j, bullet in enumerate(exp.bullets, 1):
            chunks.append(
                EvidenceChunk(id=f"exp-{i}-b{j}", section="experience", source=source, text=bullet)
            )

    for i, proj in enumerate(profile.projects, 1):
        source = f"Project: {proj.name}"
        tech = f" (tech: {', '.join(proj.tech_stack)})" if proj.tech_stack else ""
        for j, bullet in enumerate(proj.bullets, 1):
            chunks.append(
                EvidenceChunk(
                    id=f"proj-{i}-b{j}",
                    section="project",
                    source=source + tech,
                    text=bullet,
                )
            )
        if not proj.bullets and proj.tech_stack:  # project with only a tech line
            chunks.append(
                EvidenceChunk(
                    id=f"proj-{i}-tech",
                    section="project",
                    source=source,
                    text=f"{proj.name} built with {', '.join(proj.tech_stack)}",
                )
            )

    for i, edu in enumerate(profile.education, 1):
        parts = [p for p in (edu.degree, edu.institution, edu.duration, *edu.details) if p]
        chunks.append(
            EvidenceChunk(
                id=f"edu-{i}", section="education", source=f"Education: {edu.institution}",
                text=", ".join(parts),
            )
        )

    if profile.skills:
        chunks.append(
            EvidenceChunk(
                id="skills-1", section="skills", source="Skills section",
                text=", ".join(profile.skills),
            )
        )

    for i, cert in enumerate(profile.certifications, 1):
        chunks.append(
            EvidenceChunk(id=f"cert-{i}", section="certification", source="Certification", text=cert)
        )
    return chunks


def embedding_text(chunk: EvidenceChunk) -> str:
    """The string we embed: source context + the claim itself."""
    return f"{chunk.source}. {chunk.text}"
