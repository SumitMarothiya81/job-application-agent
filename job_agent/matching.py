"""Match a parsed resume against a parsed job description.

Each JD skill gets one of three statuses:
  explicit - the skill is literally on the resume (deterministic check)
  related  - not listed, but the resume shows a closely related skill
             (curated groups, e.g. Power BI <-> Tableau) or a very similar bullet
  missing  - no honest evidence

Why not just use the embedding score? We measured it on a real run: skills that
ARE on the resume scored 0.69-0.75 and skills that are NOT scored 0.58-0.64.
That gap is too narrow to trust alone (a listed skill, git, scored 0.62 while
an unlisted one, tableau, scored 0.64). So the cheap, exact check decides
'explicit', a small taxonomy handles transferable skills, and the semantic
score is only a last-resort signal with a tunable threshold.
"""

import os
import re
from enum import Enum

from pydantic import BaseModel, Field

from job_agent import embeddings
from job_agent.jd_parser import normalize_list, normalize_skill
from job_agent.resume_schema import EvidenceChunk, ResumeProfile
from job_agent.schemas import JobDescription, Seniority
from job_agent.vector_store import VectorStore

DEFAULT_RELATED_THRESHOLD = 0.75
REQUIRED_WEIGHT = 0.7
PREFERRED_WEIGHT = 0.3
SEMANTIC_SECTIONS = {"experience", "project", "certification"}  # bullet-level evidence only
SECTION_PRIORITY = {"experience": 4, "project": 3, "certification": 2, "education": 1, "skills": 1}

# Skills that transfer to each other. Names are normalized (see normalize_skill).
RELATED_GROUPS: list[set[str]] = [
    {"power bi", "tableau", "looker", "qlik", "metabase"},
    {"sql", "mysql", "postgresql", "sqlite", "sql server", "oracle"},
    {"chroma", "faiss", "pinecone", "qdrant", "weaviate", "milvus", "vector databases"},
    {"langchain", "langgraph", "llamaindex", "haystack"},
    {"pytorch", "tensorflow", "keras", "deep learning"},
    {"aws", "azure", "gcp", "google cloud"},
    {"pandas", "numpy", "polars"},
    {"matplotlib", "seaborn", "plotly"},
    {"flask", "fastapi", "django"},
    {"openai", "gemini", "google gemini", "anthropic", "claude"},
    {"git", "github", "gitlab"},
]


class SkillStatus(str, Enum):
    EXPLICIT = "explicit"
    RELATED = "related"
    MISSING = "missing"


STATUS_POINTS = {SkillStatus.EXPLICIT: 1.0, SkillStatus.RELATED: 0.5, SkillStatus.MISSING: 0.0}


class Evidence(BaseModel):
    chunk_id: str
    section: str
    source: str
    text: str
    score: float


class SkillMatch(BaseModel):
    skill: str
    kind: str  # "required" | "preferred"
    status: SkillStatus
    score: float = Field(description="Best semantic similarity to a resume bullet (0-1)")
    related_via: str | None = Field(default=None, description="Resume skill that makes it 'related'")
    evidence: list[Evidence] = Field(default_factory=list)


class MatchReport(BaseModel):
    job_title: str
    company: str | None
    overall_score: int
    verdict: str
    required_coverage: float | None
    preferred_coverage: float | None
    matches: list[SkillMatch]
    strengths: list[str]  # required skills with explicit evidence
    related_skills: list[str]
    missing_required: list[str]
    missing_preferred: list[str]
    ats_hits: list[str]
    ats_missing: list[str]
    fit_notes: list[str]


# --------------------------------------------------------------------------
# Skill comparison helpers
# --------------------------------------------------------------------------
_TOKEN = re.compile(r"[a-z0-9\+\#\.]+")


def _tokens(text: str) -> set[str]:
    """Lowercase word tokens with a crude plural fix (databases -> database)."""
    found = _TOKEN.findall(normalize_skill(text))
    return {t[:-1] if len(t) > 3 and t.endswith("s") else t for t in found}


def skill_in_known(skill: str, known: list[str]) -> bool:
    """True if `skill` is on the resume's skill list (exact or whole-phrase match).

    Single-word skills must match exactly, so 'java' never matches 'javascript'
    and 'sql' never matches 'mysql'. Multi-word skills also match when one is a
    phrase of the other ('machine learning' vs 'machine learning fundamentals').
    """
    st = _tokens(skill)
    if not st:
        return False
    for item in known:
        kt = _tokens(item)
        if st == kt:
            return True
        if len(st) >= 2 and st <= kt:
            return True
        if len(kt) >= 2 and kt <= st:
            return True
    return False


def _related_via(skill: str, known: list[str]) -> str | None:
    """A resume skill from the same group as `skill`, if any."""
    for group in RELATED_GROUPS:
        if any(_tokens(skill) == _tokens(member) for member in group):
            for member in sorted(group):
                if _tokens(member) != _tokens(skill) and skill_in_known(member, known):
                    return member
    return None


def _mentioning(skill: str, scored: list[tuple[EvidenceChunk, float]], limit: int = 2):
    """Chunks that mention the skill. Best first: experience/project sections, then
    bullets whose own text names the skill (not just the project's tech line), then score."""
    st = _tokens(skill)
    hits = [(c, s) for c, s in scored if st and st <= _tokens(f"{c.source} {c.text}")]
    hits.sort(
        key=lambda cs: (
            SECTION_PRIORITY.get(cs[0].section, 0),
            bool(st) and st <= _tokens(cs[0].text),
            cs[1],
        ),
        reverse=True,
    )
    return hits[:limit]


def _to_evidence(pairs) -> list[Evidence]:
    return [
        Evidence(chunk_id=c.id, section=c.section, source=c.source, text=c.text, score=round(s, 3))
        for c, s in pairs
    ]


def get_related_threshold() -> float:
    return float(os.getenv("RELATED_THRESHOLD", DEFAULT_RELATED_THRESHOLD))


# --------------------------------------------------------------------------
# Per-skill matching
# --------------------------------------------------------------------------
def _match_skill(skill: str, kind: str, known: list[str], store: VectorStore, threshold: float):
    scored: list[tuple[EvidenceChunk, float]] = []
    if store.chunks:
        query = embeddings.embed_query(f"experience with {skill}")
        scored = list(zip(store.chunks, (store.vectors @ query).tolist()))

    bullets = sorted(
        (cs for cs in scored if cs[0].section in SEMANTIC_SECTIONS),
        key=lambda cs: cs[1],
        reverse=True,
    )
    best = round(bullets[0][1], 3) if bullets else 0.0

    if skill_in_known(skill, known):
        evidence = _mentioning(skill, scored) or bullets[:2]
        return SkillMatch(skill=skill, kind=kind, status=SkillStatus.EXPLICIT, score=best,
                          evidence=_to_evidence(evidence))

    via = _related_via(skill, known)
    if via:
        evidence = _mentioning(via, scored) or bullets[:2]
        return SkillMatch(skill=skill, kind=kind, status=SkillStatus.RELATED, score=best,
                          related_via=via, evidence=_to_evidence(evidence))

    if bullets and best >= threshold:
        return SkillMatch(skill=skill, kind=kind, status=SkillStatus.RELATED, score=best,
                          evidence=_to_evidence([cs for cs in bullets[:2] if cs[1] >= threshold]))

    return SkillMatch(skill=skill, kind=kind, status=SkillStatus.MISSING, score=best)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def _coverage(matches: list[SkillMatch], kind: str) -> float | None:
    subset = [m for m in matches if m.kind == kind]
    if not subset:
        return None
    return sum(STATUS_POINTS[m.status] for m in subset) / len(subset)


def overall_score(required: float | None, preferred: float | None) -> int:
    """0-100: required skills weigh 70%, preferred 30% (renormalized if one is absent)."""
    if required is None and preferred is None:
        return 0
    if preferred is None:
        return round(100 * required)
    if required is None:
        return round(100 * preferred)
    return round(100 * (REQUIRED_WEIGHT * required + PREFERRED_WEIGHT * preferred))


def verdict_for(score: int) -> str:
    if score >= 75:
        return "strong"
    if score >= 55:
        return "good"
    if score >= 35:
        return "stretch"
    return "weak"


def _profile_text(profile: ResumeProfile) -> str:
    parts: list[str] = [profile.summary or "", *profile.skills, *profile.certifications]
    for exp in profile.experience:
        parts += [exp.organization, exp.role, *exp.bullets]
    for proj in profile.projects:
        parts += [proj.name, *proj.tech_stack, *proj.bullets]
    for edu in profile.education:
        parts += [edu.institution, edu.degree or "", *edu.details]
    return " ".join(parts)


def ats_keyword_coverage(profile: ResumeProfile, jd: JobDescription) -> tuple[list[str], list[str]]:
    """Which JD keywords already appear on the resume (ATS filters look for these)."""
    known = profile.known_skills()
    text_tokens = _tokens(_profile_text(profile))
    hits, missing = [], []
    for keyword in normalize_list(jd.keywords):
        kt = _tokens(keyword)
        (hits if skill_in_known(keyword, known) or (kt and kt <= text_tokens) else missing).append(keyword)
    return hits, missing


def fit_notes_for(jd: JobDescription) -> list[str]:
    notes = []
    if jd.min_years_experience is not None and jd.min_years_experience >= 2:
        notes.append(
            f"Asks for {jd.min_years_experience:g}+ years of experience; freshers are often "
            "screened out unless projects are very strong."
        )
    if jd.seniority in (Seniority.MID, Seniority.SENIOR):
        notes.append(f"Role looks {jd.seniority.value}-level, not fresher-level.")
    return notes


def match_resume_to_jd(
    profile: ResumeProfile,
    store: VectorStore,
    jd: JobDescription,
    related_threshold: float | None = None,
) -> MatchReport:
    threshold = get_related_threshold() if related_threshold is None else related_threshold
    known = profile.known_skills()

    matches: list[SkillMatch] = []
    for kind, skills in (("required", jd.required_skills), ("preferred", jd.preferred_skills)):
        matches += [_match_skill(s, kind, known, store, threshold) for s in skills]

    required_cov = _coverage(matches, "required")
    preferred_cov = _coverage(matches, "preferred")
    score = overall_score(required_cov, preferred_cov)
    ats_hits, ats_missing = ats_keyword_coverage(profile, jd)

    def names(kind: str, status: SkillStatus) -> list[str]:
        return [m.skill for m in matches if m.kind == kind and m.status == status]

    return MatchReport(
        job_title=jd.job_title,
        company=jd.company,
        overall_score=score,
        verdict=verdict_for(score),
        required_coverage=required_cov,
        preferred_coverage=preferred_cov,
        matches=matches,
        strengths=names("required", SkillStatus.EXPLICIT),
        related_skills=[m.skill for m in matches if m.status == SkillStatus.RELATED],
        missing_required=names("required", SkillStatus.MISSING),
        missing_preferred=names("preferred", SkillStatus.MISSING),
        ats_hits=ats_hits,
        ats_missing=ats_missing,
        fit_notes=fit_notes_for(jd),
    )
