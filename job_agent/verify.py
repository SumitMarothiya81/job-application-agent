"""Deterministic honesty checks for rewritten resume bullets and recruiter emails.

These run BEFORE any AI reviewer, cost nothing, and never get tired:
  * a rewrite may not contain a number that is not in the original bullet;
  * it may not mention a tool/technology that is not in the original bullet
    (or in that project's own tech stack);
  * an email may not mention technologies you do not list on your resume.
They catch the most damaging kind of resume dishonesty (invented metrics and
invented skills). They can't judge claims in plain words ('led a team'); the
LLM reviewer in agent.py covers that as a second layer.
"""

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from job_agent.jd_parser import normalize_skill
from job_agent.matching import RELATED_GROUPS, _tokens, skill_in_known
from job_agent.resume_schema import ResumeProfile
from job_agent.schemas import JobDescription

# Concrete tools/languages/platforms. Deliberately NOT concepts like 'machine learning' or 'nlp':
# those are too generic to police word-for-word, and the LLM reviewer judges them in context.
SKILL_LEXICON = {
    "python", "java", "javascript", "typescript", "sql", "mysql", "postgresql", "sqlite", "mongodb",
    "redis", "pandas", "numpy", "scipy", "scikit-learn", "tensorflow", "keras", "pytorch", "matplotlib",
    "seaborn", "plotly", "power bi", "tableau", "excel", "looker", "streamlit", "flask", "django",
    "fastapi", "docker", "kubernetes", "git", "github", "aws", "azure", "gcp", "linux", "spark",
    "hadoop", "airflow", "kafka", "langchain", "langgraph", "llamaindex", "openai", "gemini", "claude",
    "hugging face", "nltk", "spacy", "opencv", "react", "node.js", "html", "css", "c++", "c#",
    "golang", "tableau", "jira", "figma", "selenium", "junit", "pytest", "jenkins", "terraform",
}

_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?%?")


def numbers_in(text: str) -> set[str]:
    """Numbers as comparable strings: '10,000+' -> '10000', '18%' -> '18%'."""
    return {m.group().replace(",", "") for m in _NUMBER.finditer(text or "")}


def build_vocab(profile: ResumeProfile, jd: JobDescription) -> set[str]:
    """Every skill/tool name we police: a common lexicon, skill groups, the JD's skills, your concrete tools."""
    vocab = set(SKILL_LEXICON)
    for group in RELATED_GROUPS:
        vocab |= group
    vocab |= set(jd.required_skills) | set(jd.preferred_skills)
    concrete = {normalize_skill(t) for t in vocab}
    # Your own skills are policed only when they are concrete tools; concept-like entries such as
    # 'machine learning' or 'data analysis' are left to the AI reviewer, so honest wording isn't rejected.
    vocab |= {s for s in profile.known_skills() if s in concrete}
    return {normalize_skill(term) for term in vocab if term and term.strip()}


def _mentioned(text: str, vocab: set[str]) -> set[str]:
    text_tokens = _tokens(text)
    return {term for term in vocab if (t := _tokens(term)) and t <= text_tokens}


@dataclass
class RewriteCheck:
    issues: list[str] = field(default_factory=list)    # block the rewrite
    warnings: list[str] = field(default_factory=list)  # shown to the user, don't block

    @property
    def ok(self) -> bool:
        return not self.issues


def check_rewrite(source_text: str, allowed_terms: list[str], rewritten: str, vocab: set[str],
                  max_chars: int = 220) -> RewriteCheck:
    result = RewriteCheck()
    text = " ".join((rewritten or "").split())
    if not text:
        result.issues.append("the rewrite is empty")
        return result

    if len(text) > max_chars or len(text) > max(100, int(len(source_text) * 1.6)):
        result.issues.append(f"too long ({len(text)} characters)")

    added = sorted(numbers_in(text) - numbers_in(source_text))
    if added:
        result.issues.append("adds numbers that are not in the original: " + ", ".join(added))
    dropped = sorted(numbers_in(source_text) - numbers_in(text))
    if dropped:
        result.warnings.append("drops numbers from the original: " + ", ".join(dropped))

    allowed_tokens = set(_tokens(source_text))
    for term in allowed_terms:
        allowed_tokens |= _tokens(term)
    foreign = sorted(t for t in _mentioned(text, vocab) if not _tokens(t) <= allowed_tokens)
    if foreign:
        result.issues.append("mentions skills or tools that are not in the original bullet: " + ", ".join(foreign))
    return result


def check_email(subject: str, body: str, allowed_text: str, known_skills: list[str], vocab: set[str],
                max_words: int = 180) -> list[str]:
    """Problems with an email draft. `allowed_text` = resume + job text (numbers must come from there)."""
    full = f"{subject}\n{body}"
    issues = []
    if len(body.split()) > max_words:
        issues.append(f"too long ({len(body.split())} words; keep it under {max_words})")
    added = sorted(numbers_in(full) - numbers_in(allowed_text))
    if added:
        issues.append("contains numbers that are not in your resume or the job text: " + ", ".join(added))
    claimed = sorted(t for t in _mentioned(full, vocab) if not skill_in_known(t, known_skills))
    if claimed:
        issues.append("mentions skills or tools you don't list on your resume: " + ", ".join(claimed))
    return issues


def _canonical_text(text: str) -> str:
    """Lowercase, no punctuation, and British/American -ise/-ize variants made identical."""
    t = " ".join(re.sub(r"[^a-z0-9% ]+", " ", (text or "").lower()).split())
    t = re.sub(r"yz(e|ed|es|ing)\b", r"ys\1", t)
    return re.sub(r"iz(e|ed|es|ing|ation|ations)\b", r"is\1", t)


def is_trivial_change(original: str, rewritten: str) -> bool:
    """True when a 'rewrite' only changes spelling variant, case, punctuation or a typo.

    Such a change isn't tailoring; showing it would be noise, and it can quietly swap British
    spelling for American, so the original is kept.
    """
    a, b = _canonical_text(original), _canonical_text(rewritten)
    return a == b or SequenceMatcher(None, a, b).ratio() >= 0.985


def jd_term_gain(original: str, rewritten: str, terms) -> list[str]:
    """Job terms (skills/keywords) the rewrite contains that the original did not.

    A rewrite that gains nothing from the job is cosmetic ('Added' -> 'Implemented'), so it isn't shown.
    """
    before, after = _tokens(original), _tokens(rewritten)
    gained = []
    for term in terms:
        term_tokens = _tokens(term)
        if term_tokens and term_tokens <= after and not term_tokens <= before and term not in gained:
            gained.append(term)
    return gained
