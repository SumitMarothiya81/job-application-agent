"""Offline evaluation: how good are the matcher and the honesty checker? (no API calls, no keys)

Two labelled sets, written by hand BEFORE looking at results:
  * matcher cases   - for a fixed (fictional) resume, is each job skill explicit / related / missing?
  * rewrite cases   - for real-looking bullets, is a rewrite faithful or dishonest, and does verify.py
                      block the dishonest ones without blocking the faithful ones?

Dishonest cases include three SUBTLE ones the word checks cannot see ('outperformed all baselines',
'clinically validated', changed meaning). They are here on purpose: the numbers should show the gap that
the AI reviewer has to cover, not hide it.
"""

from dataclasses import dataclass

from job_agent import embeddings
from job_agent.demo import build_store, load_demo_profile
from job_agent.jd_parser import normalize_skill
from job_agent.matching import match_resume_to_jd
from job_agent.resume_schema import ResumeProfile
from job_agent.schemas import JobDescription, Seniority
from job_agent.verify import build_vocab, check_rewrite

# ---------------------------------------------------------------------------------------------
# 1) Matcher: (skill as a job posting might word it, expected status, why)
# ---------------------------------------------------------------------------------------------
MATCHER_CASES: list[tuple[str, str, str]] = [
    # explicit: the resume really lists it (including spelling variants)
    ("python", "explicit", "listed"), ("sql", "explicit", "listed"), ("pandas", "explicit", "listed"),
    ("numpy", "explicit", "listed"), ("matplotlib", "explicit", "listed"), ("git", "explicit", "listed"),
    ("power bi", "explicit", "listed"), ("streamlit", "explicit", "listed"), ("nltk", "explicit", "in a project's stack"),
    ("scikit learn", "explicit", "alias of scikit-learn"), ("sklearn", "explicit", "alias of scikit-learn"),
    ("machine learning fundamentals", "explicit", "phrase containing a listed skill"),
    ("data analysis", "explicit", "listed"), ("prompt engineering", "explicit", "listed"),
    ("gemini", "explicit", "the resume lists Google Gemini"),
    # related: a transferable skill, but not the same one
    ("tableau", "related", "same kind of tool as Power BI"), ("looker", "related", "BI tool, like Power BI"),
    ("mysql", "related", "SQL dialect"), ("postgresql", "related", "SQL dialect"), ("sql server", "related", "SQL dialect"),
    ("github", "related", "built around git"), ("seaborn", "related", "plotting, like matplotlib"),
    ("plotly", "related", "plotting, like matplotlib"), ("polars", "related", "dataframes, like pandas"),
    ("openai", "related", "LLM API, like Gemini"),
    # missing: nothing on the resume supports it
    ("docker", "missing", ""), ("kubernetes", "missing", ""), ("langchain", "missing", ""), ("langgraph", "missing", ""),
    ("faiss", "missing", ""), ("chroma", "missing", ""), ("vector databases", "missing", ""), ("aws", "missing", ""),
    ("java", "missing", ""), ("javascript", "missing", ""), ("spark", "missing", ""), ("tensorflow", "missing", ""),
    ("fastapi", "missing", ""), ("react", "missing", ""), ("excel", "missing", ""), ("c++", "missing", ""),
]

# ---------------------------------------------------------------------------------------------
# 2) Rewrites: source bullets (all from the fictional sample resume) and labelled rewrites
# ---------------------------------------------------------------------------------------------
SOURCES: dict[str, tuple[str, list[str]]] = {
    "S1": ("Built a regression model on 10,000+ flight records and reduced RMSE by 18% after feature engineering and hyperparameter tuning",
           ["python", "pandas", "scikit-learn", "streamlit"]),
    "S2": ("Deployed the model as a Streamlit web app so users can estimate fares from route and date",
           ["python", "pandas", "scikit-learn", "streamlit"]),
    "S3": ("Trained a Naive Bayes classifier with TF-IDF features that reached 97% accuracy on a held-out test set",
           ["python", "scikit-learn", "nltk"]),
    "S4": ("Cleaned and tokenized 5,000 emails using NLTK", ["python", "scikit-learn", "nltk"]),
    "S5": ("Designed an interactive Power BI dashboard analysing batting performance across formats and seasons", ["power bi", "sql"]),
    "S6": ("Wrote SQL queries to prepare match-level data for reporting", ["power bi", "sql"]),
    "S7": ("Built a Gemini-powered assistant that answers health questions with safety disclaimers", ["python", "google gemini", "streamlit"]),
    "S8": ("Added prompt templates to keep answers short and structured", ["python", "google gemini", "streamlit"]),
}


@dataclass(frozen=True)
class RewriteCase:
    source: str
    rewrite: str
    faithful: bool
    category: str


F, D = True, False
REWRITE_CASES: list[RewriteCase] = [
    # ---- faithful: rewording, reordering, stronger verbs, spelling, a skill from the project's own stack ----
    RewriteCase("S1", "Built a regression model on 10,000+ flight records, cutting RMSE by 18% through feature engineering and hyperparameter tuning", F, "reworded"),
    RewriteCase("S1", "Reduced RMSE by 18% on a 10,000+ record flight-fare regression model via feature engineering and hyperparameter tuning", F, "reordered"),
    RewriteCase("S1", "Engineered features and tuned hyperparameters of a regression model, reducing RMSE by 18% on 10000+ flight records", F, "number format"),
    RewriteCase("S1", "Built a regression model on over 10,000 flight records and reduced RMSE by 18% after feature engineering and hyperparameter tuning", F, "number format"),
    RewriteCase("S2", "Shipped the fare-prediction model as a Streamlit web app that estimates fares from route and date", F, "reworded"),
    RewriteCase("S2", "Deployed the model as a Streamlit app in Python so users can estimate fares from route and date", F, "stack skill"),
    RewriteCase("S2", "Deployed the model as a Streamlit web app, letting users estimate fares from route and date", F, "reworded"),
    RewriteCase("S3", "Trained a Naive Bayes spam classifier with TF-IDF features, achieving 97% accuracy on a held-out test set", F, "reworded"),
    RewriteCase("S3", "Trained a Naive Bayes classifier using TF-IDF features; reached 97% accuracy on a held-out test set", F, "reworded"),
    RewriteCase("S4", "Cleaned and tokenized 5,000 emails using NLTK in Python", F, "stack skill"),
    RewriteCase("S4", "Tokenized and cleaned 5,000 emails with NLTK", F, "reordered"),
    RewriteCase("S5", "Designed an interactive Power BI dashboard analyzing batting performance across formats and seasons", F, "spelling"),
    RewriteCase("S5", "Created interactive dashboards in Power BI to analyse batting performance by format and season", F, "reworded"),
    RewriteCase("S5", "Designed an interactive Power BI dashboard that analyses batting performance across formats and seasons", F, "reworded"),
    RewriteCase("S6", "Prepared match-level data for reporting by writing SQL queries", F, "reordered"),
    RewriteCase("S6", "Wrote SQL queries that prepare match-level data for reporting", F, "reworded"),
    RewriteCase("S7", "Built a Gemini-powered health-question assistant with built-in safety disclaimers", F, "reworded"),
    RewriteCase("S7", "Developed a Python assistant powered by Google Gemini that answers health questions and adds safety disclaimers", F, "stack skill"),
    RewriteCase("S8", "Added prompt templates so answers stay short and structured", F, "reworded"),
    RewriteCase("S8", "Implemented prompt templates to keep answers concise and structured", F, "verb swap"),
    # ---- dishonest: invented or changed numbers ----
    RewriteCase("S1", "Built a regression model on 10,000+ flight records and reduced RMSE by 25% after feature engineering and hyperparameter tuning", D, "changed number"),
    RewriteCase("S1", "Built a regression model on 50,000+ flight records and reduced RMSE by 18% after feature engineering and hyperparameter tuning", D, "changed number"),
    RewriteCase("S4", "Cleaned and tokenized 15,000 emails using NLTK", D, "changed number"),
    RewriteCase("S3", "Trained a Naive Bayes classifier with TF-IDF features that reached 99% accuracy on a held-out test set", D, "changed number"),
    RewriteCase("S5", "Designed an interactive Power BI dashboard used by 200 analysts to study batting performance across formats and seasons", D, "invented number"),
    # ---- dishonest: a tool or technology that is not in the bullet ----
    RewriteCase("S2", "Containerized the model with Docker and deployed it as a Streamlit web app so users can estimate fares from route and date", D, "invented tool"),
    RewriteCase("S6", "Wrote SQL queries on AWS Redshift to prepare match-level data for reporting", D, "invented tool"),
    RewriteCase("S5", "Designed interactive Power BI and Tableau dashboards analysing batting performance across formats and seasons", D, "invented tool"),
    RewriteCase("S7", "Built a Gemini-powered assistant with LangChain that answers health questions with safety disclaimers", D, "invented tool"),
    RewriteCase("S1", "Built a PyTorch regression model on 10,000+ flight records and reduced RMSE by 18% after feature engineering", D, "invented tool"),
    RewriteCase("S4", "Cleaned and tokenized 5,000 emails using NLTK and spaCy", D, "invented tool"),
    RewriteCase("S8", "Added prompt templates backed by FAISS retrieval to keep answers short and structured", D, "invented tool"),
    # ---- dishonest: inflated claims in plain words (no number, no tool) ----
    RewriteCase("S2", "Led the end-to-end deployment of the model as a Streamlit web app so users can estimate fares", D, "inflated claim"),
    RewriteCase("S6", "Owned the SQL data pipeline that prepares match-level data for reporting", D, "inflated claim"),
    RewriteCase("S7", "Architected a production-grade Gemini-powered assistant that answers health questions with safety disclaimers", D, "inflated claim"),
    RewriteCase("S3", "Spearheaded a Naive Bayes classifier with TF-IDF features that reached 97% accuracy on a held-out test set", D, "inflated claim"),
    RewriteCase("S5", "Designed an enterprise-scale interactive Power BI dashboard analysing batting performance across formats and seasons", D, "inflated claim"),
    RewriteCase("S8", "Managed a team that added prompt templates to keep answers short and structured", D, "inflated claim"),
    # ---- dishonest, SUBTLE: only a careful reader (the AI reviewer) can catch these ----
    RewriteCase("S1", "Built a regression model that outperformed all baseline models on 10,000+ flight records, reducing RMSE by 18%", D, "subtle claim"),
    RewriteCase("S7", "Built a clinically validated Gemini-powered assistant that answers health questions with safety disclaimers", D, "subtle claim"),
    RewriteCase("S2", "Deployed the model as a Streamlit web app that travellers use to book flights from route and date", D, "subtle claim"),
]

EVAL_JD = JobDescription(
    job_title="Data Science Intern", seniority=Seniority.INTERN,
    required_skills=["python", "sql", "pandas", "scikit-learn", "machine learning"],
    preferred_skills=["langchain", "langgraph", "faiss", "chroma", "docker", "git", "power bi", "tableau"],
    responsibilities=[], keywords=[], summary="Evaluation job")


# ---------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------
def evaluate_matcher(profile: ResumeProfile | None = None) -> dict:
    profile = profile or load_demo_profile()
    store = build_store(profile)
    skills = [normalize_skill(skill) for skill, _, _ in MATCHER_CASES]
    jd = JobDescription(job_title="Evaluation", seniority=Seniority.UNKNOWN, required_skills=skills,
                        preferred_skills=[], responsibilities=[], keywords=[], summary="")
    actual = {m.skill: m.status.value for m in match_resume_to_jd(profile, store, jd).matches}

    rows, confusion = [], {}
    for skill, expected, why in MATCHER_CASES:
        got = actual[normalize_skill(skill)]
        confusion[(expected, got)] = confusion.get((expected, got), 0) + 1
        rows.append({"skill": skill, "expected": expected, "actual": got, "why": why, "ok": got == expected})
    wrong = [r for r in rows if not r["ok"]]
    rank = {"missing": 0, "related": 1, "explicit": 2}
    return {
        "total": len(rows), "correct": len(rows) - len(wrong), "accuracy": (len(rows) - len(wrong)) / len(rows),
        "confusion": confusion, "errors": wrong,
        # the harmful direction: crediting a skill the resume doesn't support
        "false_credit": [r for r in wrong if rank[r["actual"]] > rank[r["expected"]] and r["expected"] == "missing"],
        "over_credit": [r for r in wrong if rank[r["actual"]] > rank[r["expected"]]],
        "under_credit": [r for r in wrong if rank[r["actual"]] < rank[r["expected"]]],
    }


def evaluate_rewrites(profile: ResumeProfile | None = None, check_claims: bool = True) -> dict:
    profile = profile or load_demo_profile()
    vocab = build_vocab(profile, EVAL_JD)
    by_category: dict[str, list[int]] = {}
    missed, wrongly_blocked = [], []
    dishonest = faithful = blocked_dishonest = 0

    for case in REWRITE_CASES:
        text, tech = SOURCES[case.source]
        issues = check_rewrite(text, tech, case.rewrite, vocab, check_claims=check_claims).issues
        blocked = bool(issues)
        if case.faithful:
            faithful += 1
            if blocked:
                wrongly_blocked.append({"case": case, "issues": issues})
        else:
            dishonest += 1
            counts = by_category.setdefault(case.category, [0, 0])
            counts[1] += 1
            if blocked:
                blocked_dishonest += 1
                counts[0] += 1
            else:
                missed.append(case)
    return {
        "total": len(REWRITE_CASES), "dishonest": dishonest, "faithful": faithful,
        "blocked_dishonest": blocked_dishonest, "recall": blocked_dishonest / dishonest,
        "wrongly_blocked": wrongly_blocked, "false_reject_rate": len(wrongly_blocked) / faithful,
        "by_category": {k: tuple(v) for k, v in by_category.items()}, "missed": missed,
    }


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------
def render_markdown(matcher: dict, baseline: dict, current: dict) -> str:
    pct = lambda x: f"{100 * x:.0f}%"  # noqa: E731
    lines = [
        "# Evaluation", "",
        f"_Embedding setup: `{embeddings.embedding_id()}`. Generated by `python -m scripts.run_eval`; "
        "no API calls. The cases were written by hand before looking at results, and are small "
        f"({matcher['total']} matcher cases, {current['total']} rewrite cases), so treat the numbers as a "
        "regression guard, not as a benchmark._", "",
        "## 1. Skill matcher", "",
        f"**Accuracy: {matcher['correct']}/{matcher['total']} ({pct(matcher['accuracy'])})**. "
        f"Skills wrongly credited that the resume doesn't support at all: **{len(matcher['false_credit'])}**. "
        f"Any over-crediting: {len(matcher['over_credit'])}. Under-crediting: {len(matcher['under_credit'])}.", "",
        "| expected \\ actual | explicit | related | missing |", "|---|---|---|---|",
    ]
    for expected in ("explicit", "related", "missing"):
        lines.append(f"| **{expected}** | " + " | ".join(str(matcher["confusion"].get((expected, got), 0))
                                                         for got in ("explicit", "related", "missing")) + " |")
    if matcher["errors"]:
        lines += ["", "Errors:", ""] + [f"- `{e['skill']}`: expected {e['expected']}, got {e['actual']}" for e in matcher["errors"]]

    lines += ["", "## 2. Honesty checker (rewrite fact-check, word-level layer only)", "",
              f"{current['dishonest']} dishonest and {current['faithful']} faithful rewrites. The AI reviewer "
              "is a second layer and is NOT measured here.", "",
              "| | blocks dishonest rewrites | wrongly blocks faithful rewrites |", "|---|---|---|",
              f"| before the claim-word rule | {baseline['blocked_dishonest']}/{baseline['dishonest']} ({pct(baseline['recall'])}) "
              f"| {len(baseline['wrongly_blocked'])}/{baseline['faithful']} |",
              f"| **now** | **{current['blocked_dishonest']}/{current['dishonest']} ({pct(current['recall'])})** "
              f"| **{len(current['wrongly_blocked'])}/{current['faithful']}** |", "",
              "| kind of dishonesty | blocked |", "|---|---|"]
    lines += [f"| {category} | {hit}/{total} |" for category, (hit, total) in current["by_category"].items()]
    if current["missed"]:
        lines += ["", "Not caught by the word-level checks (the AI reviewer must catch these):", ""]
        lines += [f"- ({c.category}) {c.rewrite}" for c in current["missed"]]
    if current["wrongly_blocked"]:
        lines += ["", "Faithful rewrites that were wrongly blocked:", ""]
        lines += [f"- {w['case'].rewrite}  -> {'; '.join(w['issues'])}" for w in current["wrongly_blocked"]]
    return "\n".join(lines) + "\n"
