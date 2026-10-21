import pytest

from job_agent import verify as v
from job_agent.demo import load_demo_profile
from job_agent.schemas import JobDescription, Seniority

SOURCE = "Built a regression model on 10,000+ flight records and reduced RMSE by 18% after feature engineering"


@pytest.fixture(scope="module")
def vocab():
    jd = JobDescription(job_title="Intern", seniority=Seniority.INTERN, required_skills=["python", "docker"],
                        preferred_skills=["machine learning fundamentals"], responsibilities=[], keywords=[], summary="x")
    return v.build_vocab(load_demo_profile(), jd)


def test_numbers_are_compared_in_a_normalized_form():
    assert v.numbers_in("10,000+ records, 18% better, 2022-2026, 3.5 CGPA") == {"10000", "18%", "2022", "2026", "3.5"}
    assert v.numbers_in("no digits here") == set()


def test_vocab_contains_lexicon_groups_jd_and_your_skills(vocab):
    assert {"docker", "tableau", "python", "power bi", "machine learning fundamentals", "streamlit"} <= vocab
    assert "data analysis" not in vocab and "prompt engineering" not in vocab   # your concept-like skills aren't policed


def test_clean_rewrite_passes(vocab):
    out = v.check_rewrite(SOURCE, ["python", "scikit-learn"],
                          "Engineered features and built a regression model on 10000+ flight records, cutting RMSE by 18%", vocab)
    assert out.ok and out.warnings == []


@pytest.mark.parametrize("rewrite, fragment", [
    ("Built a regression model on 12,000 flight records and reduced RMSE by 18%", "adds numbers"),
    ("Built a regression model on 10,000+ flight records and reduced RMSE by 18.5%", "adds numbers"),
    ("Built a Docker-deployed regression model on 10,000+ flight records, RMSE down 18%", "docker"),
    ("Built a regression model with Tableau dashboards on 10,000+ flight records, RMSE down 18%", "tableau"),
    ("", "empty"),
    ("Built a model " + "very " * 60 + "good", "too long"),
])
def test_dishonest_or_broken_rewrites_are_blocked(vocab, rewrite, fragment):
    out = v.check_rewrite(SOURCE, ["python"], rewrite, vocab)
    assert not out.ok and any(fragment in issue for issue in out.issues)


def test_skills_from_the_projects_own_stack_are_allowed_others_are_not(vocab):
    text = "Trained a classifier in scikit-learn on flight records"
    assert v.check_rewrite("Trained a classifier on flight records", ["scikit-learn"], text, vocab).ok
    assert not v.check_rewrite("Trained a classifier on flight records", [], text, vocab).ok


def test_dropping_a_number_is_a_warning_not_a_failure(vocab):
    out = v.check_rewrite(SOURCE, [], "Built a regression model on flight records and cut RMSE after feature engineering", vocab)
    assert out.ok and "drops numbers" in out.warnings[0] and "18%" in out.warnings[0]


def test_generic_words_are_not_policed(vocab):
    # 'machine learning' alone is a concept, not a tool: the AI reviewer judges it, the word check doesn't.
    assert v.check_rewrite(SOURCE, [], "Built a machine learning regression model on 10,000+ flight records, RMSE down 18%", vocab).ok


def test_email_checks(vocab):
    known = load_demo_profile().known_skills()
    allowed = "Python SQL pandas. 6 month internship. CGPA 8.4"
    assert v.check_email("Application", "I use Python and SQL for 6 month projects.", allowed, known, vocab) == []
    issues = v.check_email("Application", "I built Docker pipelines and improved speed by 40%.", allowed, known, vocab)
    assert any("40%" in i for i in issues) and any("docker" in i for i in issues)
    long_body = "word " * 200
    assert any("too long" in i for i in v.check_email("S", long_body, allowed, known, vocab))


@pytest.mark.parametrize("original, rewrite, trivial", [
    ("Designed a dashboard analysing batting performance", "Designed a dashboard analyzing batting performance", True),
    ("Organised a data cleaning drive", "Organized a data cleaning drive", True),
    ("Wrote SQL queries to prepare data", "wrote sql queries to prepare data.", True),
    ("Wrote SQL queries to prepare data", "Wrote SQL queries to prepare data", True),
    ("Built a regression model on 10,000+ flight records and reduced RMSE by 18%", "Key result: Built a regression model on 10,000+ flight records and reduced RMSE by 18%", False),
    ("Added prompt templates to keep answers short", "Created prompt templates that keep answers short and structured", False),
    ("Cleaned 5,000 emails", "Cleaned 6,000 emails", False),
])
def test_trivial_changes_are_spelling_case_and_punctuation_only(original, rewrite, trivial):
    assert v.is_trivial_change(original, rewrite) is trivial


def test_jd_term_gain_counts_only_new_job_terms():
    terms = ["python", "sql", "dashboards", "power bi"]
    assert v.jd_term_gain("Cleaned 5,000 emails using NLTK", "Cleaned 5,000 emails using NLTK in Python", terms) == ["python"]
    assert v.jd_term_gain("Built a dashboard", "Built a Power BI dashboard", terms) == ["power bi"]   # 'dashboard' was already there
    assert v.jd_term_gain("Added prompt templates", "Implemented prompt templates", terms) == []   # a verb swap gains nothing
    assert v.jd_term_gain("Used SQL daily", "Used SQL every day", terms) == []                    # already there
