"""Regression guard: the evaluation numbers may not silently get worse."""

from job_agent import evaluation as ev


def test_the_labelled_sets_are_well_formed():
    assert len({c.rewrite for c in ev.REWRITE_CASES}) == len(ev.REWRITE_CASES)            # no duplicate cases
    assert {c.source for c in ev.REWRITE_CASES} <= set(ev.SOURCES)
    assert sum(c.faithful for c in ev.REWRITE_CASES) >= 15 and sum(not c.faithful for c in ev.REWRITE_CASES) >= 15
    assert {status for _, status, _ in ev.MATCHER_CASES} == {"explicit", "related", "missing"}
    assert len({skill for skill, _, _ in ev.MATCHER_CASES}) == len(ev.MATCHER_CASES)


def test_matcher_floor():
    result = ev.evaluate_matcher()
    assert result["accuracy"] >= 0.95
    assert result["false_credit"] == []            # never credit a skill the resume doesn't support at all
    assert result["total"] == result["correct"] + len(result["errors"])


def test_honesty_checker_floors_and_known_gap():
    now, before = ev.evaluate_rewrites(), ev.evaluate_rewrites(check_claims=False)
    assert now["wrongly_blocked"] == [] and before["wrongly_blocked"] == []     # faithful rewrites always pass
    assert now["recall"] >= 0.85 and before["recall"] < now["recall"]            # the claim-word rule earns its place
    for category in ("changed number", "invented number", "invented tool", "inflated claim"):
        hit, total = now["by_category"][category]
        assert hit == total, category                                            # every plain violation is blocked
    assert {c.category for c in now["missed"]} == {"subtle claim"}               # the documented limitation
    assert now["by_category"]["subtle claim"] == (0, 3)


def test_the_report_renders_the_headline_numbers():
    text = ev.render_markdown(ev.evaluate_matcher(), ev.evaluate_rewrites(check_claims=False), ev.evaluate_rewrites())
    for fragment in ("# Evaluation", "## 1. Skill matcher", "## 2. Honesty checker", "before the claim-word rule",
                     "Not caught by the word-level checks"):
        assert fragment in text
