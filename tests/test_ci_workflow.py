"""Guards for the CI workflow, so it cannot silently stop testing or start leaking secrets."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
WORKFLOW = yaml.safe_load(RAW)
TRIGGERS = WORKFLOW.get("on", WORKFLOW.get(True))        # PyYAML reads a bare `on:` key as the boolean True
JOB = WORKFLOW["jobs"]["test"]


def test_runs_on_push_to_main_and_on_pull_requests():
    assert "push" in TRIGGERS and "pull_request" in TRIGGERS
    assert TRIGGERS["push"]["branches"] == ["main"]


def test_runs_the_test_suite_on_both_supported_pythons():
    commands = [step.get("run", "") for step in JOB["steps"]]
    assert any("pytest" in command for command in commands)
    assert any("requirements-dev.txt" in command for command in commands)
    assert {"3.12", "3.13"} <= set(JOB["strategy"]["matrix"]["python-version"])


def test_least_privilege_and_no_secrets():
    assert WORKFLOW["permissions"] == {"contents": "read"}
    assert "secrets." not in RAW and "GOOGLE_API_KEY" not in RAW           # the tests must never need a real key


def test_actions_are_pinned_to_a_major_version_and_the_job_cannot_hang():
    uses = [step["uses"] for step in JOB["steps"] if "uses" in step]
    assert uses and all(re.fullmatch(r"[\w./-]+@v\d+", item) for item in uses)   # e.g. actions/checkout@v6, never @main
    assert JOB["timeout-minutes"] <= 30
