"""Run: python -m scripts.run_eval            (prints the report and writes docs/EVALUATION.md)
       python -m scripts.run_eval --no-write (print only)

Offline: no Gemini calls, no keys. The first run downloads the local embedding model if it isn't cached."""

import sys
from pathlib import Path

from job_agent.evaluation import evaluate_matcher, evaluate_rewrites, render_markdown


def main() -> None:
    report = render_markdown(evaluate_matcher(), evaluate_rewrites(check_claims=False), evaluate_rewrites())
    print(report)
    if "--no-write" not in sys.argv:
        out = Path("docs/EVALUATION.md")
        out.parent.mkdir(exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
