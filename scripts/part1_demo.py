"""Run: python -m scripts.part1_demo"""

import logging
from pathlib import Path

from job_agent.jd_parser import parse_jd
from job_agent.llm import invoke_with_fallback

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    # Step 1: prove the connection works with a plain call
    reply = invoke_with_fallback("Reply with exactly: Gemini connection OK")
    print("1) Plain call:", reply.text, "\n")

    # Step 2: structured extraction
    jd_text = Path("data/sample_jds/ds_intern.txt").read_text(encoding="utf-8")
    jd = parse_jd(jd_text)
    print("2) Structured JD:")
    print(jd.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
