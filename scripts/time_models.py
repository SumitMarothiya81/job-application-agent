"""Run: python -m scripts.time_models            (times every model in GEMINI_MODELS)
       python -m scripts.time_models gemini-3.5-flash gemini-3.5-flash-lite

Times one small structured call per model, with the same timeout and retries the app uses, so you
can put the FASTEST model first in GEMINI_MODELS (in .env). Each call is one real Gemini request."""

import sys
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from job_agent.config import GEMINI_MODELS
from job_agent.llm import get_llm
from job_agent.prompts import JD_EXTRACTION_SYSTEM
from job_agent.schemas import JobDescription


def main() -> None:
    models = [m.removeprefix("models/") for m in sys.argv[1:]] or GEMINI_MODELS
    jd_text = Path("data/sample_jds/ds_intern.txt").read_text(encoding="utf-8")
    print(f"{'model':<30}{'result':<9}{'seconds':>8}")
    for model in models:
        started = time.monotonic()
        try:
            get_llm(model).with_structured_output(JobDescription).invoke(
                [SystemMessage(content=JD_EXTRACTION_SYSTEM), HumanMessage(content=jd_text)])
            print(f"{model:<30}{'OK':<9}{time.monotonic() - started:>8.1f}")
        except Exception as exc:  # noqa: BLE001
            print(f"{model:<30}{'FAILED':<9}{time.monotonic() - started:>8.1f}  {str(exc)[:70]}")


if __name__ == "__main__":
    main()
