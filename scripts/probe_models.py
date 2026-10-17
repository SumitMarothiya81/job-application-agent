"""Run: python -m scripts.probe_models model-a model-b ...
Tries a tiny call on each model and reports OK or the error."""

import sys

from langchain_google_genai import ChatGoogleGenerativeAI

from job_agent.config import require_api_key

for name in sys.argv[1:]:
    name = name.removeprefix("models/")
    try:
        llm = ChatGoogleGenerativeAI(
            model=name, google_api_key=require_api_key(), max_retries=1
        )
        reply = llm.invoke("Reply with the single word OK")
        print(f"{name} -> OK: {reply.text[:60]}")
    except Exception as e:
        print(f"{name} -> FAILED: {str(e)[:110]}")
