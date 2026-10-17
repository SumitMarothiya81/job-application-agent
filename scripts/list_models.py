"""Run: python -m scripts.list_models
Lists models your key can use for text generation."""

from google import genai

from job_agent.config import require_api_key

client = genai.Client(api_key=require_api_key())
for m in client.models.list():
    actions = getattr(m, "supported_actions", None) or []
    if "generateContent" in actions:
        print(m.name)
