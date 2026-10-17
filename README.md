# Job Application Agent

An agentic assistant that parses job descriptions, matches them to your resume,
tailors it honestly, drafts recruiter emails, and tracks applications.
You stay in control of every submission.

## Setup (Part 1)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # then add your GOOGLE_API_KEY
python -m pytest -v
python -m scripts.part1_demo
```
