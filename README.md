# Job Application Agent

An agentic assistant that parses job descriptions, matches them to your resume,
tailors it honestly, drafts recruiter emails, and tracks applications.
You stay in control of every submission.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your GOOGLE_API_KEY
python -m pytest -q
streamlit run app.py          # opens http://localhost:8501
```
CLI demos: `python -m scripts.part1_demo`, `part2_demo`, `part3_demo`.

## Features
- Match a job description to your resume (RAG evidence, explainable score, ATS keywords, gap advice).
- Application tracker: statuses and history, follow-up reminders (+ calendar export), response-rate stats by resume version.

## Modes
- **Demo:** fictional sample resume, the app's Gemini key, per-session and daily caps.
- **My own Gemini key:** paste your key (kept in the browser session only), upload your resume.
  Contact details are masked before anything is sent to Gemini. "Delete my data" removes it.

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (`.env`, `data/resume/`, `data/sessions/` and `.streamlit/secrets.toml` are git-ignored).
2. share.streamlit.io -> New app -> pick the repo, branch `main`, main file `app.py`.
3. Advanced settings -> Python 3.12 (or 3.13) and paste your secrets (see `.streamlit/secrets.toml.example`).
4. Deploy. The first start downloads the embedding model (~67 MB) and takes a minute.
5. Free apps sleep after 12 hours without traffic: open your link before sharing it.

## Tracker data
The demo server's disk is temporary. Use **Export my tracker** to keep a copy, or set `DATABASE_URL` to a hosted Postgres for durable storage.
