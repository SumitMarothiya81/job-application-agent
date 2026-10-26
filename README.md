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
- Job discovery: search Adzuna (official API) or import a CSV; de-duplicated, fresher-filtered and ranked against your resume.
- Apply assistant (LangGraph agent): tailors your real bullets and drafts a recruiter email; every rewrite is fact-checked and you approve each step.
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

## Job discovery setup (optional)
Get free keys at https://developer.adzuna.com and add `ADZUNA_APP_ID`, `ADZUNA_APP_KEY` (and `ADZUNA_COUNTRY`, default `in`)
to `.env` locally or to the Streamlit secrets. Without them the Discover tab still works with CSV import or visitors' own keys.
Adzuna's free tier is limited, so searches are cached for an hour. Jobs by Adzuna.

## Run with Docker
```bash
cp .env.example .env          # add your GOOGLE_API_KEY (no quotes around values: Docker keeps quotes literally)
docker compose up --build     # first build takes a few minutes; the embedding model is baked into the image
```
Open http://localhost:8501. The tracker database and indexed resumes live in the `jobagent_data` volume, so they
survive restarts; `docker compose down -v` deletes them. Your `.env`, real resumes and databases are excluded from
the image by `.dockerignore`. Without compose:
```bash
docker build -t job-application-agent .
docker run --rm -p 8501:8501 --env-file .env -v jobagent_data:/var/lib/jobagent job-application-agent
```
Run the offline evaluation inside the container: `docker run --rm --env-file .env job-application-agent python -m scripts.run_eval --no-write`.

## Run with Docker
```bash
docker compose up --build          # then open http://localhost:8501
```
The app is reachable only from your own computer (`127.0.0.1`) and runs in single-user mode (`SINGLE_USER=true`), so
your tracker is still there after a reload or restart. Your keys are read from `.env` at run time (never copied into the image). The tracker database and uploaded-resume
indexes live in a Docker volume, so they survive restarts. Without Compose:
`docker build -t job-agent .` then `docker run -p 8501:8501 --env-file .env job-agent`.
The first build downloads the embedding model into the image (about 1 GB total).
