# Job Application Agent: Progress

Stack: Python, Gemini (langchain-google-genai), Pydantic, fastembed (local embeddings), numpy, pypdf, python-docx, pytest.
Layout: package `job_agent/`, scripts run with `python -m scripts.<name>` from the project root.
Decisions: Gemini LLM with model fallback (GEMINI_MODELS list, switches on 503/429); local embeddings via
fastembed (BAAI/bge-small-en-v1.5) behind embed_passages()/embed_query(); multi-user from the start
(session_id folders, user_id in DB later, per-request API key later); PII redacted before any text goes to
Gemini; deploy early (after Part 3) on Streamlit Community Cloud / Render; demo + bring-your-own-key + self-host modes.

## Roadmap
1. [x] Foundations: setup, Gemini client + fallback, JD structured extraction + normalization + tests
2. [x] Resume ingestion + RAG: loader (PDF/DOCX/TXT), PII redaction, structured parse, bullet chunks,
       local embeddings, per-session vector store, retrieval
3a. [x] Match scoring (explicit / related / missing), ATS keyword coverage, fit notes, gap advice (CLI)
3b. [x] Streamlit app (demo + own-key modes, usage caps) built and tested headlessly; [ ] deployed to Streamlit Community Cloud
4. [x] Application tracker: SQLAlchemy DB (SQLite, Postgres-ready), statuses + history, follow-up reminders (in-app banner + .ics calendar export), stats incl. response rate per resume version, JSON export/import, Tracker tab; deployed with the Streamlit app
5. [x] Job discovery: Adzuna API client (cached, key-safe errors), CSV import, dedupe, fresher filter, quick-fit ranking, Discover tab (save to tracker / load into Match tab)
6a. [x] Agent brain: verify.py (number/tool honesty checks), agent_prompts.py, agent.py (LangGraph: parse -> match -> pick bullets -> tailor -> verify -> retry once -> HUMAN review -> email -> fact-check/template fallback -> HUMAN review -> kit), 41 tests with fake models
6b. [x] Apply assistant tab (ui_agent.py + app.py): start, bullet review, email review, kit + download, save to tracker; demo caps AGENT_SESSION_LIMIT=2, AGENT_DAILY_LIMIT=15
7. [ ] Full UI (Streamlit/FastAPI) + PDF export
8. [ ] Evaluation, tracing, Docker, final deploy, README, demo video

## Notes
- Gemini 3 models return content blocks; use reply.text for plain text.
- Vector store is exact numpy cosine search (resume = ~30-80 chunks). Chroma/FAISS can replace
  job_agent/vector_store.py later without touching other files.
- Index stores embedding_id; querying with a different embedding model raises IndexMismatchError.
- Tests set EMBEDDING_BACKEND=fake (conftest.py) so they never download models or call the API.

- Measured on a real run: skills on the resume scored 0.69-0.75 (cosine), skills NOT on it 0.58-0.64, with overlap
  (git listed 0.62 vs tableau unlisted 0.64). So status = exact skill check, then curated RELATED_GROUPS
  taxonomy, then semantic score >= RELATED_THRESHOLD (default 0.68, env-tunable). Calibrate on real data.
- Score = 100 * (0.7 * required_coverage + 0.3 * preferred_coverage); explicit=1, related=0.5, missing=0.

- Streamlit Community Cloud (checked Oct 2026 docs): memory 690MB min / 2.7GB max, apps sleep after 12h without
  traffic, GitHub required, secrets via app settings. fastembed+onnxruntime memory use must be verified on deploy.
- Bring-your-own-key: llm.use_api_key(key) context manager (ContextVar); keys are never stored on disk.
- Demo mode caps: DEMO_SESSION_LIMIT (default 5) and DEMO_DAILY_LIMIT (default 100), in-memory (reset on restart).

- Tracker: every repo method takes user_id (visitor id) and filters on it. Free hosting has ephemeral disk, so the demo
  tracker is temporary; users keep data via Export/Import. For durable data set DATABASE_URL to a hosted Postgres.
- Reminders are shown in the app and exported as .ics; email/Telegram push needs a scheduler + stored contacts (not on free hosting).
- Dates use TRACKER_TIMEZONE (default Asia/Kolkata); tzdata is in requirements for Windows.
- Gap-advice idea for Part 6: group related gaps (FAISS/Chroma/vector databases) so one advice item covers a family.

- Discovery: no scraping of LinkedIn/Naukri/Indeed. Adzuna returns only a truncated description, so 'Quick fit' is an
  ESTIMATE (60% your-skills-mentioned + 40% semantic, SEM_LOW/SEM_HIGH in discovery.py are rough and need tuning on real data).
  Server Adzuna keys come from secrets (ADZUNA_APP_ID/KEY/COUNTRY); visitors can use their own keys. Demo searches are capped
  (DISCOVERY_SESSION_LIMIT/DAILY_LIMIT); cache hits don't count. Adzuna terms: show 'Jobs by Adzuna' and use redirect_url.
- History timestamps are shown in TRACKER_TIMEZONE via to_local().

- Agent: max 2 tailor attempts and 2 email attempts (worst case ~7 Gemini calls, typical 4-6). Untested against real Gemini;
  the AI reviewer and writer quality must be judged on real runs. Part 8 adds an evaluation set.
- Paused runs live in an InMemorySaver (cache_resource): lost on app restart, deleted when a run ends or on 'Start over'.

## Current file list
job_agent/{config,llm,schemas,prompts,jd_parser,session,resume_loader,privacy,resume_schema,
           chunking,embeddings,vector_store,resume_parser,resume_index,matching,gap_advice,limits,demo,tracker_logic,tracker_db,ui_tracker,discovery,ui_discovery,verify,agent_prompts,agent,ui_agent}.py
app.py (Streamlit), data/demo/profile.json, .streamlit/secrets.toml.example
scripts/{part1_demo,part2_demo,part3_demo,list_models,probe_models}.py
tests/{test_jd_parser,test_llm_fallback,test_resume_pipeline,test_matching,test_limits_and_keys,test_demo,test_app,test_tracker_logic,test_tracker_db,test_discovery,test_verify,test_agent}.py
data/sample_resume.txt, data/sample_jds/ds_intern.txt
