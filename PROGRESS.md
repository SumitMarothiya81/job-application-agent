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
3b. [ ] Mini Streamlit page + first deploy (Streamlit Community Cloud / Render)
4. [ ] Database + application tracker + reminders
5. [ ] Job discovery (Adzuna, manual import, dedupe, ranking)
6. [ ] Smart assistant (LangGraph: tailor, verifier, email, human approval)
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

## Current file list
job_agent/{config,llm,schemas,prompts,jd_parser,session,resume_loader,privacy,resume_schema,
           chunking,embeddings,vector_store,resume_parser,resume_index,matching,gap_advice}.py
scripts/{part1_demo,part2_demo,part3_demo,list_models,probe_models}.py
tests/{test_jd_parser,test_llm_fallback,test_resume_pipeline,test_matching}.py
data/sample_resume.txt, data/sample_jds/ds_intern.txt
