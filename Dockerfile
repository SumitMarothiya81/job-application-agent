# Job Application Agent: one-command container.  Build: docker build -t job-agent .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    DATABASE_URL=sqlite:////app/state/tracker.db \
    DATA_DIR=/app/state/sessions

WORKDIR /app

# Dependencies first, so rebuilding after a code change is fast.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Only what the app needs (never the .env file, your resumes or local databases).
COPY app.py ./
COPY job_agent ./job_agent
COPY data/demo ./data/demo
COPY data/sample_jds ./data/sample_jds
COPY .streamlit/config.toml ./.streamlit/config.toml

# Bake the local embedding model into the image: the container starts fast and works offline for that step.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

# Run as a normal user, with one writable folder for the tracker database and uploaded-resume indexes.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/state \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
