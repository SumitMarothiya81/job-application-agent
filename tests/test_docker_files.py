"""Guards for the container setup: secrets must never end up inside an image."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
IGNORE = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()


def test_dockerignore_keeps_secrets_and_personal_data_out():
    for entry in (".env", ".streamlit/secrets.toml", "data/resume/", "data/sessions/", "*.db"):
        assert entry in IGNORE, entry


def test_dockerfile_copies_only_named_files_never_everything():
    copies = re.findall(r"^COPY\s+(.+)$", DOCKERFILE, re.MULTILINE)
    assert copies and not any(re.match(r"\.\s", c) or c.startswith("./ ") for c in copies)   # no 'COPY . .'
    assert not any(".env" in c for c in copies)
    for line in copies:
        *sources, _destination = line.split()
        for source in sources:
            assert (ROOT / source.rstrip("/")).exists(), f"Dockerfile copies a missing path: {source}"


def test_container_is_configured_for_a_server():
    assert "USER appuser" in DOCKERFILE and "--server.address=0.0.0.0" in DOCKERFILE
    assert "EXPOSE 8501" in DOCKERFILE and "HEALTHCHECK" in DOCKERFILE
    assert "FASTEMBED_CACHE_PATH=/app/.cache/fastembed" in DOCKERFILE        # baked model is found at runtime
    assert "DATABASE_URL=sqlite:////app/state/tracker.db" in DOCKERFILE and "DATA_DIR=/app/state/sessions" in DOCKERFILE


def test_compose_mounts_one_state_volume_and_reads_env_at_runtime():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "127.0.0.1:8501:8501" in compose and "SINGLE_USER" in compose and "appstate:/app/state" in compose and "env_file" in compose
