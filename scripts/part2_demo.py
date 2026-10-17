"""Run:
  python -m scripts.part2_demo                       (uses the sample resume + sample JD)
  python -m scripts.part2_demo --resume data/resume/my_resume.pdf --jd data/sample_jds/ds_intern.txt
  python -m scripts.part2_demo --session <id> --jd ...   (reuse an existing index, no resume parsing)
"""

import argparse
import logging
from pathlib import Path

from job_agent.jd_parser import parse_jd
from job_agent.privacy import redact_pii
from job_agent.resume_index import (
    build_resume_index,
    load_profile,
    load_store,
    retrieve_evidence,
)
from job_agent.resume_loader import load_resume_file
from job_agent.session import new_session_id

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default="data/sample_resume.txt")
    parser.add_argument("--jd", default="data/sample_jds/ds_intern.txt")
    parser.add_argument("--session", default=None, help="reuse an existing session id")
    args = parser.parse_args()

    # 1) Build (or load) the resume index
    if args.session:
        session_id = args.session
        profile = load_profile(session_id)
        print(f"Loaded existing session {session_id}")
    else:
        session_id = new_session_id()
        path = Path(args.resume)
        print("What the LLM will see (PII redacted), first lines:")
        print("\n".join(redact_pii(load_resume_file(path)).splitlines()[:4]), "\n")
        profile = build_resume_index(path.read_bytes(), path.name, session_id)
        print(f"Indexed resume. Session id: {session_id}")

    store = load_store(session_id)
    print(
        f"Profile: {len(profile.projects)} projects, {len(profile.experience)} experience, "
        f"{len(profile.skills)} skills -> {len(store.chunks)} evidence chunks\n"
    )

    # 2) Parse the JD (Part 1) and look for resume evidence for each skill
    jd = parse_jd(Path(args.jd).read_text(encoding="utf-8"))
    known = set(profile.known_skills())

    for label, skills in (("REQUIRED", jd.required_skills), ("PREFERRED", jd.preferred_skills)):
        print(f"--- {label} skills: best resume evidence ---")
        for skill in skills:
            (chunk, score), *_ = retrieve_evidence(
                session_id, f"experience with {skill}", k=1, store=store
            )
            listed = "listed" if skill in known else "NOT listed"
            print(f"{skill:<28} [{listed:>10}]  score {score:.2f}  {chunk.source}")
            print(f"{'':<28} -> {chunk.text[:90]}")
        print()


if __name__ == "__main__":
    main()
