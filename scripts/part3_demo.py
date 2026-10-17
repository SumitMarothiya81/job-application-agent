"""Run:
  python -m scripts.part3_demo --session <id>                 (reuse a Part 2 index; fewest API calls)
  python -m scripts.part3_demo                                (index the sample resume first)
  python -m scripts.part3_demo --resume data/resume/my.pdf --jd data/sample_jds/ds_intern.txt
  add --advice to also ask Gemini for skill-gap advice (one extra API call)
"""

import argparse
import logging
from pathlib import Path

from job_agent.gap_advice import advise_gaps
from job_agent.jd_parser import parse_jd
from job_agent.matching import SkillStatus, match_resume_to_jd
from job_agent.resume_index import build_resume_index, load_profile, load_store
from job_agent.session import new_session_id

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

TAG = {SkillStatus.EXPLICIT: "[OK] ", SkillStatus.RELATED: "[~~] ", SkillStatus.MISSING: "[--] "}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default="data/sample_resume.txt")
    parser.add_argument("--jd", default="data/sample_jds/ds_intern.txt")
    parser.add_argument("--session", default=None)
    parser.add_argument("--advice", action="store_true")
    args = parser.parse_args()

    if args.session:
        session_id = args.session
        profile = load_profile(session_id)
    else:
        session_id = new_session_id()
        path = Path(args.resume)
        profile = build_resume_index(path.read_bytes(), path.name, session_id)
        print(f"Indexed resume. Session id: {session_id}")
    store = load_store(session_id)

    jd = parse_jd(Path(args.jd).read_text(encoding="utf-8"))
    report = match_resume_to_jd(profile, store, jd)

    print(f"\n{report.job_title} @ {report.company or 'unknown company'}")
    print(f"MATCH SCORE: {report.overall_score}/100  ({report.verdict})")
    for label, cov in (("required", report.required_coverage), ("preferred", report.preferred_coverage)):
        if cov is not None:
            print(f"  {label} coverage: {cov:.0%}")
    for note in report.fit_notes:
        print(f"  NOTE: {note}")

    print("\n[OK] explicit   [~~] related   [--] missing")
    for m in report.matches:
        via = f" (via {m.related_via})" if m.related_via else ""
        print(f"{TAG[m.status]}{m.kind[:3]}  {m.skill:<28} sem {m.score:.2f}{via}")
        for ev in m.evidence[:1]:
            print(f"        -> {ev.source}: {ev.text[:80]}")

    print(f"\nATS keywords found  : {', '.join(report.ats_hits) or '-'}")
    print(f"ATS keywords missing: {', '.join(report.ats_missing) or '-'}")

    if args.advice:
        plan = advise_gaps(report, jd, profile.known_skills())
        print("\nGAP ADVICE")
        for item in plan.items:
            print(f"- {item.skill} [{item.priority}]")
            print(f"    learn  : {item.how_to_close}")
            print(f"    project: {item.project_idea}")


if __name__ == "__main__":
    main()
