"""Job Application Agent: Streamlit app (Part 3b).

Run locally:  streamlit run app.py
Two modes:
  Demo - fictional sample resume, server's Gemini key, strict usage caps.
  Own key - visitor's Gemini key; they can upload their own resume.
"""

import os
import uuid

import streamlit as st

st.set_page_config(page_title="Job Application Agent", page_icon="🎯", layout="wide")

# Streamlit secrets -> environment, BEFORE importing job_agent (config reads env at import).
for _name in ("GOOGLE_API_KEY", "GEMINI_MODELS", "DEMO_SESSION_LIMIT", "DEMO_DAILY_LIMIT"):
    try:
        if _name in st.secrets:
            os.environ.setdefault(_name, str(st.secrets[_name]))
    except Exception:  # no secrets file locally: fall back to .env  # noqa: BLE001
        pass

from job_agent.demo import build_store, load_demo_profile  # noqa: E402
from job_agent.gap_advice import advise_gaps  # noqa: E402
from job_agent.jd_parser import parse_jd  # noqa: E402
from job_agent.limits import UsageLimiter  # noqa: E402
from job_agent.llm import use_api_key  # noqa: E402
from job_agent.matching import SkillStatus, match_resume_to_jd  # noqa: E402
from job_agent.resume_index import build_resume_index, load_profile, load_store  # noqa: E402
from job_agent.resume_loader import ResumeLoadError  # noqa: E402
from job_agent.session import delete_session, new_session_id  # noqa: E402

SAMPLE_JD = """Data Science Intern - Acme Analytics, Bengaluru (Hybrid)

We are looking for a Data Science Intern to join our GenAI team for 6 months.

What you will do
- Build and evaluate LLM-powered features such as document Q&A
- Clean and analyse datasets using Python and SQL
- Create dashboards to present insights to stakeholders

Requirements
- Strong Python and SQL skills
- Hands-on experience with pandas and scikit-learn
- Understanding of machine learning fundamentals

Nice to have
- Experience with LangChain or LangGraph
- Knowledge of vector databases (FAISS, Chroma)
- Familiarity with Docker and Git
- Power BI or Tableau
"""

STATUS_LABEL = {
    SkillStatus.EXPLICIT: "✅ on your resume",
    SkillStatus.RELATED: "🟡 related",
    SkillStatus.MISSING: "❌ missing",
}


@st.cache_resource
def get_demo():
    profile = load_demo_profile()
    return profile, build_store(profile)


@st.cache_resource
def get_limiter() -> UsageLimiter:
    return UsageLimiter(
        per_session=int(os.getenv("DEMO_SESSION_LIMIT", "5")),
        per_day=int(os.getenv("DEMO_DAILY_LIMIT", "100")),
    )


def friendly_error(exc: Exception) -> str:
    text = str(exc)
    if "API key not valid" in text or "API_KEY_INVALID" in text:
        return "Google rejected that API key. Check that you copied it from Google AI Studio."
    if "All configured models failed" in text:
        return "Gemini is busy right now. Please try again in a minute."
    if "GOOGLE_API_KEY is missing" in text:
        return "The demo key isn't configured on this server. Use 'My own Gemini key' instead."
    return f"Something went wrong: {text[:200]}"


st.session_state.setdefault("visitor_id", uuid.uuid4().hex)
st.session_state.setdefault("result", None)

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🎯 Job Application Agent")
mode = st.sidebar.radio(
    "Gemini access", ["Demo (sample resume)", "My own Gemini key"], key="mode",
    help="Demo uses a fictional resume and the app's key, with usage limits.",
)
byo = mode == "My own Gemini key"
user_key = None
remaining_slot = None
if byo:
    user_key = st.sidebar.text_input("Gemini API key", type="password", key="user_key",
                                     help="Kept in this browser session only; never stored.")
    uploaded = st.sidebar.file_uploader("Your resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt", "md"])
    if st.sidebar.button("Index my resume", key="index_btn", disabled=not (uploaded and user_key)):
        sid = new_session_id()
        try:
            with st.spinner("Reading and indexing your resume..."), use_api_key(user_key):
                build_resume_index(uploaded.getvalue(), uploaded.name, sid)
            st.session_state["my_session_id"] = sid
            st.sidebar.success("Resume indexed.")
        except ResumeLoadError as exc:
            st.sidebar.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(friendly_error(exc))
    if st.session_state.get("my_session_id") and st.sidebar.button("Delete my data", key="delete_btn"):
        delete_session(st.session_state.pop("my_session_id"))
        st.session_state["result"] = None
        st.sidebar.info("Your resume data was deleted from this server.")
    st.sidebar.caption(
        "Before anything is sent to Gemini, your email, phone, profile links and name line "
        "are masked. Free-tier Gemini keys may let Google use prompts to improve its models; "
        "a paid key does not."
    )
else:
    remaining_slot = st.sidebar.empty()

# ---------------------------------------------------------------- resume source
my_sid = st.session_state.get("my_session_id") if byo else None
if my_sid:
    profile, store = load_profile(my_sid), load_store(my_sid)
    source_note = "Using **your** uploaded resume."
else:
    profile, store = get_demo()
    source_note = "Using the **fictional sample resume** (Riya Verma, final-year B.Tech)."

# --------------------------------------------------------------------- main page
st.title("Does this job fit your resume?")
st.caption(source_note + " Paste any job description below.")

st.button("Load sample job description", key="sample_btn",
          on_click=lambda: st.session_state.update(jd_text=SAMPLE_JD))
st.text_area("Job description", key="jd_text", height=280)
want_advice = st.checkbox("Also suggest how to close my skill gaps (one extra Gemini call)", key="advice_box")

if st.button("Analyze match", type="primary", key="analyze_btn"):
    jd_text = st.session_state.get("jd_text", "")
    if len(jd_text.strip()) < 40:
        st.warning("Please paste a longer job description.")
    elif byo and not user_key:
        st.warning("Enter your Gemini API key in the sidebar first.")
    else:
        allowed, message = (True, "")
        if not byo:
            allowed, message = get_limiter().try_consume(
                st.session_state["visitor_id"], units=2 if want_advice else 1)
        if not allowed:
            st.warning(message)
        else:
            try:
                with st.spinner("Analyzing..."), use_api_key(user_key if byo else None):
                    jd = parse_jd(jd_text)
                    report = match_resume_to_jd(profile, store, jd)
                    plan = advise_gaps(report, jd, profile.known_skills()) if want_advice else None
                st.session_state["result"] = (report, plan)
            except Exception as exc:  # noqa: BLE001
                st.error(friendly_error(exc))
if remaining_slot is not None:
    remaining_slot.caption(
        f"Demo runs left this session: {get_limiter().remaining(st.session_state['visitor_id'])}"
    )
result = st.session_state.get("result")
if result:
    report, plan = result
    st.subheader(f"{report.job_title} @ {report.company or 'unknown company'}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Match score", f"{report.overall_score}/100")
    c2.metric("Required skills covered",
              "n/a" if report.required_coverage is None else f"{report.required_coverage:.0%}")
    c3.metric("Preferred skills covered",
              "n/a" if report.preferred_coverage is None else f"{report.preferred_coverage:.0%}")
    st.write(f"Verdict: **{report.verdict}**")
    for note in report.fit_notes:
        st.warning(note)

    st.markdown("#### Skills")
    st.dataframe(
        [{"Skill": m.skill, "Type": m.kind, "Status": STATUS_LABEL[m.status],
          "Via": m.related_via or ("similar bullet" if m.status == SkillStatus.RELATED else "")} for m in report.matches],
        use_container_width=True, hide_index=True,
    )
    with st.expander("Evidence from your resume"):
        for m in report.matches:
            if m.evidence:
                st.markdown(f"**{m.skill}**")
                for ev in m.evidence[:2]:
                   st.write(f"- _{ev.source}_: {ev.text} (similarity {ev.score:.2f})")

    st.markdown("#### ATS keywords")
    k1, k2 = st.columns(2)
    k1.write("**Already on your resume:** " + (", ".join(report.ats_hits) or "-"))
    k2.write("**Not on your resume:** " + (", ".join(report.ats_missing) or "-"))

    if plan and plan.items:
        st.markdown("#### How to close the gaps")
        for item in plan.items:
            with st.expander(f"{item.skill} ({item.priority} priority)"):
                st.write(f"**Learn:** {item.how_to_close}")
                st.write(f"**Project idea:** {item.project_idea}")

st.divider()
st.caption("Matching is a guide, not a verdict. You stay in control of every application you send.")
