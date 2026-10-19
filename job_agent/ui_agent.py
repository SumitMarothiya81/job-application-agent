"""Streamlit 'Apply assistant' tab: runs the LangGraph agent and pauses for YOUR decisions."""

import uuid

import streamlit as st
from langgraph.types import Command
from pydantic import ValidationError

from job_agent import agent as agent_module
from job_agent.limits import UsageLimiter
from job_agent.llm import use_api_key
from job_agent.tracker_db import TrackerError, TrackerRepo
from job_agent.tracker_logic import ApplicationInput, Status
from job_agent.ui_tracker import first_error, flash, label

STATE_KEY = "agent"


def _state() -> dict:
    return st.session_state.setdefault(STATE_KEY, {"stage": "idle"})


def _reset(checkpointer) -> None:
    state = st.session_state.get(STATE_KEY) or {}
    if state.get("run_id"):
        try:
            checkpointer.delete_thread(state["run_id"])  # free the saved run
        except Exception:  # noqa: BLE001
            pass
    st.session_state[STATE_KEY] = {"stage": "idle"}


def _copy_from_match() -> None:
    st.session_state["agent_jd"] = st.session_state.get("jd_text", "")


def _advance(graph_input, profile, store, checkpointer, api_key, friendly_error) -> None:
    """Run the graph until it finishes or asks the user something, then rerun the page."""
    state = _state()
    graph = agent_module.build_graph(profile, store, agent_module.default_deps(), checkpointer)
    config = {"configurable": {"thread_id": state["run_id"]}}
    try:
        with st.spinner("Working on it (this can take up to a minute)..."), use_api_key(api_key):
            result = graph.invoke(graph_input, config)
    except Exception as exc:  # noqa: BLE001
        _reset(checkpointer)
        st.session_state["agent_error"] = friendly_error(exc)
        st.rerun()
        return
    state["log"] = result.get("log", state.get("log", []))
    question = agent_module.pending_interrupt(graph, config)
    if question:
        state.update(stage=question["type"], payload=question)
    else:
        state.update(stage="done", package=result["package"])
        try:
            checkpointer.delete_thread(state["run_id"])
        except Exception:  # noqa: BLE001
            pass
    st.rerun()


def _render_log() -> None:
    log = _state().get("log")
    if log:
        with st.expander("What the assistant did"):
            for line in log:
                st.write("• " + line)


# --------------------------------------------------------------------------
def _render_start(profile, store, limiter, user_id, byo, api_key, checkpointer, friendly_error) -> None:
    st.text_area("Job description", key="agent_jd", height=220,
                 placeholder="Paste the full job description here...")
    st.button("Copy from the Match tab", key="agent_copy", on_click=_copy_from_match)
    if st.button("Start the assistant", type="primary", key="agent_start"):
        jd_text = st.session_state.get("agent_jd", "")
        if len(jd_text.strip()) < 40:
            st.warning("Please paste a longer job description.")
        elif byo and not api_key:
            st.warning("Enter your Gemini API key in the sidebar first.")
        else:
            allowed, message = (True, "")
            if not byo:
                allowed, message = limiter.try_consume(user_id)
            if not allowed:
                st.warning(message.replace("Demo limit reached for this session", "Demo assistant limit reached"))
            else:
                st.session_state[STATE_KEY] = {"stage": "starting", "run_id": uuid.uuid4().hex, "jd_text": jd_text}
                _advance({"jd_text": jd_text, "log": []}, profile, store, checkpointer, api_key, friendly_error)


def _render_bullet_review(profile, store, checkpointer, api_key, friendly_error) -> None:
    state = _state()
    items = state["payload"]["items"]
    st.markdown("#### Step 1 of 2: review the rewrites")
    st.caption("Every rewrite below passed two fact-checks (no new numbers, no new tools, and an AI reviewer). "
               "You decide which ones to use. Your original bullets are never changed.")
    with st.form("agent_bullets_form"):
        for item in items:
            with st.container(border=True):
                st.markdown(f"**{item['source']}**")
                st.markdown(f"Original: {item['original']}")
                if item["status"] == "pass" and item["changed"]:
                    st.markdown(f"**Suggested:** {item['rewritten']}")
                    st.checkbox("Use the suggested version", value=True, key=f"agent_acc_{item['id']}")
                    for warning in item["warnings"]:
                        st.caption(f"⚠️ {warning}")
                elif item["status"] == "pass":
                    st.caption("The assistant suggests keeping this one as it is.")
                else:
                    st.caption("🚫 The rewrite was rejected, so your original stays: " + "; ".join(item["issues"]))
        submitted = st.form_submit_button("Continue", key="agent_bullets_submit")
    if submitted:
        accept = [i["id"] for i in items if i["status"] == "pass" and i["changed"]
                  and st.session_state.get(f"agent_acc_{i['id']}", True)]
        _advance(Command(resume={"accept": accept}), profile, store, checkpointer, api_key, friendly_error)


def _render_email_review(profile, store, checkpointer, api_key, friendly_error) -> None:
    payload = _state()["payload"]
    st.markdown("#### Step 2 of 2: review the email")
    if payload["origin"] == "template":
        st.info("The AI drafts kept mentioning things that aren't on your resume, so this is a plain, safe template. "
                "Edit it freely.")
    st.caption("Replace [Recruiter name] and [Your name]. Nothing is sent from here: copy it into your own email.")
    with st.form("agent_email_form"):
        subject = st.text_input("Subject", value=payload["subject"], key="agent_email_subject")
        body = st.text_area("Email", value=payload["body"], height=260, key="agent_email_body")
        c1, c2 = st.columns(2)
        approve = c1.form_submit_button("Keep this email", key="agent_email_approve")
        skip = c2.form_submit_button("Skip the email", key="agent_email_skip")
    if approve or skip:
        reply = {"approved": True, "subject": subject, "body": body} if approve else {"approved": False}
        _advance(Command(resume=reply), profile, store, checkpointer, api_key, friendly_error)


def _render_done(repo: TrackerRepo, user_id, today, checkpointer, default_label) -> None:
    state = _state()
    package = state["package"]
    st.success("Your application kit is ready. Nothing has been sent or submitted: you stay in control.")
    c1, c2 = st.columns(2)
    c1.metric("Match score", f"{package['match_score']}/100")
    c2.metric("Rewrites you accepted", len(package["bullets"]))

    st.markdown("#### Tailored bullets")
    if package["bullets"]:
        for item in package["bullets"]:
            st.markdown(f"- {item['rewritten']}")
            st.caption(f"was ({item['source']}): {item['original']}")
    else:
        st.write("No rewrites were accepted, so your resume stays as it is.")

    st.markdown("#### Email draft")
    if package["email"]:
        st.write(f"**Subject:** {package['email']['subject']}")
        st.code(package["email"]["body"], language=None)
    else:
        st.write("No email was kept.")
    for warning in package["warnings"]:
        st.warning(warning)

    st.download_button("⬇ Download the kit (Markdown)", data=package["markdown"],
                       file_name="application-kit.md", mime="text/markdown", key="agent_download")

    with st.expander("Save this to my tracker"):
        with st.form("agent_save_form"):
            s1, s2 = st.columns(2)
            company = s1.text_input("Company", value=package.get("company") or "", key="agent_sv_company")
            role = s2.text_input("Role", value=package["job_title"], key="agent_sv_role")
            version = s1.text_input("Resume version label", value=package["resume_version"], key="agent_sv_version")
            status = s2.selectbox("Status", [Status.SAVED, Status.APPLIED], format_func=label, key="agent_sv_status",
                                  help="Choose 'Applied' only after you have actually applied.")
            saved = st.form_submit_button("Save to my tracker", key="agent_sv_btn")
        if saved:
            try:
                repo.add(user_id, ApplicationInput(
                    company=company, role=role, status=status, resume_version=version,
                    match_score=package["match_score"], jd_text=state.get("jd_text"),
                    notes=f"Tailored {len(package['bullets'])} bullet(s)"
                          + ("; email drafted." if package["email"] else ".")), today)
            except ValidationError as exc:
                st.error(first_error(exc))
            except TrackerError as exc:
                st.error(str(exc))
            else:
                flash("success", f"Saved {company.strip()} - {role.strip()} to your tracker.")
                st.rerun()
    st.caption("Next: apply on the company's site yourself. Then mark it Applied in the tracker so the "
               "follow-up reminder starts.")


def render_agent(profile, store, repo: TrackerRepo, user_id: str, today, limiter: UsageLimiter,
                 checkpointer, byo: bool, api_key: str | None, friendly_error, default_label: str) -> None:
    st.subheader("Apply assistant")
    st.caption("Tailors your REAL bullets to a job and drafts a recruiter email. It can reword, but never invent: "
               "every rewrite is fact-checked against your resume, and you approve each step. "
               "Uses about 4 to 6 Gemini calls per run.")
    error = st.session_state.pop("agent_error", None)
    if error:
        st.error(error)

    stage = _state()["stage"]
    if stage == "idle":
        if not byo:
            st.caption(f"Demo assistant runs left this session: {limiter.remaining(user_id)}")
        _render_start(profile, store, limiter, user_id, byo, api_key, checkpointer, friendly_error)
        return

    if stage == "review_bullets":
        _render_bullet_review(profile, store, checkpointer, api_key, friendly_error)
    elif stage == "review_email":
        _render_email_review(profile, store, checkpointer, api_key, friendly_error)
    elif stage == "done":
        _render_done(repo, user_id, today, checkpointer, default_label)
    _render_log()
    if st.button("Start over", key="agent_reset"):
        _reset(checkpointer)
        st.rerun()
