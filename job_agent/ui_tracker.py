"""Streamlit 'Tracker' tab: reminders, add/edit applications, stats, export/import."""

import streamlit as st
from pydantic import ValidationError

from job_agent.tracker_db import TrackerError, TrackerRepo
from job_agent.tracker_logic import (
    NOT_APPLIED,
    STATUS_LABELS,
    TERMINAL,
    ApplicationInput,
    Status,
    follow_up_ics,
    to_local,
)

MAX_IMPORT_BYTES = 5 * 1024 * 1024
STATUSES = list(Status)


def label(status: Status | None) -> str:
    return STATUS_LABELS[status] if status else "-"


def flash(kind: str, message: str) -> None:
    """Show a message on the next run (st.rerun() would otherwise erase it)."""
    st.session_state["flash"] = (kind, message)


def _show_flash() -> None:
    item = st.session_state.pop("flash", None)
    if item:
        getattr(st, item[0])(item[1])


def first_error(exc: ValidationError) -> str:
    err = exc.errors()[0]
    field = ".".join(str(part) for part in err["loc"])
    return f"{field}: {err['msg']}" if field else err["msg"]


def _fmt_date(value) -> str:
    return f"{value:%d %b %Y}" if value else "-"


# --------------------------------------------------------------------------
def _render_due(repo: TrackerRepo, user_id: str, today) -> None:
    due = repo.due_reminders(user_id, today)
    if not due:
        return
    st.warning(f"⏰ {len(due)} follow-up(s) due")
    for app in due:
        days_late = (today - app.next_follow_up).days
        when = "due today" if days_late == 0 else f"{days_late} day(s) overdue"
        left, middle, right = st.columns([5, 2, 2])
        left.write(f"**{app.company}**, {app.role} ({label(app.status)}): {when}")
        if app.follow_up_count >= 2:
            left.caption("You've followed up 2+ times. Consider marking it as 'No response (ghosted)'.")
        if middle.button("Followed up (+7 days)", key=f"fu_{app.id}"):
            repo.mark_followed_up(user_id, app.id, today)
            flash("success", f"Reminder for {app.company} moved 7 days ahead.")
            st.rerun()
        right.download_button("Add to calendar", data=follow_up_ics(app),
                              file_name=f"follow-up-{app.id}.ics", mime="text/calendar",
                              key=f"ics_{app.id}")


def _render_metrics(repo: TrackerRepo, user_id: str, today) -> None:
    stats = repo.stats(user_id, today)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tracked", stats["total"])
    c2.metric("Applied", stats["applied"])
    c3.metric("Interviews", stats["interviewed"])
    c4.metric("Response rate", "n/a" if stats["response_rate"] is None else f"{stats['response_rate']:.0%}")


def _render_add_form(repo: TrackerRepo, user_id: str, today, expanded: bool) -> None:
    with st.expander("➕ Add an application", expanded=expanded):
        with st.form("add_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            company = c1.text_input("Company *", key="add_company")
            role = c2.text_input("Role *", key="add_role")
            status = c1.selectbox("Status", STATUSES, format_func=label, key="add_status")
            applied_on = c2.date_input("Applied on (if applied)", value=today, key="add_applied")
            url = c1.text_input("Job posting URL", key="add_url")
            resume = c2.text_input("Resume version label", key="add_resume",
                                   help="e.g. 'v1 data-science'. Lets you compare which resume gets replies.")
            contact_name = c1.text_input("Contact name", key="add_contact_name")
            contact_email = c2.text_input("Contact email", key="add_contact_email")
            notes = st.text_area("Notes", key="add_notes")
            submitted = st.form_submit_button("Add application", key="add_submit")
    if submitted:
        try:
            data = ApplicationInput(
                company=company, role=role, status=status, source_url=url, resume_version=resume,
                contact_name=contact_name, contact_email=contact_email, notes=notes,
                applied_on=None if status in NOT_APPLIED else applied_on,
            )
            repo.add(user_id, data, today)
        except ValidationError as exc:
            st.error(first_error(exc))
        except TrackerError as exc:
            st.error(str(exc))
        else:
            flash("success", f"Added {company.strip()} - {role.strip()}.")
            st.rerun()


def _render_manage(repo: TrackerRepo, user_id: str, today, apps) -> None:
    by_id = {a.id: a for a in apps}
    selected = st.selectbox("Select an application to update", list(by_id), key="tr_select",
                            format_func=lambda i: f"{by_id[i].company} - {by_id[i].role}")
    app = by_id[selected]

    if app.source_url:
        st.markdown(f"[Open the job posting]({app.source_url})")

    left, right = st.columns(2)
    new_status = left.selectbox("Status", STATUSES, index=STATUSES.index(app.status),
                                format_func=label, key=f"tr_status_{app.id}")
    note = right.text_input("Note (optional)", key=f"tr_note_{app.id}")
    if st.button("Update status", key=f"tr_update_{app.id}"):
        repo.change_status(user_id, app.id, new_status, note, today)
        flash("success", f"{app.company}: status set to {label(new_status)}.")
        st.rerun()

    with st.form(f"tr_edit_{app.id}"):
        c1, c2 = st.columns(2)
        no_reminder = c1.checkbox("No follow-up reminder", key=f"tr_noremind_{app.id}",
                                  value=app.next_follow_up is None or app.status in TERMINAL)
        follow_date = c2.date_input("Next follow-up", value=app.next_follow_up or today, key=f"tr_fu_{app.id}")
        contact_name = c1.text_input("Contact name", value=app.contact_name or "", key=f"tr_cn_{app.id}")
        contact_email = c2.text_input("Contact email", value=app.contact_email or "", key=f"tr_ce_{app.id}")
        url = c1.text_input("Job posting URL", value=app.source_url or "", key=f"tr_url_{app.id}")
        resume = c2.text_input("Resume version label", value=app.resume_version or "", key=f"tr_rv_{app.id}")
        notes = st.text_area("Notes", value=app.notes or "", key=f"tr_notes_{app.id}")
        saved = st.form_submit_button("Save changes", key=f"tr_save_{app.id}")
    if saved:
        updates = {
            "contact_name": contact_name, "contact_email": contact_email, "source_url": url,
            "resume_version": resume, "notes": notes,
            "next_follow_up": None if (no_reminder or app.status in TERMINAL) else follow_date,
        }
        try:
            repo.update_fields(user_id, app.id, updates)
        except ValidationError as exc:
            st.error(first_error(exc))
        except TrackerError as exc:
            st.error(str(exc))
        else:
            flash("success", "Changes saved.")
            st.rerun()

    with st.expander("Status history"):
        for change in app.history:
            before = label(change.from_status) if change.from_status else "New"
            line = f"{to_local(change.changed_at):%d %b %Y %H:%M}: {before} → {label(change.to_status)}"
            st.write(line + (f" ({change.note})" if change.note else ""))

    with st.expander("Delete this application"):
        confirm = st.checkbox("Yes, delete it permanently", key=f"tr_confirm_{app.id}")
        if st.button("Delete", key=f"tr_delete_{app.id}", disabled=not confirm):
            repo.delete(user_id, app.id)
            st.session_state.pop("tr_select", None)
            flash("success", f"Deleted {app.company}.")
            st.rerun()


def _render_stats(repo: TrackerRepo, user_id: str, today) -> None:
    stats = repo.stats(user_id, today)
    with st.expander("📊 Pipeline stats"):
        st.bar_chart([{"status": label(s), "count": n} for s, n in stats["by_status"].items()],
                     x="status", y="count")
        st.markdown("**Applications per week (last 8 weeks)**")
        st.bar_chart([{"week": f"{w['week_start']:%Y-%m-%d}", "applications": w["applications"]}
                      for w in stats["weekly"]], x="week", y="applications")
        if stats["by_resume_version"]:
            st.markdown("**Which resume version gets replies?**")
            st.dataframe([
                {"Resume version": name, "Applied": v["applied"], "Replies": v["responded"],
                 "Response rate": "n/a" if v["response_rate"] is None else f"{v['response_rate']:.0%}"}
                for name, v in stats["by_resume_version"].items()
            ], hide_index=True)
            st.caption("A reply is any assessment, interview, offer or rejection. Small numbers are noisy.")


def _render_export_import(repo: TrackerRepo, user_id: str, today, has_apps: bool) -> None:
    with st.expander("💾 Export / import / delete my data"):
        st.download_button("⬇ Export my tracker (JSON)", data=repo.export_json(user_id, today),
                           file_name=f"tracker-{today}.json", mime="application/json",
                           key="tr_export", disabled=not has_apps)
        uploaded = st.file_uploader("Import a tracker file (adds to your current list)", type=["json"],
                                    key="tr_import")
        if st.button("Import", key="tr_import_btn", disabled=uploaded is None):
            if uploaded.size > MAX_IMPORT_BYTES:
                st.error("File is too large (max 5 MB).")
            else:
                try:
                    count = repo.import_json(user_id, uploaded.getvalue().decode("utf-8"))
                except ValidationError as exc:
                    st.error(f"Not a valid tracker file: {first_error(exc)}")
                except (TrackerError, UnicodeDecodeError) as exc:
                    st.error(f"Could not import: {exc}")
                else:
                    flash("success", f"Imported {count} application(s).")
                    st.rerun()
        confirm = st.checkbox("I want to delete all my tracked applications", key="tr_wipe_confirm")
        if st.button("Delete all my tracker data", key="tr_wipe_btn", disabled=not confirm):
            count = repo.delete_all(user_id)
            st.session_state.pop("tr_select", None)
            flash("success", f"Deleted {count} application(s).")
            st.rerun()


def render_tracker(repo: TrackerRepo, user_id: str, today) -> None:
    _show_flash()
    st.caption("Private to this browser session. On the free demo server your data is temporary, "
               "so use Export to keep a copy.")
    apps = repo.list_applications(user_id)
    _render_due(repo, user_id, today)
    if apps:
        _render_metrics(repo, user_id, today)
    _render_add_form(repo, user_id, today, expanded=not apps)

    if not apps:
        st.info("No applications yet. Add one above, or analyze a job on the Match tab and "
                "click 'Save to my tracker'.")
    else:
        st.markdown("#### Your applications")
        choice = st.selectbox("Filter by status", ["All"] + STATUSES, key="tr_filter",
                              format_func=lambda s: "All" if s == "All" else label(s))
        shown = apps if choice == "All" else [a for a in apps if a.status == choice]
        st.dataframe([
            {"Company": a.company, "Role": a.role, "Status": label(a.status),
             "Applied": _fmt_date(a.applied_on), "Next follow-up": _fmt_date(a.next_follow_up),
             "Match": f"{a.match_score}/100" if a.match_score is not None else "-",
             "Resume": a.resume_version or ""}
            for a in shown
        ], hide_index=True)
        _render_manage(repo, user_id, today, apps)
        _render_stats(repo, user_id, today)
    _render_export_import(repo, user_id, today, bool(apps))
