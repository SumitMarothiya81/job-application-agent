"""Streamlit 'Discover' tab: search Adzuna or import a CSV, then rank, filter and save jobs."""

import streamlit as st
from pydantic import ValidationError

from job_agent.discovery import (
    AdzunaClient,
    DiscoveryError,
    SearchCache,
    listing_id,
    parse_listings_csv,
    rank_listings,
    tracked_keys,
)
from job_agent.limits import UsageLimiter
from job_agent.tracker_db import TrackerError, TrackerRepo
from job_agent.tracker_logic import ApplicationInput, Status
from job_agent.ui_tracker import first_error, flash

VERDICT_BADGE = {"fresher": "🟢 Fresher-friendly", "unclear": "⚪ Experience not stated",
                 "experienced": "🟠 Asks for experience"}


def _load_into_match(text: str) -> None:
    """Button callback: runs before the next script run, so it may set the text box's value."""
    st.session_state["jd_text"] = text
    st.session_state["disc_loaded_notice"] = True


def _pick_client(server_client: AdzunaClient | None, cache: SearchCache, country: str):
    """The visitor's own Adzuna keys win; otherwise the server's (limited) keys."""
    own_id, own_key = st.session_state.get("adz_id", "").strip(), st.session_state.get("adz_key", "").strip()
    if own_id and own_key:
        try:
            return AdzunaClient(own_id, own_key, country=country, cache=cache), True
        except DiscoveryError as exc:
            st.error(str(exc))
            return None, True
    return server_client, False


def _search_form(server_client, limiter: UsageLimiter | None, cache: SearchCache, country: str, user_id: str):
    with st.expander("Use my own Adzuna keys (optional)", expanded=server_client is None):
        st.caption("Free keys from developer.adzuna.com. They stay in this browser session only, and "
                   "searches then use your own monthly quota instead of the demo's.")
        st.text_input("Adzuna app ID", key="adz_id")
        st.text_input("Adzuna app key", type="password", key="adz_key")

    with st.form("disc_search_form"):
        what = st.text_input("Keywords", value="data science intern", key="disc_what")
        where = st.text_input("City (optional)", key="disc_where")
        c1, c2 = st.columns(2)
        days = c1.slider("Posted within (days)", 1, 30, 14, key="disc_days")
        count = c2.selectbox("Results", [10, 20, 30, 50], index=1, key="disc_count")
        submitted = st.form_submit_button("Search jobs", key="disc_search_btn")

    if not submitted:
        return
    client, own_keys = _pick_client(server_client, cache, country)
    if client is None:
        if server_client is None and not own_keys:
            st.warning("Job search isn't configured on this server. Add your own Adzuna keys above, "
                       "or import a CSV instead.")
        return
    if not own_keys and limiter is not None and limiter.remaining(user_id) <= 0:
        st.warning("Demo search limit reached. Add your own Adzuna keys above to keep searching.")
        return
    try:
        with st.spinner("Searching..."):
            result = client.search(what, where or None, results_per_page=count, max_days_old=days)
    except DiscoveryError as exc:
        st.error(str(exc))
        return
    if not own_keys and limiter is not None and not result.from_cache:
        limiter.try_consume(user_id)  # only real API calls use up the demo allowance
    st.session_state["disc_listings"] = result.listings
    st.session_state["disc_source_note"] = (
        f"{len(result.listings)} job(s) returned" + (f" of about {result.total_matches} matches" if result.total_matches else "")
        + (" (from cache)" if result.from_cache else "") + ".")


def _csv_form():
    st.caption("Collected jobs yourself (a spreadsheet, a saved search)? Upload a CSV with the columns "
               "`title`, `company` and optionally `location`, `url`, `description`. Max 200 rows.")
    uploaded = st.file_uploader("Jobs CSV", type=["csv"], key="disc_csv")
    if st.button("Import jobs", key="disc_csv_btn", disabled=uploaded is None):
        try:
            listings, warnings = parse_listings_csv(uploaded.getvalue())
        except DiscoveryError as exc:
            st.error(str(exc))
            return
        if not listings:
            st.warning("No usable rows found in that file.")
            return
        st.session_state["disc_listings"] = listings
        st.session_state["disc_source_note"] = f"{len(listings)} job(s) imported from your CSV." + "".join(
            f" {w}" for w in warnings)


def _save_to_tracker(repo, user_id, today, item) -> None:
    listing = item.listing
    try:
        repo.add(user_id, ApplicationInput(
            company=listing.company, role=listing.title, status=Status.SAVED,
            source_url=listing.url, location=listing.location, jd_text=listing.description or None,
            notes=f"Found via {listing.source}. Quick fit estimate: {item.fit}/100 (from the snippet).",
        ), today)
    except ValidationError as exc:
        st.error(first_error(exc))
    except TrackerError as exc:
        st.error(str(exc))
    else:
        flash("success", f"Saved {listing.company} - {listing.title} to your tracker.")
        st.rerun()


def _render_results(profile, store, repo, user_id, today) -> None:
    listings = st.session_state.get("disc_listings")
    if listings is None:
        st.info("Search for jobs or import a CSV to see matches ranked against your resume.")
        return

    st.caption(st.session_state.get("disc_source_note", ""))
    o1, o2 = st.columns(2)
    include_experienced = o1.checkbox("Include roles that ask for experience", key="disc_include_exp")
    hide_tracked = o2.checkbox("Hide jobs already in my tracker", value=True, key="disc_hide_tracked")

    result = rank_listings(listings, profile, store, tracked_keys(repo.list_applications(user_id)),
                           include_experienced, hide_tracked)
    hidden = []
    if result.duplicates_removed:
        hidden.append(f"{result.duplicates_removed} duplicate(s) merged")
    if result.hidden_experienced:
        hidden.append(f"{result.hidden_experienced} experienced role(s) hidden")
    if result.hidden_tracked:
        hidden.append(f"{result.hidden_tracked} already in your tracker")
    st.markdown(f"**{len(result.ranked)} job(s) shown**" + (f" ({', '.join(hidden)})" if hidden else ""))
    st.caption("'Quick fit' is an estimate from the short job snippet. For an accurate score, open the "
               "posting, copy the full description and analyze it on the Match tab.")

    if st.session_state.pop("disc_loaded_notice", None):
        st.success("Loaded into the Match tab. Paste the full description there for an accurate score.")

    for item in result.ranked:
        listing, key = item.listing, listing_id(item.listing)
        with st.container(border=True):
            left, right = st.columns([5, 2])
            left.markdown(f"**{listing.title}**, {listing.company}")
            meta = [listing.location, f"posted {listing.posted_on:%d %b %Y}" if listing.posted_on else None]
            if listing.salary_min:
                meta.append(f"salary from {listing.salary_min:,.0f}")
            left.caption(" · ".join(m for m in meta if m))
            if listing.description:
                left.write(listing.description[:300] + ("…" if len(listing.description) > 300 else ""))
            if item.matched_skills:
                left.caption("Your skills it mentions: " + ", ".join(item.matched_skills))
            right.metric("Quick fit", f"{item.fit}/100")
            right.caption(f"{VERDICT_BADGE[item.verdict]}: {item.verdict_reason}")

            b1, b2, b3 = st.columns(3)
            if listing.url:
                b1.link_button("Open posting", listing.url)
            if b2.button("Save to tracker", key=f"disc_save_{key}"):
                _save_to_tracker(repo, user_id, today, item)
            b3.button("Analyze in Match tab", key=f"disc_an_{key}", on_click=_load_into_match,
                      args=(f"{listing.title} - {listing.company}, {listing.location or ''}\n\n{listing.description}",))

    if any(i.listing.source == "adzuna" for i in result.ranked):
        st.caption("Jobs by [Adzuna](https://www.adzuna.com). Links open the original posting through Adzuna.")


def render_discovery(profile, store, repo: TrackerRepo, user_id: str, today,
                     server_client: AdzunaClient | None, limiter: UsageLimiter | None,
                     cache: SearchCache, country: str = "in") -> None:
    st.subheader("Find jobs that fit your resume")
    st.caption("Jobs come from Adzuna's official API or a CSV you upload. Nothing is scraped from LinkedIn, "
               "Naukri or Indeed.")
    mode = st.radio("Where do the jobs come from?", ["Search Adzuna", "Import a CSV"],
                    horizontal=True, key="disc_mode")
    if mode == "Search Adzuna":
        _search_form(server_client, limiter, cache, country, user_id)
        if limiter is not None and server_client is not None:
            st.caption(f"Demo searches left this session: {limiter.remaining(user_id)}")
    else:
        _csv_form()
    _render_results(profile, store, repo, user_id, today)
