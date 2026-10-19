"""Job discovery: Adzuna search, CSV import, de-duplication, fresher filter, quick ranking.

Deliberately NOT here: scraping LinkedIn/Naukri/Indeed pages (against their terms,
and brittle). Jobs come from Adzuna's official API or from a CSV the user prepares.
Adzuna only returns a short, truncated description, so every score in this module is
a quick ESTIMATE. The accurate match still needs the full text (Match tab).
"""

import csv
import hashlib
import html
import io
import re
import time
from dataclasses import dataclass
from datetime import date

import httpx
from pydantic import BaseModel, Field

from job_agent import embeddings
from job_agent.matching import _tokens
from job_agent.resume_schema import ResumeProfile
from job_agent.vector_store import VectorStore

ADZUNA_BASE = "https://api.adzuna.com/v1/api/jobs"
MAX_CSV_ROWS = 200
SORT_OPTIONS = {"date", "relevance", "salary"}


class DiscoveryError(Exception):
    """A problem that is safe to show to the user (never contains API keys)."""


# --------------------------------------------------------------------------
# Data model + cleaning
# --------------------------------------------------------------------------
class JobListing(BaseModel):
    source: str  # "adzuna" | "csv"
    external_id: str | None = None
    title: str = Field(min_length=1, max_length=200)
    company: str = Field(min_length=1, max_length=120)
    location: str | None = Field(default=None, max_length=120)
    description: str = Field(default="", max_length=3000)  # a snippet, not the full posting
    url: str | None = Field(default=None, max_length=500)
    posted_on: date | None = None
    salary_min: float | None = None
    salary_max: float | None = None


def _clean(text, limit: int) -> str:
    """Strip HTML tags (Adzuna highlights keywords with <strong>), decode entities, clip."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    return " ".join(text.split())[:limit]


def _safe_url(value) -> str | None:
    """Only plain http(s) links are ever shown or opened."""
    if isinstance(value, str) and re.match(r"^https?://[^\s<>\"']+$", value.strip(), re.IGNORECASE):
        return value.strip()[:500]
    return None


def _parse_date(value) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return None


def _number(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def parse_adzuna_results(payload: dict) -> list[JobListing]:
    """Turn Adzuna's JSON into listings. Entries without a title or link are skipped."""
    listings: list[JobListing] = []
    for item in (payload or {}).get("results") or []:
        if not isinstance(item, dict):
            continue
        title = _clean(item.get("title"), 200)
        url = _safe_url(item.get("redirect_url"))
        if not title or not url:
            continue
        company = _clean((item.get("company") or {}).get("display_name"), 120) or "Unknown company"
        listings.append(JobListing(
            source="adzuna",
            external_id=str(item["id"]) if item.get("id") is not None else None,
            title=title,
            company=company,
            location=_clean((item.get("location") or {}).get("display_name"), 120) or None,
            description=_clean(item.get("description"), 3000),
            url=url,
            posted_on=_parse_date(item.get("created")),
            salary_min=_number(item.get("salary_min")),
            salary_max=_number(item.get("salary_max")),
        ))
    return listings


# --------------------------------------------------------------------------
# Adzuna client (+ cache, because the free tier is limited)
# --------------------------------------------------------------------------
class SearchCache:
    """Small in-memory cache so repeating a search doesn't spend the monthly API quota."""

    def __init__(self, ttl_seconds: float = 3600, max_entries: int = 200, clock=time.monotonic):
        self.ttl, self.max_entries, self._clock = ttl_seconds, max_entries, clock
        self._items: dict = {}

    def get(self, key):
        entry = self._items.get(key)
        if entry is None:
            return None
        stored_at, value = entry
        if self._clock() - stored_at > self.ttl:
            del self._items[key]
            return None
        return value

    def set(self, key, value) -> None:
        if len(self._items) >= self.max_entries:
            self._items.pop(next(iter(self._items)))  # drop the oldest entry
        self._items[key] = (self._clock(), value)


@dataclass
class SearchResult:
    listings: list[JobListing]
    from_cache: bool
    total_matches: int | None = None


class AdzunaClient:
    def __init__(self, app_id: str, app_key: str, country: str = "in",
                 client: httpx.Client | None = None, cache: SearchCache | None = None,
                 timeout: float = 15.0):
        if not app_id or not app_key:
            raise DiscoveryError("Adzuna app ID and key are required.")
        country = (country or "in").strip().lower()
        if not re.fullmatch(r"[a-z]{2}", country):
            raise DiscoveryError("Country must be a two-letter code, like 'in' or 'gb'.")
        self._app_id, self._app_key, self.country = app_id.strip(), app_key.strip(), country
        self._client, self._cache, self._timeout = client, cache, timeout

    def search(self, what: str, where: str | None = None, page: int = 1,
               results_per_page: int = 20, max_days_old: int = 14, sort_by: str = "date") -> SearchResult:
        what = " ".join((what or "").split())
        if not 2 <= len(what) <= 100:
            raise DiscoveryError("Enter search keywords (2 to 100 characters).")
        where = " ".join((where or "").split())[:80] or None
        if sort_by not in SORT_OPTIONS:
            raise DiscoveryError("Unknown sort order.")
        page = min(max(int(page), 1), 5)
        results_per_page = min(max(int(results_per_page), 1), 50)
        max_days_old = min(max(int(max_days_old), 1), 60)

        cache_key = (self.country, what.lower(), (where or "").lower(), page,
                     results_per_page, max_days_old, sort_by)
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return SearchResult(cached[0], True, cached[1])

        params = {"app_id": self._app_id, "app_key": self._app_key, "what": what,
                  "results_per_page": results_per_page, "max_days_old": max_days_old,
                  "sort_by": sort_by, "content-type": "application/json"}
        if where:
            params["where"] = where
        payload = self._get(f"{ADZUNA_BASE}/{self.country}/search/{page}", params)

        listings = parse_adzuna_results(payload)
        total = payload.get("count") if isinstance(payload.get("count"), int) else None
        if self._cache is not None:
            self._cache.set(cache_key, (listings, total))
        return SearchResult(listings, False, total)

    def _get(self, url: str, params: dict) -> dict:
        """One GET. Errors are re-raised WITHOUT the URL, because it contains the API key."""
        try:
            if self._client is not None:
                response = self._client.get(url, params=params, timeout=self._timeout)
            else:
                with httpx.Client() as client:
                    response = client.get(url, params=params, timeout=self._timeout)
        except httpx.TimeoutException:
            raise DiscoveryError("Adzuna did not respond in time. Please try again.") from None
        except httpx.HTTPError:
            raise DiscoveryError("Could not reach Adzuna. Check your connection and try again.") from None

        if response.status_code in (401, 403):
            raise DiscoveryError("Adzuna rejected the app ID or key. Check that they are correct.")
        if response.status_code == 429:
            raise DiscoveryError("Adzuna's request limit was reached. Try again later.")
        if response.status_code >= 400:
            raise DiscoveryError(f"Adzuna returned an error ({response.status_code}). Try again later.")
        try:
            data = response.json()
        except ValueError:
            raise DiscoveryError("Adzuna sent an unexpected response.") from None
        if not isinstance(data, dict):
            raise DiscoveryError("Adzuna sent an unexpected response.")
        return data


# --------------------------------------------------------------------------
# CSV import (jobs you collected yourself: a spreadsheet, a saved search...)
# --------------------------------------------------------------------------
_HEADER_ALIASES = {
    "title": {"title", "job title", "jobtitle", "role", "job role", "position", "designation"},
    "company": {"company", "company name", "companyname", "employer", "organization", "organisation", "firm"},
    "location": {"location", "city", "place", "job location"},
    "url": {"url", "link", "job url", "job link", "apply link", "posting url", "apply url"},
    "description": {"description", "job description", "details", "summary", "jd"},
}


def _decode_csv(data: bytes) -> str:
    """Excel and Notepad can save as UTF-16 or UTF-8 with a BOM; accept all of them."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    return data.decode("utf-8-sig", errors="replace")


def _detect_delimiter(text: str) -> str:
    """Excel in some regions writes ';' instead of ','. Also handles tab and pipe."""
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    return max([",", ";", "\t", "|"], key=first_line.count) if first_line else ","


def _canonical(header: str) -> str | None:
    name = " ".join((header or "").replace("_", " ").lower().split())
    return next((canon for canon, names in _HEADER_ALIASES.items()
                 if name in names or name.replace(" ", "") in names), None)


def _read_table(text: str) -> list[list[str]]:
    """Rows of cells. Also repairs files where each whole line ended up in ONE quoted cell
    (what Excel writes when the text was pasted into a single column)."""
    def parse(raw: str) -> list[list[str]]:
        rows = csv.reader(io.StringIO(raw), delimiter=_detect_delimiter(raw))
        return [r for r in rows if any(cell.strip() for cell in r)]

    rows = parse(text)
    if rows and all(len(r) == 1 for r in rows) and any(d in rows[0][0] for d in ",;\t|"):
        rows = parse("\n".join(r[0] for r in rows))
    return rows


def parse_listings_csv(data: bytes, max_rows: int = MAX_CSV_ROWS) -> tuple[list[JobListing], list[str]]:
    """Read a CSV with columns title, company and optionally location, url, description.

    Common alternative column names (job title, employer, city, apply link...) are accepted.
    """
    rows = _read_table(_decode_csv(data))
    raw_header = rows[0] if rows else []
    index: dict[str, int] = {}
    for position, name in enumerate(raw_header):
        canon = _canonical(name)
        if canon and canon not in index:
            index[canon] = position
    if "title" not in index or "company" not in index:
        found = ", ".join(_clean(h, 30) for h in raw_header[:8] if h.strip()) or "none"
        raise DiscoveryError(
            f"The CSV needs at least the columns 'title' and 'company'. Columns found: {found}.")

    def cell(row, name):
        i = index.get(name)
        return row[i] if i is not None and i < len(row) else None

    body = rows[1:]
    listings: list[JobListing] = []
    skipped = bad_links = 0
    for row in body[:max_rows]:
        title, company = _clean(cell(row, "title"), 200), _clean(cell(row, "company"), 120)
        if not title or not company:
            skipped += 1
            continue
        raw_url = cell(row, "url")
        url = _safe_url(raw_url)
        if raw_url and raw_url.strip() and not url:
            bad_links += 1
        listings.append(JobListing(
            source="csv", title=title, company=company,
            location=_clean(cell(row, "location"), 120) or None,
            description=_clean(cell(row, "description"), 3000), url=url,
        ))

    warnings = []
    if len(body) > max_rows:
        warnings.append(f"Only the first {max_rows} rows were read.")
    if skipped:
        warnings.append(f"{skipped} row(s) skipped (missing title or company).")
    if bad_links:
        warnings.append(f"{bad_links} link(s) ignored (links must start with http:// or https://).")
    return listings, warnings


# --------------------------------------------------------------------------
# De-duplication
# --------------------------------------------------------------------------
_LEGAL_SUFFIX = {"pvt", "private", "ltd", "limited", "llp", "inc", "incorporated", "corp",
                 "corporation", "llc", "plc", "co", "company"}
_CITY_ALIASES = {"bangalore": "bengaluru", "gurgaon": "gurugram", "bombay": "mumbai",
                 "madras": "chennai", "calcutta": "kolkata"}


def normalize_company(name: str) -> str:
    words = re.sub(r"[^a-z0-9 ]+", " ", (name or "").lower()).split()
    return " ".join(w for w in words if w not in _LEGAL_SUFFIX)


def normalize_title(title: str) -> str:
    title = re.sub(r"\([^)]*\)", " ", (title or "").lower())
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", title).split())


def _city(location: str | None) -> str:
    first = (location or "").split(",")[0].lower().strip()
    first = re.sub(r"[^a-z ]+", "", first).strip()
    return _CITY_ALIASES.get(first, first)


def listing_key(listing: JobListing) -> str:
    return f"{normalize_company(listing.company)}|{normalize_title(listing.title)}|{_city(listing.location)}"


def listing_id(listing: JobListing) -> str:
    """Short stable id, used for widget keys."""
    return hashlib.md5(listing_key(listing).encode()).hexdigest()[:10]


def dedupe_listings(listings: list[JobListing]) -> list[JobListing]:
    """One entry per company + title + city; when duplicates differ, keep the most detailed."""
    best: dict[str, JobListing] = {}
    for item in listings:
        key = listing_key(item)
        current = best.get(key)
        if current is None or len(item.description) > len(current.description):
            best[key] = item
    return list(best.values())


def tracked_keys(applications) -> set[tuple[str, str]]:
    """(company, role) pairs already in the tracker, for hiding duplicates of your own list."""
    return {(normalize_company(a.company), normalize_title(a.role)) for a in applications}


def is_tracked(listing: JobListing, tracked: set[tuple[str, str]]) -> bool:
    return (normalize_company(listing.company), normalize_title(listing.title)) in tracked


# --------------------------------------------------------------------------
# Fresher filter
# --------------------------------------------------------------------------
_SENIOR_TITLE = re.compile(
    r"\b(senior|sr|lead|principal|staff|manager|head|director|architect|vp|vice president|chief)\b", re.I)
_FRESHER_WORDS = re.compile(
    r"\b(fresher|freshers|entry[- ]level|graduate|trainee|intern|internship|junior|jr|campus)\b", re.I)
_YEARS = re.compile(
    r"(?P<lo>\d+(?:\.\d+)?)\s*(?:\+|plus)?\s*(?:(?:-|–|to)\s*(?P<hi>\d+(?:\.\d+)?)\s*)?\+?\s*(?:years?|yrs?)\b",
    re.I)


def estimate_min_years(text: str) -> float | None:
    """Highest 'N years' figure that sits next to the word experience (None if none found).

    '10+ years in business' is ignored; '3-5 years of experience' counts as 3.
    """
    values = []
    for match in _YEARS.finditer(text or ""):
        window = text[max(0, match.start() - 40): match.end() + 40]
        if re.search(r"\bexperience\b|\bexp\b", window, re.I):
            values.append(float(match.group("lo")))
    return max(values) if values else None


@dataclass(frozen=True)
class FresherVerdict:
    label: str  # "fresher" | "unclear" | "experienced"
    reason: str


def fresher_verdict(title: str, description: str) -> FresherVerdict:
    if _SENIOR_TITLE.search(title or ""):
        return FresherVerdict("experienced", "senior-level title")
    years = estimate_min_years(f"{title}. {description}")
    if years is not None and years >= 2:
        return FresherVerdict("experienced", f"asks for {years:g}+ years of experience")
    if _FRESHER_WORDS.search(f"{title} {description}"):
        return FresherVerdict("fresher", "fresher / entry-level wording")
    if years is not None:
        return FresherVerdict("fresher", f"asks for {years:g} year(s) or less")
    return FresherVerdict("unclear", "no experience level stated")


# --------------------------------------------------------------------------
# Quick ranking (an estimate from the snippet, not the real match score)
# --------------------------------------------------------------------------
SEM_LOW, SEM_HIGH = 0.55, 0.80  # rough cosine range for resume-vs-posting text; tune on real data
EXPERIENCED_PENALTY = 0.6
FRESHER_BONUS = 5


class RankedListing(BaseModel):
    listing: JobListing
    fit: int
    rank_score: float
    verdict: str
    verdict_reason: str
    matched_skills: list[str]
    already_tracked: bool = False


@dataclass
class RankResult:
    ranked: list[RankedListing]
    duplicates_removed: int
    hidden_experienced: int
    hidden_tracked: int


def quick_fit(listing: JobListing, profile: ResumeProfile, store: VectorStore) -> tuple[int, list[str]]:
    """0-100: 60% how many of YOUR skills the posting mentions, 40% semantic closeness."""
    text = f"{listing.title}. {listing.description}"
    text_tokens = _tokens(text)
    matched = [s for s in profile.known_skills() if (t := _tokens(s)) and t <= text_tokens]
    skill_signal = min(1.0, len(matched) / 4)

    semantic = 0.0
    if store.chunks:
        sims = sorted((store.vectors @ embeddings.embed_query(text[:800])).tolist(), reverse=True)[:3]
        semantic = min(1.0, max(0.0, (sum(sims) / len(sims) - SEM_LOW) / (SEM_HIGH - SEM_LOW)))
    return round(100 * (0.6 * skill_signal + 0.4 * semantic)), matched


def rank_listings(listings: list[JobListing], profile: ResumeProfile, store: VectorStore,
                  tracked: set[tuple[str, str]] | None = None, include_experienced: bool = False,
                  hide_tracked: bool = True) -> RankResult:
    tracked = tracked or set()
    unique = dedupe_listings(listings)
    ranked: list[RankedListing] = []
    hidden_experienced = hidden_tracked = 0

    for listing in unique:
        verdict = fresher_verdict(listing.title, listing.description)
        already = is_tracked(listing, tracked)
        if already and hide_tracked:
            hidden_tracked += 1
            continue
        if verdict.label == "experienced" and not include_experienced:
            hidden_experienced += 1
            continue
        fit, matched = quick_fit(listing, profile, store)
        score = fit * EXPERIENCED_PENALTY if verdict.label == "experienced" else fit
        score += FRESHER_BONUS if verdict.label == "fresher" else 0
        ranked.append(RankedListing(listing=listing, fit=fit, rank_score=round(score, 2),
                                    verdict=verdict.label, verdict_reason=verdict.reason,
                                    matched_skills=matched, already_tracked=already))

    ranked.sort(key=lambda r: (r.rank_score, r.listing.posted_on or date.min), reverse=True)
    return RankResult(ranked, len(listings) - len(unique), hidden_experienced, hidden_tracked)
