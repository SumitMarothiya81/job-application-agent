from datetime import date

import httpx
import pytest

from job_agent import discovery as d
from job_agent.demo import build_store, load_demo_profile
from job_agent.discovery import AdzunaClient, DiscoveryError, JobListing, SearchCache

PAYLOAD = {
    "count": 321,
    "results": [
        {"id": "111", "title": "Data Science <strong>Intern</strong>", "redirect_url": "https://adzuna.example/jobs/111",
         "company": {"display_name": "Acme Technologies Pvt. Ltd."}, "location": {"display_name": "Bengaluru, Karnataka"},
         "description": "Work with <strong>Python</strong> &amp; SQL. Freshers welcome.", "created": "2026-10-15T09:14:13Z",
         "salary_min": 300000, "salary_max": 400000},
        {"id": 222, "title": "Analyst", "redirect_url": "javascript:alert(1)", "company": {"display_name": "Evil"}},  # bad link
        {"id": 333, "title": "", "redirect_url": "https://adzuna.example/jobs/333"},                              # no title
        {"id": 444, "title": "ML Engineer", "redirect_url": "https://adzuna.example/jobs/444"},                  # minimal entry
        "not a dict",
    ],
}


def make_client(handler, **kwargs):
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return AdzunaClient("my-id", "my-secret-key", client=http, **kwargs)


# ---- parsing ---------------------------------------------------------------------
def test_parse_adzuna_results_cleans_and_skips_bad_entries():
    listings = d.parse_adzuna_results(PAYLOAD)
    assert [l.external_id for l in listings] == ["111", "444"]
    first = listings[0]
    assert first.title == "Data Science Intern"                       # <strong> tags removed
    assert first.description == "Work with Python & SQL. Freshers welcome."  # entities decoded
    assert first.company == "Acme Technologies Pvt. Ltd." and first.location == "Bengaluru, Karnataka"
    assert first.posted_on == date(2026, 10, 15) and first.salary_min == 300000
    assert listings[1].company == "Unknown company" and listings[1].posted_on is None


def test_parse_handles_empty_payloads():
    assert d.parse_adzuna_results({}) == [] and d.parse_adzuna_results({"results": None}) == []


# ---- client ----------------------------------------------------------------------
def test_search_builds_the_right_request_and_returns_listings():
    seen = {}

    def handler(request):
        seen["url"], seen["params"] = request.url, dict(request.url.params)
        return httpx.Response(200, json=PAYLOAD)

    result = make_client(handler, country="IN").search("  data   science intern ", where="Pune",
                                                       results_per_page=500, max_days_old=7, sort_by="date")
    assert seen["url"].path == "/v1/api/jobs/in/search/1"
    params = seen["params"]
    assert params["what"] == "data science intern" and params["where"] == "Pune"
    assert params["results_per_page"] == "50"                 # clamped to Adzuna's maximum
    assert params["max_days_old"] == "7" and params["sort_by"] == "date"
    assert params["app_id"] == "my-id" and params["app_key"] == "my-secret-key"
    assert len(result.listings) == 2 and result.total_matches == 321 and not result.from_cache


@pytest.mark.parametrize("status, message", [(401, "rejected"), (403, "rejected"), (429, "limit"),
                                             (500, r"error \(500\)")])
def test_http_errors_become_safe_messages(status, message):
    client = make_client(lambda request: httpx.Response(status))
    with pytest.raises(DiscoveryError, match=message) as info:
        client.search("python")
    assert "my-secret-key" not in str(info.value)


def test_network_errors_never_leak_the_api_key():
    def boom(request):
        raise httpx.ConnectError("failed for " + str(request.url))   # the URL contains app_key

    with pytest.raises(DiscoveryError) as info:
        make_client(boom).search("python")
    assert "my-secret-key" not in str(info.value) and "Could not reach" in str(info.value)

    def slow(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(DiscoveryError, match="in time"):
        make_client(slow).search("python")


def test_bad_json_and_bad_inputs():
    with pytest.raises(DiscoveryError, match="unexpected"):
        make_client(lambda r: httpx.Response(200, text="<html>")).search("python")
    with pytest.raises(DiscoveryError, match="unexpected"):
        make_client(lambda r: httpx.Response(200, json=["a list"])).search("python")
    client = make_client(lambda r: httpx.Response(200, json=PAYLOAD))
    for bad in ("", " ", "x", "y" * 101):
        with pytest.raises(DiscoveryError, match="keywords"):
            client.search(bad)
    with pytest.raises(DiscoveryError, match="sort"):
        client.search("python", sort_by="random")
    with pytest.raises(DiscoveryError):
        AdzunaClient("", "key")
    with pytest.raises(DiscoveryError, match="two-letter"):
        AdzunaClient("id", "key", country="india")


def test_cache_avoids_a_second_request_and_expires():
    calls = []
    now = {"t": 0.0}
    cache = SearchCache(ttl_seconds=100, clock=lambda: now["t"])

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=PAYLOAD)

    client = make_client(handler, cache=cache)
    first = client.search("Python Intern", where="Pune")
    second = client.search("  python   intern ", where="pune")     # same search, different spacing/case
    assert len(calls) == 1 and not first.from_cache and second.from_cache
    assert second.total_matches == 321 and len(second.listings) == 2

    client.search("python intern", where="Delhi")                    # different search -> new request
    assert len(calls) == 2
    now["t"] = 101                                                   # past the ttl
    assert not client.search("python intern", where="Pune").from_cache and len(calls) == 3


def test_cache_evicts_the_oldest_entry():
    cache = SearchCache(max_entries=2)
    cache.set("a", 1), cache.set("b", 2), cache.set("c", 3)
    assert cache.get("a") is None and cache.get("b") == 2 and cache.get("c") == 3


# ---- CSV -------------------------------------------------------------------------
def test_csv_import_with_bom_and_validation():
    csv_bytes = ("\ufefftitle,Company,location,URL,description\n"
                 "Data Analyst Intern,Acme,Pune,https://jobs.example.com/1,Use SQL\n"
                 ",NoTitle,,,\n"
                 "ML Intern,Beta,Delhi,javascript:alert(1),Python\n"
                 "<b>BI</b> Intern,Gamma,,,\n").encode("utf-8")
    listings, warnings = d.parse_listings_csv(csv_bytes)
    assert [l.title for l in listings] == ["Data Analyst Intern", "ML Intern", "BI Intern"]
    assert listings[0].url == "https://jobs.example.com/1" and listings[1].url is None
    assert any("1 row(s) skipped" in w for w in warnings) and any("1 link(s) ignored" in w for w in warnings)
    assert all(l.source == "csv" for l in listings)


def test_csv_requires_columns_and_caps_rows():
    with pytest.raises(DiscoveryError, match="title.*company") as info:
        d.parse_listings_csv(b"name,employer\nx,y\n")
    assert "Columns found: name, employer." in str(info.value)                      # tells the user what it found
    with pytest.raises(DiscoveryError, match="Columns found: none"):
        d.parse_listings_csv(b"")
    rows = "title,company\n" + "".join(f"Job {i},Co {i}\n" for i in range(10))
    listings, warnings = d.parse_listings_csv(rows.encode(), max_rows=3)
    assert len(listings) == 3 and any("first 3 rows" in w for w in warnings)


# ---- de-duplication -------------------------------------------------------------------
def job(title="Data Science Intern", company="Acme Technologies Pvt. Ltd.", location="Bengaluru", description="", **kw):
    return JobListing(source="adzuna", title=title, company=company, location=location,
                      description=description, url="https://x.example/1", **kw)


def test_normalization_merges_spelling_variants():
    assert d.normalize_company("ACME Technologies Private Limited") == d.normalize_company("Acme Technologies Pvt. Ltd.")
    assert d.normalize_title("Data Science Intern (Remote)") == d.normalize_title("data-science  intern")
    assert d.listing_key(job(location="Bangalore, Karnataka")) == d.listing_key(job(location="Bengaluru"))


def test_dedupe_keeps_most_detailed_and_distinguishes_jobs():
    short, long_ = job(), job(company="ACME Technologies Private Limited", description="A longer description")
    other_city, other_role = job(location="Pune"), job(title="Data Analyst")
    result = d.dedupe_listings([short, long_, other_city, other_role])
    assert len(result) == 3
    assert next(l for l in result if d.listing_key(l) == d.listing_key(short)).description == "A longer description"


def test_tracked_matching_ignores_location():
    class App:
        company, role = "acme technologies", "Data Science Intern"

    tracked = d.tracked_keys([App()])
    assert d.is_tracked(job(), tracked) and not d.is_tracked(job(title="Data Analyst"), tracked)


# ---- fresher filter ---------------------------------------------------------------------
@pytest.mark.parametrize("text, years", [
    ("Minimum 3 years of experience in Python", 3.0),
    ("3-5 years experience required", 3.0),
    ("5+ yrs exp with SQL", 5.0),
    ("0-1 years of experience", 0.0),
    ("We are a company with 10+ years in business", None),   # not about the candidate
    ("Great team, freshers welcome", None),
    ("1 year experience with SQL and 4 years of experience overall", 4.0),
])
def test_estimate_min_years(text, years):
    assert d.estimate_min_years(text) == years


@pytest.mark.parametrize("title, description, label", [
    ("Senior Data Scientist", "Freshers welcome", "experienced"),            # title wins
    ("Data Analyst", "Requires 4+ years of experience", "experienced"),
    ("Data Analyst", "1-2 years of experience", "fresher"),
    ("Software Engineer Trainee", "Join our team", "fresher"),
    ("Data Science Intern", "", "fresher"),
    ("Data Analyst", "Work with dashboards", "unclear"),
])
def test_fresher_verdict(title, description, label):
    assert d.fresher_verdict(title, description).label == label


# ---- ranking ------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def resume():
    profile = load_demo_profile()
    return profile, build_store(profile)


def test_quick_fit_prefers_postings_that_mention_your_skills(resume):
    profile, store = resume
    good, good_skills = d.quick_fit(job(title="Python Data Intern", description="Python, SQL, pandas and scikit-learn"), profile, store)
    poor, poor_skills = d.quick_fit(job(title="Retail Sales Executive", description="Sell products to customers"), profile, store)
    assert good > poor and {"python", "sql", "pandas"} <= set(good_skills) and poor_skills == []
    assert 0 <= poor <= good <= 100


def test_rank_listings_sorts_filters_and_counts(resume):
    profile, store = resume
    listings = [
        job(title="Python Data Intern", company="A", description="Python SQL pandas freshers"),
        job(title="Python Data Intern", company="A Pvt Ltd", description="Python SQL pandas freshers (dup)"),
        job(title="Sales Trainee", company="B", description="Sell products"),
        job(title="Senior Python Engineer", company="C", description="Python SQL pandas scikit-learn"),
        job(title="Data Intern", company="D", description="Python"),
    ]
    tracked = {(d.normalize_company("D"), d.normalize_title("Data Intern"))}

    result = d.rank_listings(listings, profile, store, tracked)
    titles = [r.listing.company for r in result.ranked]
    assert d.normalize_company(titles[0]) == "a" and "C" not in titles and "D" not in titles  # best first; senior + tracked hidden
    assert (result.duplicates_removed, result.hidden_experienced, result.hidden_tracked) == (1, 1, 1)
    assert [r.rank_score for r in result.ranked] == sorted([r.rank_score for r in result.ranked], reverse=True)

    everything = d.rank_listings(listings, profile, store, tracked, include_experienced=True, hide_tracked=False)
    senior = next(r for r in everything.ranked if r.listing.company == "C")
    assert senior.verdict == "experienced" and senior.rank_score < senior.fit            # penalized
    assert next(r for r in everything.ranked if r.listing.company == "D").already_tracked


# ---- CSV tolerance (real-world files from Notepad / Excel) --------------------------------
SAMPLE_ROWS = ("title,company,location,url,description\r\n"
               "Python Data Intern,Acme,Pune,https://example.com/1,Python SQL pandas freshers welcome\r\n"
               "Sales Trainee,ShopCo,Jaipur,,Sell products\r\n")


@pytest.mark.parametrize("name, data", [
    ("windows line endings", SAMPLE_ROWS.encode("utf-8")),
    ("blank first line", ("\r\n\r\n" + SAMPLE_ROWS).encode("utf-8")),
    ("utf-8 BOM", SAMPLE_ROWS.encode("utf-8-sig")),
    ("utf-16 (Notepad 'Unicode')", SAMPLE_ROWS.encode("utf-16")),
    ("semicolon separator (Excel)", SAMPLE_ROWS.replace(",", ";").encode("utf-8")),
    ("tab separator", SAMPLE_ROWS.replace(",", "\t").encode("utf-8")),
    ("spaces and capitals in the header", SAMPLE_ROWS.replace("title,company", " Title , COMPANY ").encode("utf-8")),
])
def test_csv_accepts_common_real_world_formats(name, data):
    listings, warnings = d.parse_listings_csv(data)
    assert [l.title for l in listings] == ["Python Data Intern", "Sales Trainee"], name
    assert listings[0].company == "Acme" and listings[0].location == "Pune"


def test_csv_header_aliases():
    data = b"Job Title,Employer,City,Apply Link,Job Description\nData Intern,Acme,Pune,https://x.example/1,SQL\n"
    (listing,), _ = d.parse_listings_csv(data)
    assert (listing.title, listing.company, listing.location, listing.url, listing.description) == (
        "Data Intern", "Acme", "Pune", "https://x.example/1", "SQL")


def test_csv_error_lists_the_columns_it_found():
    with pytest.raises(DiscoveryError) as info:
        d.parse_listings_csv(b"Python Data Intern,Acme,Pune\nSales Trainee,ShopCo,Jaipur\n")  # no header line
    assert "Columns found: Python Data Intern, Acme, Pune" in str(info.value)
    with pytest.raises(DiscoveryError, match="Columns found: none"):
        d.parse_listings_csv(b"")


# ---- CSV files saved by Excel / Notepad in other layouts ----------------------------------
GOOD_ROWS = [("Python Data Intern", "Acme"), ("Sales Trainee", "ShopCo")]
TABLE = [["title", "company", "location", "url"],
         ["Python Data Intern", "Acme", "Pune", "https://jobs.example.com/1"],
         ["Sales Trainee", "ShopCo", "Jaipur", ""]]


def as_text(delimiter):
    return "\r\n".join(delimiter.join(row) for row in TABLE) + "\r\n"


@pytest.mark.parametrize("delimiter", [";", "\t", "|", ","])
def test_csv_detects_the_delimiter(delimiter):
    listings, warnings = d.parse_listings_csv(as_text(delimiter).encode("utf-8"))
    assert [(l.title, l.company) for l in listings] == GOOD_ROWS and warnings == []
    assert listings[0].url == "https://jobs.example.com/1" and listings[0].location == "Pune"


@pytest.mark.parametrize("encoding", ["utf-16", "utf-8-sig", "utf-8"])
def test_csv_handles_notepad_encodings(encoding):
    listings, _ = d.parse_listings_csv(as_text(",").encode(encoding))
    assert [(l.title, l.company) for l in listings] == GOOD_ROWS


def test_csv_repairs_lines_wrapped_in_one_quoted_cell():
    wrapped = "\r\n".join('"' + ",".join(row) + '"' for row in TABLE) + "\r\n"   # what Excel writes if each line sits in column A
    listings, _ = d.parse_listings_csv(wrapped.encode("utf-8"))
    assert [(l.title, l.company) for l in listings] == GOOD_ROWS


def test_csv_ignores_blank_lines_and_short_rows():
    text = "title,company,description\n\nPython Intern,Acme\n,,\n"
    listings, _ = d.parse_listings_csv(text.encode())
    assert [(l.title, l.description) for l in listings] == [("Python Intern", "")]
