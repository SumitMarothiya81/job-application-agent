# Part 5: Job discovery: what we built and why

(Rewrite in your own words. Questions to answer:)
- Why Adzuna's official API and CSV import instead of scraping LinkedIn/Naukri/Indeed?
- Why is the 'Quick fit' only an estimate? (Adzuna truncates descriptions) How would you validate it?
- Why 60% skills-mentioned and 40% semantic similarity? What would you tune with real data?
- How does the client avoid leaking the API key in error messages? (the URL contains app_key)
- Why cache searches, and why don't cache hits use the demo allowance?
- How are duplicates detected? (normalized company + title + city; legal suffixes and city aliases)
- How does the fresher filter work, and where does it fail? ('10+ years in business', 'Associate', non-English text)
- Why must the 'Analyze in Match tab' button use a callback instead of setting session_state directly?
- Why only http(s) links, and why strip HTML from Adzuna text?
- What does the CSV importer protect against? (row cap, bad links, missing columns, HTML in cells)
