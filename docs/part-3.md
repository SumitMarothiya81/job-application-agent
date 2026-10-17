# Part 3a: Matching and skill gaps: what we built and why

(Rewrite in your own words. Questions to answer:)
- Why isn't the raw embedding score enough to say "the candidate has this skill"? (our measured 0.58-0.64 vs 0.69-0.75 overlap)
- Why three statuses (explicit / related / missing) instead of a yes/no?
- Why a curated RELATED_GROUPS taxonomy for Power BI <-> Tableau style skills?
- Why must single-word skills match exactly ('java' vs 'javascript', 'sql' vs 'mysql')?
- Why is the semantic fallback restricted to bullet sections (not the skills list or education)?
- Why weight required 70% / preferred 30%, and what would you change with real data?
- What are ATS keywords and why do we report them separately from skills?
- Why does the gap advisor only cover skills we pass in, and why does code (not the LLM) set priority?
- How would you evaluate whether the match score is "good"? (Part 8: labelled JD/resume pairs)
