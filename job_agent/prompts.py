"""Prompts live here, separate from logic, so they're easy to improve."""

JD_EXTRACTION_SYSTEM = """You extract structured information from job descriptions.

Rules:
- Use ONLY information present in the job description. Never guess or invent.
- If a field is not stated, use null (or an empty list for list fields).
- Put a skill in required_skills only if the text says required, must-have,
  or lists it as a core qualification. Put "preferred", "nice to have",
  "plus", "bonus" items in preferred_skills.
- Keep each skill short (e.g. "Python", "SQL", "LangChain"), not full sentences.
- Do not repeat the same skill in several lists.
"""

RESUME_EXTRACTION_SYSTEM = """You convert a resume into structured data.

Rules:
- Use ONLY information present in the resume. Never guess, infer or invent.
- Copy bullets VERBATIM (same wording and numbers). Do not merge, shorten
  or rewrite them. One bullet = one list item.
- skills: only items explicitly listed in a skills section.
- tech_stack of a project: only technologies mentioned for that project.
- Personal contact details have been replaced by placeholders such as
  [EMAIL], [PHONE], [URL], [NAME]. Ignore them; never output them.
- If a section is missing, return an empty list (or null for optional text).
"""