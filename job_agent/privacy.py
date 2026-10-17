"""Remove personal contact details before text is sent to any cloud LLM.

Why: the free Gemini tier may use prompts to improve Google's models.
Skills, projects and bullets are enough for matching; your phone number
and email are not. We re-insert real contact details only at PDF export.
"""

import re

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(
    r"(?:https?://\S+|www\.\S+|(?:linkedin\.com|github\.com|gitlab\.com)/\S*)",
    re.IGNORECASE,
)
# A run of digits with separators; accepted only if it has 10-13 digits total
_PHONE = re.compile(r"(?<![\w.])\+?\d[\d\s().-]{8,}\d(?![\w])")


def _mask_phone(match: re.Match) -> str:
    digits = sum(ch.isdigit() for ch in match.group())
    return "[PHONE]" if 10 <= digits <= 13 else match.group()


def _looks_like_name(line: str) -> bool:
    words = line.split()
    return (
        1 <= len(words) <= 4
        and all(w.replace(".", "").replace("'", "").replace("-", "").isalpha() for w in words)
        and all(w[0].isupper() or w.isupper() for w in words)
    )


def redact_pii(text: str, redact_name: bool = True) -> str:
    """Mask emails, phone numbers, profile URLs and (heuristically) the name line."""
    text = _EMAIL.sub("[EMAIL]", text)
    text = _URL.sub("[URL]", text)
    text = _PHONE.sub(_mask_phone, text)

    if redact_name:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.strip():  # first non-empty line
                if _looks_like_name(line.strip()):
                    lines[i] = "[NAME]"
                break
        text = "\n".join(lines)
    return text
