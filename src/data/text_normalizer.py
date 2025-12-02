"""Transcript normalization utilities.

The goal is to keep text compatible with the char-level vocab while stripping
book-style artifacts (e.g., Roman numeral headings) and normalizing punctuation.
"""

from __future__ import annotations

import re
import unicodedata

_REPLACEMENTS = {
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\u2013": "-",
    "\u2014": "-",
    "\u2047": "?",  # double question mark
    "\xa0": " ",
}

_HEADING_RE = re.compile(r"^(?:[ivxlcdm]+\.|\d+\.)\s+", re.IGNORECASE)
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")


def normalize_transcript(text: str) -> str:
    """Lowercase, strip whitespace, normalize quotes/dashes, drop leading headings."""
    if text is None:
        return ""
    s = str(text)
    for src, tgt in _REPLACEMENTS.items():
        s = s.replace(src, tgt)
    s = unicodedata.normalize("NFKC", s)
    s = _NON_ASCII_RE.sub(" ", s)
    s = _HEADING_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()
