"""Turn extracted PDF blocks into clean, speakable text.

Every function here is pure (str/data in, str/data out) so the whole cleaning
pipeline is unit-testable without touching PyMuPDF or the TTS model. The public
entry point is :func:`clean_pages`; the individual steps are exported so they
can be tested and reused in isolation.
"""

from __future__ import annotations

import unicodedata

import regex as re

from .extract import Block, Page

# --- Character-level normalization ------------------------------------------

# Ligatures and typographic characters that NFKC does not always fold the way we
# want for speech. Mapped explicitly so behavior is predictable.
_CHAR_MAP = {
    "­": "",   # soft hyphen -> nothing
    "‘": "'",  # left single quote
    "’": "'",  # right single quote / apostrophe
    "“": '"',  # left double quote
    "”": '"',  # right double quote
    "–": "-",  # en dash
    "—": "-",  # em dash
    "−": "-",  # minus sign
    "…": "...",  # ellipsis
    " ": " ",  # non-breaking space
    "•": " ",  # bullet
}

_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}


def normalize_unicode(text: str) -> str:
    """Fold ligatures, smart quotes, dashes and stray control characters.

    Applies NFKC normalization first (which handles most ligatures and
    full-width forms) then a small explicit map for characters NFKC leaves in a
    form that reads awkwardly aloud.
    """
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _CHAR_MAP.items():
        text = text.replace(src, dst)
    return text


# --- Line joining ------------------------------------------------------------

# A word split across a line break: "hyphen-\nation" -> "hyphenation".
_HYPHEN_BREAK = re.compile(r"(\p{L})-\s*\n\s*(\p{L})")


def dehyphenate(text: str) -> str:
    """Rejoin words hyphenated across a line break.

    Only fires between two letters so genuine hyphenated compounds that happen
    to wrap (``well-\nknown``) collapse to a single token — acceptable, since
    the alternative of guessing a dictionary is worse for speech.
    """
    # Loop because adjacent breaks can overlap on the shared boundary.
    prev = None
    while prev != text:
        prev = text
        text = _HYPHEN_BREAK.sub(r"\1\2", text)
    return text


# --- Reference / citation markers -------------------------------------------

# Bracketed numeric citations like "[12]", "[3, 4]", "[5-9]" — but not prose in
# brackets, which contains letters.
_REF_BRACKET = re.compile(r"\[\s*\d+(?:\s*[-,]\s*\d+)*\s*\]")


def strip_reference_markers(text: str) -> str:
    """Remove inline bracketed numeric citation markers."""
    return _REF_BRACKET.sub("", text)


# --- Page-number / running-line detection -----------------------------------

_ROMAN = r"[ivxlcdm]+"
# A line that is nothing but a page marker: bare number, roman numeral,
# "Page 12", "12 of 340", "- 12 -", possibly wrapped in punctuation.
_PAGE_NUMBER = re.compile(
    rf"""^\s*(?:
        [-–—\[\(]*\s*\d+\s*[-–—\]\)]* |
        page\s+\d+(?:\s+of\s+\d+)? |
        \d+\s+of\s+\d+ |
        {_ROMAN}
    )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


def is_page_number(line: str) -> bool:
    """True if a line is purely a page marker (and nothing worth speaking)."""
    return bool(_PAGE_NUMBER.match(line))


def _match_key(text: str) -> str:
    """Normalize block text for header/footer matching.

    Lowercased, digits removed (page numbers vary per page), whitespace
    collapsed. Two headers that differ only in their page number collapse to the
    same key.
    """
    text = re.sub(r"\d+", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def find_running_lines(
    pages: list[Page],
    zone: float = 0.12,
    threshold: float = 0.5,
    min_pages: int = 4,
) -> set[str]:
    """Find header/footer text that repeats across pages near top or bottom.

    A block counts as a running line if its normalized key appears (in the top
    ``zone`` or bottom ``zone`` band of the page) on at least ``threshold`` of
    the pages. Disabled for documents shorter than ``min_pages`` where repetition
    is not yet evidence of a running header.
    """
    if len(pages) < min_pages:
        return set()

    counts: dict[str, int] = {}
    for page in pages:
        seen_on_page: set[str] = set()
        for block in page:
            in_top = block.rel_top < zone
            in_bottom = block.page_height > 0 and (
                block.y_bottom / block.page_height > 1 - zone
            )
            if not (in_top or in_bottom):
                continue
            key = _match_key(block.text)
            if key and key not in seen_on_page:
                counts[key] = counts.get(key, 0) + 1
                seen_on_page.add(key)

    needed = max(min_pages - 1, threshold * len(pages))
    return {key for key, count in counts.items() if count >= needed}


# --- Paragraph reflow --------------------------------------------------------


def reflow_paragraphs(text: str) -> str:
    """Collapse intra-paragraph line breaks into spaces; keep paragraph breaks.

    A blank line separates paragraphs (preserved as ``\\n\\n``); single newlines
    inside a paragraph are just PDF line wrapping and become spaces. Trailing and
    repeated whitespace is collapsed.
    """
    # Normalize newlines and split on blank lines into paragraphs.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n\s*\n", text)
    cleaned: list[str] = []
    for para in paragraphs:
        joined = re.sub(r"\s+", " ", para).strip()
        # Tidy stray whitespace left before punctuation (e.g. after a removed
        # citation marker: "shown  ." -> "shown.").
        joined = re.sub(r"\s+([.,;:!?])", r"\1", joined)
        if joined:
            cleaned.append(joined)
    return "\n\n".join(cleaned)


# --- Orchestration -----------------------------------------------------------


def clean_block_text(text: str) -> str:
    """Drop page-number-only lines from within a block, keep the rest."""
    kept = [line for line in text.splitlines() if not is_page_number(line)]
    return "\n".join(kept)


def clean_pages(pages: list[Page]) -> str:
    """Full cleaning pipeline: positioned blocks -> speakable text.

    Order matters: unicode folding and de-hyphenation run before paragraph
    reflow (which destroys the line breaks they rely on).
    """
    running = find_running_lines(pages)

    page_texts: list[str] = []
    for page in pages:
        block_texts: list[str] = []
        for block in page:
            if _match_key(block.text) in running:
                continue
            body = clean_block_text(block.text)
            if body.strip():
                block_texts.append(body)
        # Blocks are visually separate units -> paragraph boundaries.
        page_texts.append("\n\n".join(block_texts))

    text = "\n\n".join(pt for pt in page_texts if pt.strip())

    text = normalize_unicode(text)
    text = dehyphenate(text)
    text = strip_reference_markers(text)
    text = reflow_paragraphs(text)
    return text
