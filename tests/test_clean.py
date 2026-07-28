"""Unit tests for the pure text-cleaning functions.

These exercise the quality-critical core without loading PyMuPDF or the TTS
model — fast and deterministic.
"""

from pdftoaudio.clean import (
    clean_pages,
    dehyphenate,
    find_running_lines,
    is_page_number,
    normalize_unicode,
    reflow_paragraphs,
    strip_reference_markers,
)
from pdftoaudio.extract import Block


def _block(text, y_top=400.0, height=800.0):
    """Build a Block with a bbox placing its top at ``y_top``."""
    return Block(text=text, bbox=(0.0, y_top, 500.0, y_top + 20.0), page_height=height)


# --- normalize_unicode -------------------------------------------------------


def test_ligatures_folded():
    assert normalize_unicode("ﬁnal ﬂoat oﬀice") == "final float office"


def test_smart_quotes_and_dashes():
    assert normalize_unicode("“hi” — it’s") == '"hi" - it\'s'


def test_soft_hyphen_removed():
    assert normalize_unicode("cooper­ate") == "cooperate"


# --- dehyphenate -------------------------------------------------------------


def test_dehyphenate_joins_across_break():
    assert dehyphenate("hyphen-\nation") == "hyphenation"


def test_dehyphenate_handles_spaces_around_break():
    assert dehyphenate("infor-  \n  mation") == "information"


def test_dehyphenate_leaves_inline_hyphen():
    assert dehyphenate("well-known") == "well-known"


# --- strip_reference_markers -------------------------------------------------


def test_strip_single_reference():
    assert strip_reference_markers("shown by Smith [12] later") == "shown by Smith  later"


def test_strip_reference_list_and_range():
    assert strip_reference_markers("prior work [3, 4] and [5-9]") == "prior work  and "


def test_keep_bracketed_prose():
    text = "the result [see appendix] holds"
    assert strip_reference_markers(text) == text


# --- is_page_number ----------------------------------------------------------


def test_page_number_variants():
    assert is_page_number("12")
    assert is_page_number("- 7 -")
    assert is_page_number("Page 5")
    assert is_page_number("12 of 340")
    assert is_page_number("iv")


def test_not_page_number():
    assert not is_page_number("Chapter 12: Introduction")
    assert not is_page_number("The year 1984 mattered")


# --- reflow_paragraphs -------------------------------------------------------


def test_reflow_joins_wrapped_lines():
    assert reflow_paragraphs("a wrapped\nline here") == "a wrapped line here"


def test_reflow_preserves_paragraph_breaks():
    assert reflow_paragraphs("para one\n\npara two") == "para one\n\npara two"


def test_reflow_collapses_whitespace():
    assert reflow_paragraphs("too    much   space") == "too much space"


def test_reflow_tidies_space_before_punctuation():
    # Left behind after a stripped citation marker.
    assert reflow_paragraphs("shown by Smith  . Next") == "shown by Smith. Next"


# --- find_running_lines ------------------------------------------------------


def test_detects_repeating_header():
    pages = [[_block("A Very Long Book Title", y_top=10.0)] for _ in range(6)]
    running = find_running_lines(pages)
    assert "a very long book title" in running


def test_ignores_headers_on_short_docs():
    pages = [[_block("Title", y_top=10.0)] for _ in range(2)]
    assert find_running_lines(pages) == set()


def test_body_text_not_flagged_as_running():
    pages = [[_block(f"unique body {i}", y_top=400.0)] for i in range(6)]
    assert find_running_lines(pages) == set()


# --- clean_pages (integration of the pure pipeline) --------------------------


def test_clean_pages_end_to_end():
    header = "Running Header"
    pages = []
    for i in range(5):
        pages.append(
            [
                _block(header, y_top=10.0),
                _block("This is a sen-\ntence that wraps [1].", y_top=400.0),
                _block(str(i + 1), y_top=780.0),  # page number at the bottom
            ]
        )
    result = clean_pages(pages)
    assert "Running Header" not in result
    assert "sentence that wraps" in result  # de-hyphenated
    assert "[1]" not in result  # reference stripped
    assert "1\n" not in result and not result.strip().endswith("5")
