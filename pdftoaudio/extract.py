"""PDF text extraction via PyMuPDF.

Extraction is kept deliberately dumb: it returns positioned text blocks and
leaves all normalization to :mod:`pdftoaudio.clean`. Keeping block geometry
(the bounding box + page height) lets the cleaner detect running headers and
footers by where they sit on the page, which is far more reliable than text
heuristics alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Block:
    """A single positioned text block on a page."""

    text: str
    #: Bounding box (x0, y0, x1, y1) in PDF points, origin at top-left.
    bbox: tuple[float, float, float, float]
    #: Height of the page this block belongs to, in points.
    page_height: float

    @property
    def y_top(self) -> float:
        return self.bbox[1]

    @property
    def y_bottom(self) -> float:
        return self.bbox[3]

    @property
    def rel_top(self) -> float:
        """Vertical position of the block top as a 0..1 fraction of page height."""
        if self.page_height <= 0:
            return 0.0
        return self.y_top / self.page_height


#: One page is a list of its text blocks in reading order.
Page = list[Block]


def _parse_page_range(page_range: str | None, page_count: int) -> range:
    """Turn a 1-based inclusive ``"start-end"`` string into a 0-based range.

    ``None`` selects every page. A bare number selects a single page. Bounds are
    clamped to the document, so an over-long range simply stops at the last page.
    """
    if page_range is None:
        return range(page_count)

    text = page_range.strip()
    if "-" in text:
        start_s, _, end_s = text.partition("-")
        start = int(start_s) if start_s.strip() else 1
        end = int(end_s) if end_s.strip() else page_count
    else:
        start = end = int(text)

    start = max(1, start)
    end = min(page_count, end)
    if start > end:
        raise ValueError(f"empty page range: {page_range!r}")
    # Convert 1-based inclusive to 0-based half-open.
    return range(start - 1, end)


def extract_pages(pdf_path: str | Path, page_range: str | None = None) -> list[Page]:
    """Extract positioned text blocks from a PDF.

    Args:
        pdf_path: Path to the source PDF.
        page_range: Optional 1-based inclusive range like ``"3-12"`` or ``"5"``.
            ``None`` (default) reads the whole document.

    Returns:
        A list of pages, each a list of :class:`Block` in reading order.
    """
    import fitz  # PyMuPDF — imported lazily so `Block` and tests don't load it.

    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"no such PDF: {path}")

    pages: list[Page] = []
    with fitz.open(path) as doc:
        for index in _parse_page_range(page_range, doc.page_count):
            page = doc[index]
            height = page.rect.height
            blocks: Page = []
            # "blocks" yields (x0, y0, x1, y1, text, block_no, block_type).
            for x0, y0, x1, y1, text, _block_no, block_type in page.get_text("blocks"):
                if block_type != 0:  # skip images and other non-text blocks
                    continue
                if not text.strip():
                    continue
                blocks.append(Block(text=text, bbox=(x0, y0, x1, y1), page_height=height))
            pages.append(blocks)
    return pages
