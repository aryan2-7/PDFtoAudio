"""Command-line entry point: ``pdftoaudio input.pdf [output]``.

Orchestrates the pipeline extract -> clean -> synthesize -> write, with a
deliberately minimal surface (input PDF + optional output path). Voice, language
and speed use sensible defaults from :mod:`pdftoaudio.tts`.
"""

from __future__ import annotations

# Muting third-party warning noise must happen before torch / PyMuPDF / kokoro
# are imported (directly or lazily), so it runs at module load, first thing.
from ._quiet import silence_third_party_noise

silence_third_party_noise()

import argparse
import shutil
import sys
from pathlib import Path

from . import __version__
from .audio import format_from_path, write_stream
from .clean import clean_pages
from .extract import extract_pages
from .tts import model_is_cached, synthesize


def _default_output(input_path: Path) -> Path:
    """Derive ``<name>.wav`` next to the input PDF."""
    return input_path.with_suffix(".wav")


def _check_espeak() -> None:
    """Warn early if espeak-ng is missing (Kokoro needs it to phonemize)."""
    if shutil.which("espeak-ng") is None and shutil.which("espeak") is None:
        print(
            "warning: espeak-ng not found on PATH. Kokoro needs it for "
            "phonemization — install with `brew install espeak-ng` "
            "(macOS) or your package manager.",
            file=sys.stderr,
        )


def _fmt_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s"


def _make_progress():
    """Return an ``on_progress(index, seconds)`` callback for synthesis.

    Animates a single in-place line on a real terminal; stays silent when output
    is redirected so logs don't fill with carriage returns.
    """
    if not sys.stderr.isatty():
        return None

    def report(index: int, seconds: float) -> None:
        print(
            f"\r  synthesizing… {index} segment(s), {_fmt_duration(seconds)} of audio",
            end="",
            file=sys.stderr,
            flush=True,
        )

    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdftoaudio",
        description="Convert a PDF into a spoken audio file (WAV or MP3).",
    )
    parser.add_argument("input", type=Path, help="path to the source PDF")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="output audio path (.wav or .mp3); defaults to <input>.wav",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    input_path: Path = args.input
    output_path: Path = args.output or _default_output(input_path)

    # Validate inputs before doing any expensive work.
    if not input_path.is_file():
        print(f"error: no such file: {input_path}", file=sys.stderr)
        return 1
    if input_path.suffix.lower() != ".pdf":
        print(f"error: input must be a PDF, got {input_path.suffix!r}", file=sys.stderr)
        return 1
    try:
        format_from_path(output_path)  # fail fast on bad extension
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _check_espeak()

    print(f"Extracting text from {input_path.name}…")
    pages = extract_pages(input_path)
    text = clean_pages(pages)
    if not text.strip():
        print("error: no readable text found in the PDF", file=sys.stderr)
        return 1
    print(f"Cleaned {len(text):,} characters across {len(pages)} page(s).")

    if model_is_cached():
        print("Synthesizing speech…")
    else:
        print("Synthesizing speech (first run downloads the ~350MB model)…")
    progress = _make_progress()
    try:
        duration = write_stream(
            synthesize(text), output_path, on_progress=progress
        )
    except ValueError as exc:
        if progress is not None:
            print(file=sys.stderr)  # end the progress line
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if progress is not None:
        print(file=sys.stderr)  # newline after the in-place progress line

    print(f"✓ Wrote {output_path} ({_fmt_duration(duration)} of audio).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
