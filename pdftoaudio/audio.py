"""Stream synthesized audio chunks to a WAV or MP3 file.

Uses ``soundfile`` in streaming write mode so a whole audiobook never has to be
held in memory at once (24 kHz mono float32 is ~5.7 MB per minute). The output
container is chosen from the file extension; libsndfile writes both WAV and MP3
natively so no external encoder is needed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import soundfile as sf

from . import SAMPLE_RATE

# Map file extensions to libsndfile container formats.
_EXT_FORMAT = {
    ".wav": "WAV",
    ".mp3": "MP3",
}

# libsndfile subtype per container. WAV gets 16-bit PCM (universal); MP3 uses a
# VBR quality level, which its writer maps to a sensible bitrate.
_FORMAT_SUBTYPE = {
    "WAV": "PCM_16",
    "MP3": "MPEG_LAYER_III",
}


def format_from_path(path: str | Path) -> str:
    """Resolve the libsndfile format name from a file extension.

    Raises:
        ValueError: if the extension is not a supported audio container, or the
            installed libsndfile cannot write it (e.g. an old build without MP3).
    """
    suffix = Path(path).suffix.lower()
    fmt = _EXT_FORMAT.get(suffix)
    if fmt is None:
        supported = ", ".join(sorted(_EXT_FORMAT))
        raise ValueError(
            f"unsupported output extension {suffix!r}; use one of: {supported}"
        )
    if fmt not in sf.available_formats():
        raise ValueError(
            f"this libsndfile build cannot write {fmt}; output to .wav instead"
        )
    return fmt


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def write_stream(
    chunks: Iterable[np.ndarray],
    out_path: str | Path,
    gap_seconds: float = 0.3,
    lead_seconds: float = 0.25,
    tail_seconds: float = 0.5,
    on_progress: Callable[[int, float], None] | None = None,
) -> float:
    """Write audio chunks to ``out_path`` with consistent silence padding.

    Chunks arrive already trimmed of Kokoro's variable trailing silence (see
    :func:`pdftoaudio.tts.trim_silence`), so pacing is imposed here: a short
    lead-in, a uniform ``gap`` between segments, and a tail so the file never
    ends abruptly on the last word.

    Args:
        chunks: Iterable of 1-D float32 mono arrays at :data:`SAMPLE_RATE`.
        out_path: Destination ``.wav`` or ``.mp3`` file.
        gap_seconds: Silence inserted between consecutive segments.
        lead_seconds: Silence before the first segment.
        tail_seconds: Silence after the last segment.
        on_progress: Optional callback invoked per segment with
            ``(segment_index, seconds_written_so_far)`` for progress reporting.

    Returns:
        Total duration written, in seconds.

    Raises:
        ValueError: if the extension is unsupported or no audio was produced.
    """
    path = Path(out_path)
    fmt = format_from_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    gap = _silence(gap_seconds) if gap_seconds > 0 else None
    frames = 0
    wrote_any = False

    with sf.SoundFile(
        str(path),
        mode="w",
        samplerate=SAMPLE_RATE,
        channels=1,
        format=fmt,
        subtype=_FORMAT_SUBTYPE[fmt],
    ) as out:
        for index, chunk in enumerate(chunks):
            if not wrote_any:
                if lead_seconds > 0:
                    out.write(_silence(lead_seconds))
                    frames += int(lead_seconds * SAMPLE_RATE)
            elif gap is not None:
                out.write(gap)
                frames += gap.size
            out.write(chunk)
            frames += chunk.size
            wrote_any = True
            if on_progress is not None:
                on_progress(index + 1, frames / SAMPLE_RATE)

        if wrote_any and tail_seconds > 0:
            out.write(_silence(tail_seconds))
            frames += int(tail_seconds * SAMPLE_RATE)

    if not wrote_any:
        path.unlink(missing_ok=True)
        raise ValueError("no audio was produced (empty or unreadable input text)")

    return frames / SAMPLE_RATE
