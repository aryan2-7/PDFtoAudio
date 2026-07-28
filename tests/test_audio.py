"""Unit tests for the audio writer and silence trimming.

These use synthetic NumPy audio, so they run without torch, kokoro, or the model
download — the whole streaming/padding path is still exercised.
"""

import numpy as np
import pytest
import soundfile as sf

from pdftoaudio import SAMPLE_RATE
from pdftoaudio.audio import format_from_path, write_stream
from pdftoaudio.tts import trim_silence


# --- trim_silence ------------------------------------------------------------


def test_trim_removes_leading_and_trailing_silence():
    speech = np.array([0.5, -0.4, 0.3], dtype=np.float32)
    padded = np.concatenate([np.zeros(100, np.float32), speech, np.zeros(200, np.float32)])
    assert np.array_equal(trim_silence(padded), speech)


def test_trim_all_silence_returns_none():
    assert trim_silence(np.zeros(1000, dtype=np.float32)) is None


def test_trim_keeps_interior_quiet_samples():
    # A quiet sample between two loud ones must be preserved.
    audio = np.array([0.5, 0.0, 0.5], dtype=np.float32)
    assert np.array_equal(trim_silence(audio), audio)


# --- format_from_path --------------------------------------------------------


def test_format_from_path_wav_and_mp3():
    assert format_from_path("out.wav") == "WAV"
    assert format_from_path("OUT.MP3") == "MP3"


def test_format_from_path_rejects_unknown():
    with pytest.raises(ValueError, match="unsupported output extension"):
        format_from_path("out.ogg")


# --- write_stream ------------------------------------------------------------


def _tone(seconds: float) -> np.ndarray:
    return np.full(int(seconds * SAMPLE_RATE), 0.2, dtype=np.float32)


def test_write_stream_pads_and_reports_duration(tmp_path):
    out = tmp_path / "out.wav"
    events = []
    duration = write_stream(
        [_tone(1.0), _tone(1.0)],
        out,
        gap_seconds=0.3,
        lead_seconds=0.25,
        tail_seconds=0.5,
        on_progress=lambda i, s: events.append((i, s)),
    )
    # lead + chunk + gap + chunk + tail = 0.25 + 1 + 0.3 + 1 + 0.5 = 3.05s
    assert duration == pytest.approx(3.05, abs=0.01)
    assert sf.info(str(out)).duration == pytest.approx(3.05, abs=0.02)
    assert [i for i, _ in events] == [1, 2]  # one callback per segment


def test_write_stream_empty_raises_and_removes_file(tmp_path):
    out = tmp_path / "empty.wav"
    with pytest.raises(ValueError, match="no audio was produced"):
        write_stream([], out)
    assert not out.exists()
