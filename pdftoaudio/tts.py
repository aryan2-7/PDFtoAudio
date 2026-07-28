"""Kokoro neural TTS wrapper.

Wraps Kokoro's ``KPipeline`` behind a small streaming generator so the rest of
the app deals only in NumPy audio chunks and never imports torch or kokoro
directly. Chunks are yielded one paragraph at a time, keeping memory flat for
book-length inputs.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator
from pathlib import Path

import numpy as np

# Default voice/language. Kept as module constants rather than CLI flags to keep
# the v1 interface minimal (see plan). lang_code 'a' == American English and must
# match the voice prefix ('a' voices).
DEFAULT_VOICE = "af_heart"
DEFAULT_LANG_CODE = "a"
DEFAULT_SPEED = 1.0

# The official Kokoro-82M weights on the Hugging Face Hub. Passing this
# explicitly avoids Kokoro's "defaulting repo_id" notice.
_REPO_ID = "hexgrad/Kokoro-82M"

# Kokoro splits its input on this pattern; our cleaner emits paragraphs joined by
# blank lines, so we chunk on paragraph boundaries.
_SPLIT_PATTERN = r"\n\n+"


def _hf_cache_root() -> Path:
    """Resolve the Hugging Face hub cache directory without importing hf."""
    if hub := os.environ.get("HF_HUB_CACHE"):
        return Path(hub)
    if home := os.environ.get("HF_HOME"):
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def model_is_cached(voice: str = DEFAULT_VOICE) -> bool:
    """True if the model weights and the given voice are fully downloaded.

    Checks the Hugging Face cache on disk without importing ``huggingface_hub``,
    so it is safe to call before deciding whether to run offline.
    """
    cache_dir = "models--" + _REPO_ID.replace("/", "--")
    snapshots = _hf_cache_root() / cache_dir / "snapshots"
    if not snapshots.is_dir():
        return False
    needed = ("config.json", "kokoro-v1_0.pth", f"voices/{voice}.pt")
    return any(
        all((snap / rel).exists() for rel in needed) for snap in snapshots.iterdir()
    )


def _enable_offline_if_cached(voice: str) -> None:
    """Use the cached model without contacting the hub, when it's complete.

    Skips a per-run network round-trip once the weights are downloaded — and with
    it the hub's "unauthenticated requests" notice, which is emitted from
    compiled code and cannot be filtered by Python's warnings machinery. Falls
    back to online mode (first run, or an incomplete cache) so downloads still
    work. Must run before kokoro imports ``huggingface_hub``, which reads
    ``HF_HUB_OFFLINE`` at import time.
    """
    if os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE"):
        return
    if model_is_cached(voice):
        os.environ["HF_HUB_OFFLINE"] = "1"


def _select_device() -> str:
    """Pick the fastest available torch device for this machine."""
    import torch

    if torch.backends.mps.is_available():
        # Kokoro uses a few ops without MPS kernels; fall back to CPU for those
        # instead of crashing.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _to_numpy(audio) -> np.ndarray | None:
    """Coerce a Kokoro audio chunk to a 1-D float32 NumPy array."""
    if audio is None:
        return None
    if hasattr(audio, "detach"):  # torch.Tensor
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    return audio if audio.size else None


def trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray | None:
    """Trim leading/trailing near-silence from a chunk.

    Kokoro pads each segment with ~460 ms of trailing silence, which varies per
    chunk and stacks with the writer's own inter-segment gap into overlong
    pauses. Trimming to the speech bounds lets :mod:`pdftoaudio.audio` impose one
    consistent gap instead. Returns ``None`` if the chunk is entirely silent.
    """
    loud = np.flatnonzero(np.abs(audio) > threshold)
    if loud.size == 0:
        return None
    return audio[loud[0] : loud[-1] + 1]


def synthesize(
    text: str,
    voice: str = DEFAULT_VOICE,
    lang_code: str = DEFAULT_LANG_CODE,
    speed: float = DEFAULT_SPEED,
    device: str | None = None,
) -> Iterator[np.ndarray]:
    """Stream synthesized audio for ``text``, one chunk (paragraph) at a time.

    Args:
        text: Cleaned, speakable text with paragraphs separated by blank lines.
        voice: Kokoro voice id; its language prefix must match ``lang_code``.
        lang_code: Kokoro single-char language code (``'a'`` = American English).
        speed: Playback speed multiplier.
        device: Force a torch device; ``None`` auto-selects mps/cuda/cpu.

    Yields:
        1-D float32 NumPy arrays of 24 kHz mono samples. Empty chunks are skipped.
    """
    if not text.strip():
        return

    # Must precede the kokoro import: huggingface_hub reads HF_HUB_OFFLINE once,
    # at its own import time.
    _enable_offline_if_cached(voice)

    # Imported lazily so unit tests and `--help` don't pay the torch import cost.
    from kokoro import KPipeline

    # Kokoro's model init (LSTM/weight_norm notices) and istftnet inference (an
    # out-tensor resize notice attributed to kokoro, not torch) emit torch
    # warnings we can't act on. Suppress them precisely around model init and
    # inference; genuine warnings elsewhere (e.g. soundfile) stay visible.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline = KPipeline(
            lang_code=lang_code, repo_id=_REPO_ID, device=device or _select_device()
        )
        for result in pipeline(
            text, voice=voice, speed=speed, split_pattern=_SPLIT_PATTERN
        ):
            chunk = _to_numpy(getattr(result, "audio", None))
            if chunk is None:
                continue
            chunk = trim_silence(chunk)
            if chunk is not None:
                yield chunk
