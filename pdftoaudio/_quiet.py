"""Silence noisy third-party import/runtime chatter for a clean CLI.

The model stack (torch, PyMuPDF/SWIG, huggingface_hub, transformers) emits a
handful of deprecation and info warnings we can neither fix nor act on. They
clutter the terminal and their long, wrapped lines look like broken output. This
module mutes exactly those sources — nothing from our own code — and must run
*before* the heavy libraries are imported.
"""

from __future__ import annotations

import logging
import os
import warnings

_CONFIGURED = False


def silence_third_party_noise() -> None:
    """Install warning filters, env vars and logger levels. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    # Quiet the Hugging Face hub: no download progress bars and no telemetry.
    # HF_HUB_DISABLE_XET bypasses the hf_xet Rust transfer library, whose
    # "unauthenticated requests" notice is printed from compiled code and cannot
    # be filtered by Python's warnings/logging machinery.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Deprecation/future warnings are never actionable for a CLI end-user; the
    # SWIG (PyMuPDF) import notices and torch's weight_norm notice live here.
    # (Kokoro's inference-time torch UserWarnings are suppressed at their source
    # in pdftoaudio.tts, so genuine UserWarnings elsewhere stay visible.)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Loggers used by the model/download stack. ERROR keeps real failures visible.
    for name in ("huggingface_hub", "transformers", "torch", "urllib3", "filelock"):
        logging.getLogger(name).setLevel(logging.ERROR)
