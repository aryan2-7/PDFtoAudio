# PDFtoAudio

A Python CLI tool that converts PDFs into spoken audio using [PyMuPDF](https://pymupdf.readthedocs.io/)
for text extraction and [Kokoro](https://github.com/hexgrad/kokoro) neural TTS for speech.

- **Academic-grade text cleaning** — de-hyphenates line breaks, strips repeating
  headers/footers and page numbers, folds ligatures, and removes citation markers.
- **Streaming synthesis** — audio is written chunk-by-chunk, so book-length PDFs
  convert without holding the whole audiobook in memory.
- **WAV or MP3 output** — chosen from the output file extension (no external encoder).
- Runs on CPU, or GPU-accelerated on Apple Silicon (MPS) / CUDA automatically.

## Requirements

- Python **>= 3.14**
- [uv](https://docs.astral.sh/uv/)
- **espeak-ng** — Kokoro's phonemizer (a system dependency):

  ```bash
  # macOS
  brew install espeak-ng
  # Debian/Ubuntu
  sudo apt-get install espeak-ng
  ```

## Install

## A proper install link will be provided after the project is finished

```bash
uv sync
```

The Kokoro model (~350 MB) downloads automatically from Hugging Face on first run
and is cached locally afterward.

## Usage

```bash
# Basic: writes book.wav next to the input
uv run pdftoaudio book.pdf

# Choose an output path and format (.wav or .mp3)
uv run pdftoaudio book.pdf audiobook.mp3
```

Defaults: voice `af_heart`, American English, speed `1.0`.

## Development

```bash
uv run pytest        # unit tests for the text-cleaning core (no model needed)
```

### Layout

| Module | Responsibility |
| --- | --- |
| `pdftoaudio/extract.py` | PyMuPDF → positioned text blocks |
| `pdftoaudio/clean.py`   | Pure regex normalization → speakable text |
| `pdftoaudio/tts.py`     | Kokoro wrapper → stream of NumPy audio chunks |
| `pdftoaudio/audio.py`   | Streaming WAV/MP3 writer via soundfile |
| `pdftoaudio/cli.py`     | argparse entry point orchestrating the pipeline |
