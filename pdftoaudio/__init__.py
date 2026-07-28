"""PDFtoAudio — convert PDFs into spoken audio using PyMuPDF and Kokoro TTS."""

__version__ = "0.1.0"

# Kokoro emits audio at a fixed 24 kHz sample rate.
SAMPLE_RATE = 24000
