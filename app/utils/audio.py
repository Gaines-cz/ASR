"""Audio utilities for format detection and encoding."""

import base64
import io
from pathlib import Path

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

SUPPORTED_FORMATS = {"wav", "mp3", "m4a", "ogg", "webm", "flac"}

# MIME type to format mapping
MIME_TO_FORMAT = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mp3": "mp3",
    "audio/mpeg": "mp3",
    "audio/m4a": "m4a",
    "audio/ogg": "ogg",
    "audio/webm": "webm",
    "audio/flac": "flac",
}


def detect_audio_format(filename: str, mime_type: str) -> str:
    """Detect audio format from filename and MIME type.

    Args:
        filename: The audio file name
        mime_type: The MIME type of the audio

    Returns:
        The detected format string (e.g., 'wav', 'mp3')
    """
    # Try to get format from MIME type
    if mime_type in MIME_TO_FORMAT:
        return MIME_TO_FORMAT[mime_type]

    # Fall back to extension
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext in SUPPORTED_FORMATS:
        return ext

    # Default fallback
    return "wav"


def validate_audio_format(format: str) -> bool:
    """Validate if the audio format is supported.

    Args:
        format: The audio format to validate

    Returns:
        True if supported, False otherwise
    """
    return format.lower() in SUPPORTED_FORMATS


def encode_audio_to_data_url(audio_bytes: bytes, format: str) -> str:
    """Encode audio bytes to a data URL.

    Args:
        audio_bytes: Raw audio bytes
        format: Audio format (e.g., 'wav', 'mp3')

    Returns:
        Data URL string: data:audio/{format};base64,{base64_content}
    """
    base64_content = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:audio/{format};base64,{base64_content}"


def get_mime_type(format: str) -> str:
    """Get MIME type for a given audio format.

    Args:
        format: Audio format (e.g., 'wav', 'mp3')

    Returns:
        The corresponding MIME type
    """
    format_lower = format.lower()
    for mime, fmt in MIME_TO_FORMAT.items():
        if fmt == format_lower:
            return mime
    return "audio/wav"


def convert_to_wav(audio_bytes: bytes, source_format: str) -> bytes:
    """Convert audio to WAV format.

    Args:
        audio_bytes: Raw audio bytes
        source_format: Source format (e.g., 'webm', 'ogg')

    Returns:
        WAV encoded audio bytes

    Raises:
        ImportError: If pydub is not available
        Exception: If conversion fails
    """
    if not PYDUB_AVAILABLE:
        raise ImportError("pydub is required for audio conversion. Install with: pip install pydub")

    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=source_format)
    output = io.BytesIO()
    audio.export(output, format="wav")
    return output.getvalue()
