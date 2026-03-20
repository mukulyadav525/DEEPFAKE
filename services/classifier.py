"""
services/classifier.py
Classifies uploaded file by extension into one of our supported modalities.
"""

EXTENSION_MAP = {
    # Images
    ".jpg": "image", ".jpeg": "image", ".png": "image",
    ".webp": "image", ".bmp": "image", ".tiff": "image", ".gif": "image",

    # Audio
    ".mp3": "audio", ".wav": "audio", ".ogg": "audio",
    ".flac": "audio", ".m4a": "audio", ".aac": "audio",

    # Video
    ".mp4": "video", ".mov": "video", ".avi": "video",
    ".mkv": "video", ".webm": "video",

    # Documents
    ".pdf": "pdf", ".docx": "pdf", ".txt": "pdf",
}


def classify_file(suffix: str) -> str | None:
    """Returns modality string or None if unsupported."""
    return EXTENSION_MAP.get(suffix.lower())
