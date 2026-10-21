# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from defakto.errors import UnsupportedMediaError

EXTENSIONS: dict[str, set[str]] = {
    "text": {".txt", ".md", ".text"},
    "image": {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"},
    "audio": {".wav", ".mp3", ".flac", ".m4a", ".ogg"},
    "video": {".mp4", ".mov", ".avi", ".mkv", ".webm"},
}
SUPPORTED_MEDIA_TYPES = ("text",)


@dataclass(frozen=True)
class Media:
    """An input ready for an encoder: its media type, its content, and where it came from."""

    media_type: str
    content: Any
    name: str | None = None


def media_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    for media_type, suffixes in EXTENSIONS.items():
        if suffix in suffixes:
            return media_type
    known = ", ".join(sorted(s for suffixes in EXTENSIONS.values() for s in suffixes))
    raise UnsupportedMediaError(
        f"Can't tell the media type of '{path.name}' from its extension. "
        f"Pass media_type explicitly, or use one of: {known}"
    )


def load_media(source: str | Path, media_type: str | None = None) -> Media:
    """Turn a file path or a string of text into ``Media``.

    A ``Path``, or a string naming an existing file, is read as a file. Any other string
    is treated as the text to analyze.
    """
    path = _as_path(source)
    if path is None:
        if media_type not in (None, "text"):
            raise UnsupportedMediaError(
                f"Got a string, but media_type is '{media_type}'. Pass a file path instead."
            )
        return Media(media_type="text", content=str(source))

    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")
    detected = media_type or media_type_for(path)
    if detected not in SUPPORTED_MEDIA_TYPES:
        raise UnsupportedMediaError(
            f"{detected.capitalize()} analysis isn't available yet in this version of defakto "
            f"(supported: {', '.join(SUPPORTED_MEDIA_TYPES)})."
        )
    return Media(media_type="text", content=path.read_text(encoding="utf-8"), name=str(path))


def _as_path(source: str | Path) -> Path | None:
    if isinstance(source, Path):
        return source.expanduser()
    if len(source) > 1024 or "\n" in source:
        return None
    candidate = Path(source).expanduser()
    try:
        if candidate.is_file():
            return candidate
    except OSError:
        return None
    looks_like_file = not any(c.isspace() for c in source) and any(
        candidate.suffix.lower() in suffixes for suffixes in EXTENSIONS.values()
    )
    return candidate if looks_like_file else None
