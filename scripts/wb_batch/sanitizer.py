from __future__ import annotations

from pathlib import Path
import re as _re

__all__ = [
    "sanitize_basename",
    "sanitize_rel_path",
    "encode_collection_name",
    "decode_collection_name",
]


def sanitize_basename(name: str) -> str:
    """Sanitize a filename (basename only) to be WandB-safe and avoid '__' sequences.
    - Remove '+' and '?' explicitly.
    - Replace any other disallowed characters with '_'.
    - Collapse consecutive underscores to a single '_' to avoid '__' inside filenames.
    """
    name = name.replace("+", "").replace("?", "")
    name = "".join(ch if (_re.match(r"[A-Za-z0-9._-]", ch)) else "_" for ch in name)
    name = _re.sub(r"_+", "_", name)
    return name


def sanitize_rel_path(rel_path: str) -> str:
    """Sanitize only the basename of a relative path string."""
    p = Path(rel_path)
    new_name = sanitize_basename(p.name)
    if new_name == p.name:
        return rel_path
    return str(p.with_name(new_name))


def encode_collection_name(rel_entry: str) -> str:
    """Encode a relative path into a WandB collection/artifact name."""
    return rel_entry.replace("/", "__").replace("\\", "__")


def decode_collection_name(collection: str) -> str:
    """Decode a collection name back to a relative path."""
    return collection.replace("__", "/")


