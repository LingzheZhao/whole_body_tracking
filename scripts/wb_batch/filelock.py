from __future__ import annotations

from pathlib import Path
import os

__all__ = ["append_with_lock", "rewrite_excluding_with_lock"]


def append_with_lock(file_path: Path, text: str) -> None:
    """Append a line to file with an exclusive lock (best-effort, POSIX)."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        try:
            import fcntl  # POSIX only
            fcntl.flock(f, fcntl.LOCK_EX)
        except Exception:
            pass
        f.write(text)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
        try:
            import fcntl  # re-import inside scope for mypy
            fcntl.flock(f, fcntl.LOCK_UN)
        except Exception:
            pass


def rewrite_excluding_with_lock(file_path: Path, rels_to_remove: set[str]) -> None:
    """Rewrite file excluding given entries with an exclusive lock (best-effort, POSIX)."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[str] = []
    try:
        with open(file_path, "r", encoding="utf-8") as rf:
            existing = [line.rstrip("\n") for line in rf if line.strip()]
    except FileNotFoundError:
        existing = []
    remaining = [line for line in existing if line not in rels_to_remove]
    with open(file_path, "w", encoding="utf-8") as f:
        try:
            import fcntl  # POSIX only
            fcntl.flock(f, fcntl.LOCK_EX)
        except Exception:
            pass
        for line in remaining:
            f.write(line + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
        try:
            import fcntl  # re-import inside scope for mypy
            fcntl.flock(f, fcntl.LOCK_UN)
        except Exception:
            pass


