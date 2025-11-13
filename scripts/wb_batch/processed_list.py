from __future__ import annotations

from pathlib import Path
from .filelock import append_with_lock, rewrite_excluding_with_lock
from .sanitizer import sanitize_rel_path

__all__ = ["ProcessedList"]


class ProcessedList:
    """Manage the processed_motions.txt with locking and fast membership checks."""

    def __init__(self, base_root: Path):
        self.base_root = base_root
        self.path = (base_root if base_root.is_dir() else base_root.parent) / "processed_motions.txt"
        self.processed_rel_set: set[str] = set()
        self.processed_stem_set: set[str] = set()

    def load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    for line in f:
                        entry = line.strip()
                        if entry:
                            sanitized = sanitize_rel_path(entry)
                            self.processed_rel_set.add(sanitized)
                            self.processed_stem_set.add(Path(sanitized).stem)
            except Exception as e:
                print(f"[WARN]: Failed to read processed list {self.path}: {e}")

    def append(self, rel_entry: str) -> None:
        sanitized = sanitize_rel_path(rel_entry)
        if sanitized not in self.processed_rel_set:
            append_with_lock(self.path, sanitized + "\n")
            self.processed_rel_set.add(sanitized)
            self.processed_stem_set.add(Path(sanitized).stem)

    def contains(self, rel_entry: str) -> bool:
        sanitized = sanitize_rel_path(rel_entry)
        return sanitized in self.processed_rel_set or Path(sanitized).stem in self.processed_stem_set

    def backup_and_remove(self, rels_to_remove: set[str]) -> Path | None:
        """Backup current processed list and write a new one with given entries removed."""
        if not self.path.exists() or not rels_to_remove:
            return None
        try:
            from datetime import datetime as _dt
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.path.with_name(f"{self.path.stem}.bak.{ts}{self.path.suffix}")
            self.path.rename(backup_path)
            remaining = [rel for rel in sorted(self.processed_rel_set) if rel not in rels_to_remove]
            with open(self.path, "w", encoding="utf-8") as f:
                for rel in remaining:
                    f.write(rel + "\n")
            self.processed_rel_set = set(remaining)
            self.processed_stem_set = {Path(r).stem for r in remaining}
            print(f"[INFO]: Backed up processed list to {backup_path} and removed {len(rels_to_remove)} entries.")
            return backup_path
        except Exception as e:
            print(f"[WARN]: Failed to backup/remove processed list entries: {e}")
            return None

    def remove(self, rel_entry: str) -> None:
        """Remove an entry from processed list."""
        sanitized = sanitize_rel_path(rel_entry)
        if sanitized in self.processed_rel_set:
            try:
                rewrite_excluding_with_lock(self.path, {sanitized})
            except Exception as e:
                print(f"[WARN]: Failed to rewrite processed list when removing {sanitized}: {e}")
            self.processed_rel_set.discard(sanitized)
            self.processed_stem_set.discard(Path(sanitized).stem)


