from __future__ import annotations

from typing import Iterable, Iterator, Tuple, Any

try:
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover
    _tqdm = None

__all__ = ["progress_iter", "drain_results"]


def progress_iter(iterable: Iterable, total: int | None = None, desc: str | None = None):
    """tqdm wrapper with graceful fallback."""
    if _tqdm is None:
        return iterable
    kwargs: dict[str, Any] = {}
    if total is not None:
        kwargs["total"] = total
    if desc is not None:
        kwargs["desc"] = desc
    return _tqdm(iterable, **kwargs)


def drain_results(pairs: list[Tuple[str, "Any"]], total: int | None = None, desc: str | None = None) -> Iterator[tuple[str, bool]]:
    """Drain a list of (key, ApplyResult-like) pairs with progress, yielding (key, ok)."""
    for key, res in progress_iter(pairs, total=total if total is not None else len(pairs), desc=desc):
        ok = False
        try:
            ok = bool(res.get())
        except Exception:
            ok = False
        yield key, ok
