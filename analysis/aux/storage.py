"""File storage for the auxiliary batch: append-only JSONL, raw blobs, and
the idempotency check every collector uses to fetch only new rows.

Nothing here writes to SQLite -- the study rule is "store raw, derive later",
and the derived shape (blocked booleans, status buckets, domain classes)
belongs to Phase A. Keeping auxiliary data entirely in files also means the
collector schema and the three existing notebooks are untouched.

Layout under data/02_aux/ (the root is injected, tests use tmp_path):

    <collector>/index.jsonl              one row per collected unit
    <collector>/<key>/...                raw payloads (txt/html/xml/json)
    <collector>/bodies/<sha>.html        capped page bodies

Analysis-layer code (INSTRUMENT.md section 12).
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Callable, Iterable

from collector import paths
DEFAULT_AUX_ROOT = paths.AUX_DIR

_WRITE_LOCK = threading.Lock()


def url_key(url: str) -> str:
    """Stable filesystem-safe key for a URL (Task 5 body blobs, Task 7
    pages). sha256 rather than a slug so pathological URLs can't collide or
    blow the path length limit."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def domain_key(domain: str) -> str:
    """A roster domain, normalized for use as a directory name."""
    d = domain.strip().lower()
    if d.startswith("http://") or d.startswith("https://"):
        from urllib.parse import urlsplit

        d = urlsplit(d).hostname or d
    if d.startswith("www."):
        d = d[4:]
    return d.strip("/").replace("/", "_") or "unknown"


class AuxStore:
    def __init__(self, root: Path | str = DEFAULT_AUX_ROOT) -> None:
        self.root = Path(root)

    # -- paths --------------------------------------------------------

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def ensure_parent(self, p: Path) -> Path:
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # -- JSONL index (append-only) ----------------------------------

    def append_record(self, index_path: Path, record: dict[str, Any]) -> None:
        self.ensure_parent(index_path)
        with _WRITE_LOCK:
            with index_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str))
                f.write("\n")
                f.flush()

    def read_records(self, index_path: Path) -> list[dict[str, Any]]:
        if not index_path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def done_keys(self, index_path: Path, key_fn: Callable[[dict[str, Any]], Any]) -> set[Any]:
        """Keys already present in an index file -- a collector skips these
        unless --force. `key_fn` extracts the idempotency key from a row."""
        keys: set[Any] = set()
        for record in self.read_records(index_path):
            try:
                keys.add(key_fn(record))
            except Exception:
                continue
        return keys

    # -- raw blobs --------------------------------------------------

    def write_bytes(self, rel: str | Path, data: bytes) -> Path:
        p = self.ensure_parent(self.root / rel)
        p.write_bytes(data)
        return p

    def write_text(self, rel: str | Path, text: str) -> Path:
        p = self.ensure_parent(self.root / rel)
        p.write_text(text, encoding="utf-8")
        return p

    def write_json(self, rel: str | Path, obj: Any) -> Path:
        p = self.ensure_parent(self.root / rel)
        p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return p


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)
