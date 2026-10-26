"""Disk-backed cache for Phase A LLM calls (Post-Collection Coding Pipeline
Spec: "Every LLM call cached to disk, keyed by input hash plus model plus
prompt_version. Reruns hit cache.").

One JSON file per cache key, gitignored under data/03_coded/llm_cache/ --
plain files rather than a diskcache/joblib dependency: at the scale this
pipeline runs at (dozens to low hundreds of distinct (stage, model,
prompt_version, input) calls, not millions), a directory of JSON files is
trivially inspectable/diffable by hand, matching this repo's existing
"no dependency where a few lines of stdlib solves it" style (see
collector/config.py).

Analysis-layer code (INSTRUMENT.md section 12: analysis/ may import from
collector/, never the reverse) -- this module imports nothing from collector/.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def cache_key(*, input_text: str, model: str, prompt_version: str) -> str:
    """Stable key for one LLM call. Order-sensitive components joined with a
    NUL byte (never appears in normal text, unlike a printable separator that
    could itself appear inside input_text) so a change in any one component
    always changes the key, even if concatenation without a separator would
    coincidentally collide."""
    payload = "\x00".join((model, prompt_version, input_text)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class LLMCache:
    """cache_dir is the per-extractor/per-stage subdirectory (e.g.
    data/03_coded/llm_cache/anthropic_extractor/) -- callers namespace by
    passing a different cache_dir per stage, not by prefixing keys."""

    cache_dir: Path

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    def set(self, key: str, value: dict[str, Any]) -> None:
        """value should include the parsed result plus tokens_input/
        tokens_output so calibration.measure_actual_token_cost can read real
        cost straight from cache entries without re-calling anything."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)
        tmp_path.replace(path)  # atomic on POSIX: no reader ever sees a partial file

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"
