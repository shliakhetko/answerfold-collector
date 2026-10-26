"""Deterministic company-name matcher (no model, no AI call).

Used ONLY for UI calibration (see ledger entry
`ui-calibration-design-2026-09-29`): the same matcher runs over UI answer
text and over API `raw_response_text`, so the two sides are counted under
identical rules. It is NOT how the study's own API mentions were made
(those come from the coding pipeline); it is a like-for-like yardstick.

Rules: names are the canonical_name plus every surface form of each
in-frame entity; text and names are NFKC-normalized with whitespace
collapsed; a name matches on non-alphanumeric boundaries. Names of 5 or
more characters match case-insensitively; shorter names match
case-sensitively (short names collide with ordinary words).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

CASE_INSENSITIVE_MIN_LEN = 5


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


@dataclass(frozen=True)
class CompiledName:
    entity_id: str
    pattern: re.Pattern[str]


def build_matcher(entities: list[dict[str, Any]]) -> list[CompiledName]:
    compiled: list[CompiledName] = []
    for entity in entities:
        if entity.get("meets_frame") != "in":
            continue
        names = {normalize(entity.get("canonical_name") or "")}
        names.update(normalize(sf) for sf in (entity.get("surface_forms") or "").split(";"))
        for name in sorted(n for n in names if n):
            flags = re.IGNORECASE if len(name) >= CASE_INSENSITIVE_MIN_LEN else 0
            pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", flags)
            compiled.append(CompiledName(entity["entity_id"], pattern))
    return compiled


def match_entities(text: str, matcher: list[CompiledName]) -> set[str]:
    text = normalize(text or "")
    return {c.entity_id for c in matcher if c.pattern.search(text)}
