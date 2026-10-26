"""Shared types + JSON-parsing helper for A1's three extractors
(anthropic_extractor.py / openai_extractor.py / gemini_extractor.py).

Each extractor mirrors collector/engines/*.py's adapter contract: never
raises past its own boundary, errors become a populated `error` field
instead. Deterministic verification (evidence_span/surface_form literal-
substring checks) happens downstream in a1_answer_coding.py, not here --
these modules only ever ask the model and report what it said.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any


@dataclass(frozen=True)
class MentionCandidate:
    surface_form: str
    evidence_span: str
    is_recommended: bool


@dataclass(frozen=True)
class BatchItem:
    """One request inside a vendor Batch API submission -- custom_id is what
    correlates a result back to the run it came from once the batch
    completes (Anthropic/OpenAI: an explicit field on the request/result;
    Gemini: paired back up positionally by each vendor's own
    submit_*_batch/fetch_*_batch_results, see gemini_extractor.py)."""
    custom_id: str
    raw_response_text: str


def dict_to_namespace(value: Any) -> Any:
    """Recursively turns a plain dict/list (as returned by json.loads on a
    vendor Batch API's JSONL result file) into SimpleNamespace/list, so the
    existing attribute-access (`getattr(response, ...)`) parsing helpers in
    each extractor module work unchanged on both a live SDK response object
    and a replayed batch result -- one parser, two sources."""
    if isinstance(value, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [dict_to_namespace(v) for v in value]
    return value


@dataclass
class ExtractionResult:
    mentions: list[MentionCandidate] = field(default_factory=list)
    model_returned: str | None = None
    error: str | None = None
    raw_envelope: str | None = None
    tokens_input: int | None = None
    tokens_output: int | None = None


# Models sometimes wrap JSON in a markdown code fence despite being asked
# not to -- strip it before parsing rather than fail the whole extraction.
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

# A backslash that does not begin a valid JSON escape (\" \\ \/ \b \f \n \r
# \t \uXXXX). The cheap/fast extractor models occasionally emit a raw
# backslash inside an evidence_span ("\(", a stray "\ ", a Windows path)
# that makes json.loads reject the whole array.
_BAD_JSON_ESCAPE_RE = re.compile(r'\\(?![\\/"bfnrt]|u[0-9a-fA-F]{4})')

_REQUIRED_MENTION_KEYS = ("surface_form", "evidence_span", "is_recommended")


def _repair_json_escapes(s: str) -> str:
    return _BAD_JSON_ESCAPE_RE.sub(r"\\\\", s)


def _salvage_mention_objects(s: str) -> list[dict]:
    """Recover the leading complete `{...}` objects from a JSON array that
    json.loads rejects -- almost always output truncated at the model's
    max_tokens (an unterminated string in the final object) or a stray
    escape. Decodes object by object and stops at the first one that will
    not parse; every object before the break is kept."""
    start = s.find("[")
    if start == -1:
        return []
    decoder = json.JSONDecoder()
    out: list[dict] = []
    i = start + 1
    while i < len(s):
        while i < len(s) and s[i] in " \t\r\n,":
            i += 1
        if i >= len(s) or s[i] == "]":
            break
        try:
            obj, end = decoder.raw_decode(s, i)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict) and all(k in obj for k in _REQUIRED_MENTION_KEYS):
            out.append(obj)
        i = end
    return out


def parse_mentions_json(text: str) -> list[MentionCandidate]:
    """Parse one extractor's raw text output into MentionCandidates.

    Tolerant of the two ways the fast extractor models produce not-quite-JSON:
    an invalid backslash escape (repaired), and output truncated at max_tokens
    (the leading complete objects are salvaged). Still raises when nothing at
    all is recoverable -- callers wrap this in their own _safe() so one bad
    response degrades that extractor's result, not the whole A1 run."""
    stripped = _CODE_FENCE_RE.sub("", text).strip()
    data: Any = None
    for candidate in (stripped, _repair_json_escapes(stripped)):
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if data is None:
        data = _salvage_mention_objects(_repair_json_escapes(stripped))
        if not data:
            json.loads(stripped)  # nothing salvageable -- re-raise the real error
    return [
        MentionCandidate(
            surface_form=item["surface_form"],
            evidence_span=item["evidence_span"],
            is_recommended=bool(item["is_recommended"]),
        )
        for item in data
    ]
