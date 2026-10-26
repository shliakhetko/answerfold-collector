"""A1 arbiter: confirms/rejects the disputed set -- candidates named by only
one of the two primary extractors (a1_answer_coding.py's PRIMARY_EXTRACTORS)
-- with one small targeted call per run instead of a full re-extraction.

This module holds the shared arbiter types (ArbiterVote/ArbiterResult/
ArbiterBatchItem, parse_votes_json) plus the *Anthropic* implementation of
the vote call and its Message Batches. The live config (2026-09-02) runs
Claude + Gemini as PRIMARY_EXTRACTORS and OpenAI as the synchronous
ARBITER_EXTRACTOR (call_openai_arbiter_vote in openai_extractor.py) --
OpenAI's Batch API is unusable on this account, so the arbiter runs
synchronously over just the disputed runs. call_arbiter_vote /
submit_arbiter_batch below stay for an ARBITER_EXTRACTOR="anthropic"
reconfig.

Self-judging disclosure (spec, "Calibration and checks"): every extractor
vendor here also appears among the engines under test (see
anthropic_extractor.py). All voters grade their own vendor's collected
answers; the arbiter vote is only ever spent on candidates the two
primaries disagree on.

Never raises -- same adapter contract as every other call site in this repo.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from analysis.coding.extractors.models import EXTRACTOR_MODELS
from analysis.coding.extractors.prompts import build_arbiter_prompt
from collector.config import AnthropicConfig

MAX_TOKENS = 1024  # small: confirm/deny + is_recommended per candidate, no re-extraction

DEFAULT_POLL_INTERVAL_SECONDS = 30.0
DEFAULT_MAX_WAIT_SECONDS = 3600.0

# See anthropic_extractor.py's identical constant -- both values confirmed
# via a live smoke test (2026-08-26).
_ANTHROPIC_IN_PROGRESS_STATUS = "in_progress"

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass(frozen=True)
class ArbiterVote:
    surface_form: str
    confirmed: bool
    is_recommended: bool


@dataclass(frozen=True)
class ArbiterBatchItem:
    custom_id: str  # this repo uses run_id
    raw_response_text: str
    candidates: list[dict[str, str]]


@dataclass
class ArbiterResult:
    votes: list[ArbiterVote] = field(default_factory=list)
    model_returned: str | None = None
    error: str | None = None
    raw_envelope: str | None = None
    tokens_input: int | None = None
    tokens_output: int | None = None


class MessagesClient(Protocol):
    """Same structural type as extractors/anthropic_extractor.py's -- lets
    tests inject a fake object with a `.messages.create()` method."""
    messages: Any


def _build_client(config: AnthropicConfig) -> Any:
    import anthropic  # local import: a fake `client` injected by tests never
                       # needs the real `anthropic` package importable.
    return anthropic.Anthropic(
        api_key=config.api_key,
        timeout=config.timeout_seconds,
        max_retries=config.max_retries,
    )


def call_arbiter_vote(
    *,
    raw_response_text: str,
    candidates: list[dict[str, str]],
    config: AnthropicConfig | None = None,
    client: MessagesClient | None = None,
) -> ArbiterResult:
    """candidates: [{"surface_form":, "evidence_span":}, ...] -- the
    disputed set for one run (named by exactly one of the two primary
    extractors). One call covers every disputed candidate for that run, not
    one call per candidate."""
    if not candidates:
        return ArbiterResult()  # nothing to adjudicate -- never call the model for an empty set

    cfg = config or AnthropicConfig.from_env()
    active_client = client or _build_client(cfg)

    candidates_json = json.dumps(
        [{"surface_form": c["surface_form"], "evidence_span": c["evidence_span"]} for c in candidates]
    )
    kwargs: dict[str, Any] = {
        "model": EXTRACTOR_MODELS["anthropic"],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,  # spec: "All LLM calls: temperature 0"
        "messages": [{"role": "user", "content": build_arbiter_prompt(raw_response_text, candidates_json)}],
    }

    try:
        response = active_client.messages.create(**kwargs)
    except Exception as exc:  # contract: never raise past this point
        return ArbiterResult(error=f"{type(exc).__name__}: {exc}")

    return _parse_response(response)


def _parse_response(response: Any) -> ArbiterResult:
    parse_errors: list[str] = []

    def _safe(label: str, fn):
        try:
            return fn()
        except Exception as exc:
            parse_errors.append(f"{label}: {type(exc).__name__}: {exc}")
            return None

    model_returned = _safe("model_returned", lambda: getattr(response, "model", None))
    text = _safe("text", lambda: _extract_text(response))
    votes = (_safe("votes", lambda: parse_votes_json(text)) if text else []) or []
    usage = _safe("usage", lambda: getattr(response, "usage", None))
    tokens_input = getattr(usage, "input_tokens", None) if usage is not None else None
    tokens_output = getattr(usage, "output_tokens", None) if usage is not None else None

    return ArbiterResult(
        votes=votes,
        model_returned=model_returned,
        error="; ".join(parse_errors) or None,
        raw_envelope=text,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
    )


def parse_votes_json(text: str) -> list[ArbiterVote]:
    stripped = _CODE_FENCE_RE.sub("", text).strip()
    data = json.loads(stripped)
    return [
        ArbiterVote(
            surface_form=item["surface_form"],
            confirmed=bool(item["confirmed"]),
            is_recommended=bool(item["is_recommended"]),
        )
        for item in data
    ]


def _extract_text(response: Any) -> str | None:
    chunks: list[str] = []
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) != "text":
            continue
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)
    return "\n".join(chunks) if chunks else None


def _arbiter_params(raw_response_text: str, candidates: list[dict[str, str]]) -> dict[str, Any]:
    candidates_json = json.dumps(
        [{"surface_form": c["surface_form"], "evidence_span": c["evidence_span"]} for c in candidates]
    )
    return {
        "model": EXTRACTOR_MODELS["anthropic"],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "messages": [{"role": "user", "content": build_arbiter_prompt(raw_response_text, candidates_json)}],
    }


def submit_arbiter_batch(
    items: list[ArbiterBatchItem], *, config: AnthropicConfig | None = None, client: Any | None = None
) -> str:
    """One Anthropic Message Batch covering every run's disputed set (one
    request per run that had a disagreement, across the whole pipeline run)
    -- returns the batch id to poll/fetch."""
    cfg = config or AnthropicConfig.from_env()
    active_client = client or _build_client(cfg)
    requests = [
        {"custom_id": item.custom_id, "params": _arbiter_params(item.raw_response_text, item.candidates)}
        for item in items
    ]
    batch = active_client.messages.batches.create(requests=requests)
    return batch.id


def poll_arbiter_batch(
    batch_id: str,
    *,
    config: AnthropicConfig | None = None,
    client: Any | None = None,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
) -> str:
    """Same mechanics as anthropic_extractor.py's poll_anthropic_batch
    (duplicated, not imported -- matches this repo's established
    _build_client-per-module convention)."""
    cfg = config or AnthropicConfig.from_env()
    active_client = client or _build_client(cfg)
    elapsed = 0.0
    while True:
        batch = active_client.messages.batches.retrieve(batch_id)
        if batch.processing_status != _ANTHROPIC_IN_PROGRESS_STATUS:
            return batch.processing_status
        if elapsed >= max_wait_seconds:
            raise TimeoutError(
                f"Anthropic arbiter batch {batch_id} still {batch.processing_status} after {max_wait_seconds}s"
            )
        time.sleep(poll_interval_seconds)
        elapsed += poll_interval_seconds


def fetch_arbiter_batch_results(
    batch_id: str, *, config: AnthropicConfig | None = None, client: Any | None = None
) -> dict[str, ArbiterResult]:
    """Keyed by custom_id -- results are NOT guaranteed to come back in
    submission order (Anthropic's own documented behavior)."""
    cfg = config or AnthropicConfig.from_env()
    active_client = client or _build_client(cfg)
    results: dict[str, ArbiterResult] = {}
    for entry in active_client.messages.batches.results(batch_id):
        result_type = getattr(entry.result, "type", None)
        if result_type == "succeeded":
            results[entry.custom_id] = _parse_response(entry.result.message)
        else:
            results[entry.custom_id] = ArbiterResult(error=f"batch_result_{result_type}")
    return results
