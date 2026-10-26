"""OpenAI ('ChatGPT') Responses-API adapter.

Follows the engine adapter contract, INSTRUMENT.md section 7:
  1. Never raises past this module's boundary -- every failure becomes a
     populated `error` field on an otherwise-complete AdapterResult.
  2. Never interprets -- `raw_response_text` is passed through unmodified.
  3. Always reports whether search fired (tri-state; see types.AdapterResult).

Storage-ignorant (section 12): this module never imports collector.storage.
Orchestration (run_id, instrument_commit, DB write) lives in
collector/runner/execute.py.

Tool type and output shapes are pinned against the installed `openai` SDK,
not a live response -- re-verify before a real run (section 6; README's
vendor behaviour log). Extraction below is defensive (getattr with
defaults) because this shape may drift.
"""
from __future__ import annotations

import json
import time
from typing import Any, Protocol
from urllib.parse import urlparse

from collector.config import OpenAIConfig
from collector.engines.types import AdapterResult, Source

# Vendor implementation detail, not an experiment parameter -- named once
# here rather than inlined in the call. Not an env var: which tool to use is
# an adapter fact, not something an experiment cell configures (contrast with
# model_requested, which IS experiment-declared per section 6).
WEB_SEARCH_TOOL_TYPE = "web_search"

# Pinned buyer geography for this study (INSTRUMENT.md section 6): every
# engine is anchored to the same US-based buyer so the methodology can state
# one rule instead of one per vendor. Left unset, OpenAI infers this from the
# request's origin IP -- pinning it explicitly means the row no longer
# depends on where the instrument happens to run. No separate language
# field on this tool; country is the only lever here.
WEB_SEARCH_USER_LOCATION_COUNTRY = "US"


class ResponsesClient(Protocol):
    """Structural type for the OpenAI client's public surface used here --
    lets tests inject a fake object with a `.responses.create()` method, no
    network and no real package import required for the fake path."""
    responses: Any


def _build_client(config: OpenAIConfig) -> Any:
    import openai  # local import: a fake `client` injected by tests never
                    # needs the real `openai` package importable.
    return openai.OpenAI(
        api_key=config.api_key,
        timeout=config.timeout_seconds,
        max_retries=config.max_retries,
    )


def call_chatgpt(
    *,
    prompt_text: str,
    model_requested: str,
    config: OpenAIConfig | None = None,
    client: ResponsesClient | None = None,
) -> AdapterResult:
    """Send one prompt to the OpenAI Responses API. Never raises.

    `model_requested` is the pinned identifier for this experiment cell
    (EXPERIMENT.md / INSTRUMENT.md section 6) -- this function never chooses
    or defaults a model itself.

    Setup errors (missing API key -> ConfigError; client construction
    failure) intentionally propagate as exceptions -- they mean collection
    cannot start, which is not the same failure mode as a request that was
    attempted and failed (section 7.1 covers the latter only).
    """
    cfg = config or OpenAIConfig.from_env()
    active_client = client or _build_client(cfg)

    kwargs: dict[str, Any] = {
        "model": model_requested,
        "input": prompt_text,
        "tools": [
            {
                "type": WEB_SEARCH_TOOL_TYPE,
                "user_location": {
                    "type": "approximate",
                    "country": WEB_SEARCH_USER_LOCATION_COUNTRY,
                },
            }
        ],
    }
    if cfg.max_output_tokens is not None:
        kwargs["max_output_tokens"] = cfg.max_output_tokens

    start = time.monotonic()
    try:
        response = active_client.responses.create(**kwargs)
    except Exception as exc:  # contract: never raise past this point
        latency_ms = int((time.monotonic() - start) * 1000)
        return AdapterResult(
            error=f"{type(exc).__name__}: {exc}",
            latency_ms=latency_ms,
            request_payload=kwargs,
        )
    latency_ms = int((time.monotonic() - start) * 1000)

    return _parse_response(response, latency_ms=latency_ms, request_payload=kwargs)


def _parse_response(response: Any, *, latency_ms: int, request_payload: dict[str, Any]) -> AdapterResult:
    """Each field is extracted independently so one malformed field (a
    likely failure mode given the vendor-shape uncertainty flagged above)
    degrades that field to None/empty rather than blanking the whole row."""
    parse_errors: list[str] = []

    def _safe(label: str, fn):
        try:
            return fn()
        except Exception as exc:
            parse_errors.append(f"{label}: {type(exc).__name__}: {exc}")
            return None

    model_returned = _safe("model_returned", lambda: getattr(response, "model", None))
    raw_response_text = _safe("raw_response_text", lambda: _extract_output_text(response))
    fanout = _safe("search_fired/fan_out_queries", lambda: _extract_search_fanout(response))
    search_fired, fan_out_queries = fanout if fanout is not None else (None, None)
    sources = _safe("sources", lambda: _extract_sources(response)) or []
    finish_reason = _safe("finish_reason", lambda: _extract_finish_reason(response))
    api_cost_or_tokens = _safe("api_cost_or_tokens", lambda: _extract_usage(response))
    raw_response_envelope = _safe("raw_response_envelope", lambda: _serialize_envelope(response))

    return AdapterResult(
        model_returned=model_returned,
        raw_response_text=raw_response_text,
        search_fired=search_fired,
        fan_out_queries=fan_out_queries,
        latency_ms=latency_ms,
        finish_reason=finish_reason,
        error="; ".join(parse_errors) or None,
        api_cost_or_tokens=api_cost_or_tokens,
        raw_response_envelope=raw_response_envelope,
        sources=sources,
        request_payload=request_payload,
    )


def _extract_output_text(response: Any) -> str | None:
    text = getattr(response, "output_text", None)
    if text:
        return text
    chunks: list[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for block in getattr(item, "content", None) or []:
            if getattr(block, "type", None) != "output_text":
                continue
            block_text = getattr(block, "text", None)
            if block_text:
                chunks.append(block_text)
    return "\n".join(chunks) if chunks else None


def _extract_search_fanout(response: Any) -> tuple[bool, str | None]:
    fired = False
    queries: list[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "web_search_call":
            continue
        fired = True
        action = getattr(item, "action", None)
        if action is None:
            continue
        # The vendor action shape carries either a list (`queries`) or a
        # single string (`query`) depending on version -- capture whichever
        # is present, as given (section 6: no cross-vendor normalization).
        action_queries = getattr(action, "queries", None)
        if action_queries:
            queries.extend(action_queries)
        else:
            single_query = getattr(action, "query", None)
            if single_query:
                queries.append(single_query)
    # Absence of web_search_call items is itself a real "False" (search did
    # not fire) -- not the same as the None returned when parsing itself
    # blows up (see AdapterResult.search_fired docstring).
    return fired, (json.dumps(queries) if queries else None)


def _extract_sources(response: Any) -> list[Source]:
    sources: list[Source] = []
    position = 0
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for block in getattr(item, "content", None) or []:
            for annotation in getattr(block, "annotations", None) or []:
                if getattr(annotation, "type", None) != "url_citation":
                    continue
                url = getattr(annotation, "url", None)
                if not url:
                    continue
                position += 1
                sources.append(
                    Source(
                        position=position,
                        url_raw=url,
                        domain=_domain_from_url(url),
                        title=getattr(annotation, "title", None),
                    )
                )
    return sources


def _extract_finish_reason(response: Any) -> str | None:
    incomplete = getattr(response, "incomplete_details", None)
    reason = getattr(incomplete, "reason", None) if incomplete is not None else None
    return reason or getattr(response, "status", None)


def _extract_usage(response: Any) -> str | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return json.dumps(usage.model_dump())
    if isinstance(usage, dict):
        return json.dumps(usage)
    return json.dumps(vars(usage), default=str)


def _serialize_envelope(response: Any) -> str | None:
    """The complete unparsed response, because every other field above is a
    parse of it (section 8.1) -- if a parse is wrong, this stays fixable."""
    if hasattr(response, "model_dump_json"):
        return response.model_dump_json()
    if hasattr(response, "model_dump"):
        return json.dumps(response.model_dump(), default=str)
    if isinstance(response, dict):
        return json.dumps(response, default=str)
    return json.dumps(vars(response), default=str)


def _domain_from_url(url: str) -> str | None:
    try:
        return urlparse(url).netloc or None
    except Exception:
        return None
