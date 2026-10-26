"""Anthropic ('Claude') Messages-API adapter, using the `web_search` server tool.

Follows the engine adapter contract, INSTRUMENT.md section 7:
  1. Never raises past this module's boundary -- every failure becomes a
     populated `error` field on an otherwise-complete AdapterResult.
  2. Never interprets -- `raw_response_text` is passed through unmodified.
  3. Always reports whether search fired (tri-state; see types.AdapterResult).

Storage-ignorant (section 12): this module never imports collector.storage.
Orchestration (run_id, instrument_commit, DB write) lives in
collector/runner/execute.py.

Tool type and output shapes are pinned against the installed `anthropic`
SDK, not a live response -- re-verify before a real run (section 6;
README's vendor behaviour log). Extraction below is defensive (getattr
with defaults) because this shape may drift.
"""
from __future__ import annotations

import json
import time
from typing import Any, Protocol
from urllib.parse import urlparse

from collector.config import AnthropicConfig
from collector.engines.types import AdapterResult, Source

# Pinned to the basic (non-dynamic-filtering) tool version deliberately: a
# plain search-and-cite pattern, matching the ChatGPT and Gemini adapters,
# without code-execution provisioning. Newer versions exist -- re-verify
# per module docstring.
WEB_SEARCH_TOOL_TYPE = "web_search_20250305"

# Pinned buyer geography for this study (INSTRUMENT.md section 6): every
# engine is anchored to the same US-based buyer so the methodology can state
# one rule instead of one per vendor. Anthropic's web_search tool exposes
# country/city/region/timezone on user_location but no separate language
# field -- country is the only lever here.
WEB_SEARCH_USER_LOCATION_COUNTRY = "US"


class MessagesClient(Protocol):
    """Structural type for the anthropic client's public surface used here --
    lets tests inject a fake object with a `.messages.create()` method, no
    network and no real package import required for the fake path."""
    messages: Any


def _build_client(config: AnthropicConfig) -> Any:
    import anthropic  # local import: a fake `client` injected by tests never
                       # needs the real `anthropic` package importable.
    return anthropic.Anthropic(
        api_key=config.api_key,
        timeout=config.timeout_seconds,
        max_retries=config.max_retries,
    )


def call_claude(
    *,
    prompt_text: str,
    model_requested: str,
    config: AnthropicConfig | None = None,
    client: MessagesClient | None = None,
) -> AdapterResult:
    """Send one prompt to the Anthropic Messages API. Never raises.

    `model_requested` is the pinned identifier for this experiment cell
    (EXPERIMENT.md / INSTRUMENT.md section 6) -- this function never chooses
    or defaults a model itself.

    Setup errors (missing API key -> ConfigError; client construction
    failure) intentionally propagate as exceptions -- they mean collection
    cannot start, which is not the same failure mode as a request that was
    attempted and failed (section 7.1 covers the latter only).
    """
    cfg = config or AnthropicConfig.from_env()
    active_client = client or _build_client(cfg)

    web_search_tool: dict[str, Any] = {
        "type": WEB_SEARCH_TOOL_TYPE,
        "name": "web_search",
        "user_location": {
            "type": "approximate",
            "country": WEB_SEARCH_USER_LOCATION_COUNTRY,
        },
    }
    if cfg.web_search_max_uses is not None:
        # Omitted, not defaulted, when unset: Anthropic's own guidance is
        # that comparative/multi-entity questions -- this study's subject --
        # can need 10+ searches, so an invented cap here would silently
        # truncate the real answer rather than reflect what a buyer sees
        # through the consumer surface, which applies no such cap either
        # (INSTRUMENT.md section 6).
        web_search_tool["max_uses"] = cfg.web_search_max_uses

    kwargs: dict[str, Any] = {
        "model": model_requested,
        "max_tokens": cfg.max_tokens,  # required by the Anthropic API, unlike
                                        # OpenAI's optional max_output_tokens
        "messages": [{"role": "user", "content": prompt_text}],
        "tools": [web_search_tool],
    }

    start = time.monotonic()
    try:
        response = active_client.messages.create(**kwargs)
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
    finish_reason = _safe("finish_reason", lambda: getattr(response, "stop_reason", None))
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
    chunks: list[str] = []
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) != "text":
            continue
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)
    return "\n".join(chunks) if chunks else None


def _extract_search_fanout(response: Any) -> tuple[bool, str | None]:
    fired = False
    queries: list[str] = []
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) != "server_tool_use":
            continue
        if getattr(block, "name", None) != "web_search":
            continue
        fired = True
        tool_input = getattr(block, "input", None) or {}
        query = tool_input.get("query") if hasattr(tool_input, "get") else None
        if query:
            queries.append(query)
    # Absence of server_tool_use/web_search blocks is itself a real "False"
    # (search did not fire) -- not the same as the None returned when
    # parsing itself blows up (see AdapterResult.search_fired docstring).
    return fired, (json.dumps(queries) if queries else None)


def _extract_sources(response: Any) -> list[Source]:
    """Sources are pulled from `citations` inside `text` blocks -- the point
    where a source was actually used in the answer -- mirroring the ChatGPT
    adapter's use of url_citation annotations, rather than the raw
    web_search_tool_result result list (which may include results Claude
    never cited)."""
    sources: list[Source] = []
    position = 0
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) != "text":
            continue
        for citation in getattr(block, "citations", None) or []:
            if getattr(citation, "type", None) != "web_search_result_location":
                continue
            url = getattr(citation, "url", None)
            if not url:
                continue
            position += 1
            sources.append(
                Source(
                    position=position,
                    url_raw=url,
                    domain=_domain_from_url(url),
                    title=getattr(citation, "title", None),
                )
            )
    return sources


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
