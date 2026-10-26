"""Google Gemini adapter, grounded with the `google_search` tool.

Follows the engine adapter contract, INSTRUMENT.md section 7:
  1. Never raises past this module's boundary -- every failure becomes a
     populated `error` field on an otherwise-complete AdapterResult.
  2. Never interprets -- `raw_response_text` is passed through unmodified.
  3. Always reports whether search fired (tri-state; see types.AdapterResult).

Storage-ignorant (section 12): this module never imports collector.storage.
Orchestration (run_id, instrument_commit, DB write) lives in
collector/runner/execute.py.

Uses the long-established `models.generate_content` / `grounding_metadata`
shape, pinned against the installed `google-genai` SDK, not a live
response -- re-verify before a real run (section 6; README's vendor
behaviour log). Extraction below is defensive (getattr with defaults)
because this shape may drift.
"""
from __future__ import annotations

import json
import time
from typing import Any, Protocol
from urllib.parse import urlparse

from collector.config import GeminiConfig
from collector.engines.types import AdapterResult, Source

# Pinned buyer geography for this study (INSTRUMENT.md section 6): every
# engine is anchored to the same US-based buyer so the methodology can state
# one rule instead of one per vendor. The google_search grounding tool has no
# country-code knob -- the only location lever the SDK exposes is
# tool_config.retrieval_config's lat_lng + language_code (types.RetrievalConfig,
# same shape used for Google Maps grounding; re-verify against a live
# response per module docstring). lat/lng is the geographic center of the
# contiguous US (near Lebanon, Kansas) -- a representative national point,
# not a real city.
RETRIEVAL_LOCATION_LAT = 39.8283
RETRIEVAL_LOCATION_LNG = -98.5795
RETRIEVAL_LANGUAGE_CODE = "en"


class GenerativeClient(Protocol):
    """Structural type for the genai client's public surface used here --
    lets tests inject a fake object with a `.models.generate_content()`
    method, no network and no real package import required for the fake
    path."""
    models: Any


def _build_client(config: GeminiConfig) -> Any:
    from google import genai  # local import: a fake `client` injected by
                               # tests never needs the real package importable.
    from google.genai import types

    http_options = types.HttpOptions(
        timeout=int(config.timeout_seconds * 1000),  # SDK takes milliseconds
        retry_options=types.HttpRetryOptions(attempts=config.max_retries),
    )
    return genai.Client(api_key=config.api_key, http_options=http_options)


def call_gemini(
    *,
    prompt_text: str,
    model_requested: str,
    config: GeminiConfig | None = None,
    client: GenerativeClient | None = None,
) -> AdapterResult:
    """Send one prompt to the Gemini API with Google Search grounding.
    Never raises.

    `model_requested` is the pinned identifier for this experiment cell
    (EXPERIMENT.md / INSTRUMENT.md section 6) -- this function never chooses
    or defaults a model itself.

    Setup errors (missing API key -> ConfigError; client construction
    failure) intentionally propagate as exceptions -- they mean collection
    cannot start, which is not the same failure mode as a request that was
    attempted and failed (section 7.1 covers the latter only).
    """
    cfg = config or GeminiConfig.from_env()
    active_client = client or _build_client(cfg)

    from google.genai import types

    retrieval_config = types.RetrievalConfig(
        lat_lng=types.LatLng(latitude=RETRIEVAL_LOCATION_LAT, longitude=RETRIEVAL_LOCATION_LNG),
        language_code=RETRIEVAL_LANGUAGE_CODE,
    )
    generation_config: dict[str, Any] = {
        "tools": [types.Tool(google_search=types.GoogleSearch())],
        "tool_config": types.ToolConfig(retrieval_config=retrieval_config),
    }
    if cfg.max_output_tokens is not None:
        generation_config["max_output_tokens"] = cfg.max_output_tokens

    # Plain-dict mirror of `generation_config` for storage only -- the real
    # call below uses typed SDK objects (types.Tool/types.GoogleSearch/
    # types.ToolConfig), not this dict. `{"google_search": {}}` is an accurate
    # rendering of an unconfigured GoogleSearch() (verified against the
    # installed SDK: every field on GoogleSearch is optional and left unset
    # here).
    request_payload: dict[str, Any] = {
        "model": model_requested,
        "contents": prompt_text,
        "tools": [{"google_search": {}}],
        "tool_config": {
            "retrieval_config": {
                "lat_lng": {"latitude": RETRIEVAL_LOCATION_LAT, "longitude": RETRIEVAL_LOCATION_LNG},
                "language_code": RETRIEVAL_LANGUAGE_CODE,
            }
        },
    }
    if cfg.max_output_tokens is not None:
        request_payload["max_output_tokens"] = cfg.max_output_tokens

    start = time.monotonic()
    try:
        response = active_client.models.generate_content(
            model=model_requested,
            contents=prompt_text,
            config=types.GenerateContentConfig(**generation_config),
        )
    except Exception as exc:  # contract: never raise past this point
        latency_ms = int((time.monotonic() - start) * 1000)
        return AdapterResult(
            error=f"{type(exc).__name__}: {exc}",
            latency_ms=latency_ms,
            request_payload=request_payload,
        )
    latency_ms = int((time.monotonic() - start) * 1000)

    return _parse_response(response, latency_ms=latency_ms, request_payload=request_payload)


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

    model_returned = _safe("model_returned", lambda: getattr(response, "model_version", None))
    raw_response_text = _safe("raw_response_text", lambda: getattr(response, "text", None))
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


def _first_candidate(response: Any) -> Any | None:
    candidates = getattr(response, "candidates", None) or []
    return candidates[0] if candidates else None


def _extract_search_fanout(response: Any) -> tuple[bool, str | None]:
    candidate = _first_candidate(response)
    grounding_metadata = getattr(candidate, "grounding_metadata", None) if candidate is not None else None
    queries = list(getattr(grounding_metadata, "web_search_queries", None) or []) if grounding_metadata else []
    # Absence of grounding_metadata (or an empty query list) is itself a real
    # "False" (search did not fire) -- not the same as the None returned when
    # parsing itself blows up (see AdapterResult.search_fired docstring).
    fired = bool(queries)
    return fired, (json.dumps(queries) if queries else None)


def _extract_sources(response: Any) -> list[Source]:
    candidate = _first_candidate(response)
    grounding_metadata = getattr(candidate, "grounding_metadata", None) if candidate is not None else None
    chunks = getattr(grounding_metadata, "grounding_chunks", None) or [] if grounding_metadata else []
    sources: list[Source] = []
    position = 0
    for chunk in chunks:
        web = getattr(chunk, "web", None)
        if web is None:
            continue
        url = getattr(web, "uri", None)
        if not url:
            continue
        position += 1
        sources.append(
            Source(
                position=position,
                url_raw=url,
                domain=getattr(web, "domain", None) or _domain_from_url(url),
                title=getattr(web, "title", None),
            )
        )
    return sources


def _extract_finish_reason(response: Any) -> str | None:
    candidate = _first_candidate(response)
    if candidate is None:
        return None
    reason = getattr(candidate, "finish_reason", None)
    return getattr(reason, "value", reason) if reason is not None else None


def _extract_usage(response: Any) -> str | None:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return json.dumps(usage.model_dump(mode="json"))
    if isinstance(usage, dict):
        return json.dumps(usage)
    return json.dumps(vars(usage), default=str)


def _serialize_envelope(response: Any) -> str | None:
    """The complete unparsed response, because every other field above is a
    parse of it (section 8.1) -- if a parse is wrong, this stays fixable."""
    if hasattr(response, "model_dump_json"):
        return response.model_dump_json()
    if hasattr(response, "model_dump"):
        return json.dumps(response.model_dump(mode="json"), default=str)
    if isinstance(response, dict):
        return json.dumps(response, default=str)
    return json.dumps(vars(response), default=str)


def _domain_from_url(url: str) -> str | None:
    try:
        return urlparse(url).netloc or None
    except Exception:
        return None
