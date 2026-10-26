"""Per-engine token-usage and search-fired extraction from a stored run's
raw response.

Analysis-layer code (INSTRUMENT.md section 12: analysis/ may import from
collector/, never the reverse). Operates on the plain dict shape a `response`
field has once round-tripped through JSON in data/01_raw/runs/*.jsonl -- NOT the SDK
response objects collector/engines/*.py's adapters see at call time. That is
why this re-implements the extraction with dict lookups instead of importing
the adapters' own `_extract_usage`/`_extract_search_fanout` helpers, which use
`getattr(sdk_obj, ...)` and would silently return None against a dict.

reasoning_tokens/tool_tokens are None wherever a vendor doesn't expose that
breakdown (OpenAI and Anthropic report no separate tool-token count; only
Gemini's usage_metadata carries `tool_use_prompt_token_count`) -- never
guessed, same rule as the token counts.

extract_extra_usage_fields() pulls out the remaining usage-object fields
(cache hits/writes, tool-use counts) that vary too much per vendor to force
into shared TokenUsage columns -- e.g. OpenAI's `cached_tokens` and
Anthropic's `cache_read_input_tokens` measure related but not identical
things (different TTLs, different billing). Kept as vendor-native key names,
one dict per engine, matching INSTRUMENT.md section 6: "no cross-vendor
normalization" at collection/extraction time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenUsage:
    input: int | None
    output: int | None
    total: int | None
    reasoning_tokens: int | None
    tool_tokens: int | None


def extract_openai(response: dict[str, Any] | None) -> TokenUsage:
    usage = (response or {}).get("usage")
    if not usage:
        return TokenUsage(None, None, None, None, None)
    return TokenUsage(
        input=usage.get("input_tokens"),
        output=usage.get("output_tokens"),
        total=usage.get("total_tokens"),
        reasoning_tokens=(usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
        tool_tokens=None,  # not broken out by this vendor
    )


def extract_gemini(response: dict[str, Any] | None) -> TokenUsage:
    usage = (response or {}).get("usage_metadata")
    if not usage:
        return TokenUsage(None, None, None, None, None)
    return TokenUsage(
        input=usage.get("prompt_token_count"),
        output=usage.get("candidates_token_count"),
        total=usage.get("total_token_count"),
        reasoning_tokens=usage.get("thoughts_token_count"),
        tool_tokens=usage.get("tool_use_prompt_token_count"),
    )


def extract_anthropic(response: dict[str, Any] | None) -> TokenUsage:
    usage = (response or {}).get("usage")
    if not usage:
        return TokenUsage(None, None, None, None, None)
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    # No total_tokens field from this vendor. Sum only
    # input+output: cache_creation_input_tokens/cache_read_input_tokens are
    # informational variants of input tokens, not additional ones -- adding
    # them in would double-count against how Anthropic actually bills.
    total = (
        input_tokens + output_tokens
        if input_tokens is not None and output_tokens is not None
        else None
    )
    return TokenUsage(
        input=input_tokens,
        output=output_tokens,
        total=total,
        reasoning_tokens=(usage.get("output_tokens_details") or {}).get("thinking_tokens"),
        # server_tool_use.web_search_requests is a *count of searches*, not a
        # token count -- Anthropic folds search-result content straight into
        # input_tokens with no separate breakdown (see web-search-tool docs).
        tool_tokens=None,
    )


def extract_google_ai_mode(response: dict[str, Any] | None) -> TokenUsage:
    # SerpApi's google_ai_mode response has no token field of any kind --
    # always None, never guessed.
    return TokenUsage(None, None, None, None, None)


EXTRACTORS = {
    "openai_chatgpt": extract_openai,
    "google_gemini": extract_gemini,
    "anthropic_claude": extract_anthropic,
    "google_ai_mode": extract_google_ai_mode,
}


def extract_usage(engine: str, response: dict[str, Any] | None) -> TokenUsage:
    """Dispatch to the engine-specific extractor. Unknown engines get an
    all-None TokenUsage rather than a KeyError -- a new engine's usage
    extraction just hasn't been added yet, which is not a run-loading
    failure."""
    extractor = EXTRACTORS.get(engine)
    if extractor is None:
        return TokenUsage(None, None, None, None, None)
    return extractor(response)


def _search_fired_openai(response: dict[str, Any]) -> bool:
    return any(item.get("type") == "web_search_call" for item in response.get("output") or [])


def _search_fired_gemini(response: dict[str, Any]) -> bool:
    candidates = response.get("candidates") or []
    grounding_metadata = (candidates[0] if candidates else {}).get("grounding_metadata") or {}
    return bool(grounding_metadata.get("web_search_queries"))


def _search_fired_anthropic(response: dict[str, Any]) -> bool:
    return any(
        block.get("type") == "server_tool_use" and block.get("name") == "web_search"
        for block in response.get("content") or []
    )


def _search_fired_google_ai_mode(response: dict[str, Any]) -> bool:
    # Mirrors collector/engines/google_ai_mode.py: search_fired is
    # unconditionally True on a successful parse -- Google AI Mode is
    # inherently a search product, no non-search mode exists. A record only
    # carries a `response` at all when the call succeeded (collector/storage/
    # jsonl.py stores error XOR response, never both), so reaching this
    # function already implies success.
    return True


SEARCH_FIRED_EXTRACTORS = {
    "openai_chatgpt": _search_fired_openai,
    "google_gemini": _search_fired_gemini,
    "anthropic_claude": _search_fired_anthropic,
    "google_ai_mode": _search_fired_google_ai_mode,
}


def extract_search_fired(engine: str, response: dict[str, Any] | None) -> bool | None:
    """Dispatch to the engine-specific search-fired check. None (not False)
    when the engine is unrecognized or there's no response to inspect --
    "unknown" must never collapse into "did not search" (same tri-state
    rule as AdapterResult.search_fired; see schema.sql's comment on the
    column)."""
    extractor = SEARCH_FIRED_EXTRACTORS.get(engine)
    if extractor is None or response is None:
        return None
    return extractor(response)


def _extra_openai(usage: dict[str, Any]) -> dict[str, Any]:
    details = usage.get("input_tokens_details") or {}
    return {"cached_tokens": details.get("cached_tokens"), "cache_write_tokens": details.get("cache_write_tokens")}


def _extra_gemini(usage: dict[str, Any]) -> dict[str, Any]:
    return {"cached_content_tokens": usage.get("cached_content_token_count")}


def _extra_anthropic(usage: dict[str, Any]) -> dict[str, Any]:
    server_tool_use = usage.get("server_tool_use") or {}
    return {
        "cache_creation_tokens": usage.get("cache_creation_input_tokens"),
        "cache_read_tokens": usage.get("cache_read_input_tokens"),
        "web_search_requests": server_tool_use.get("web_search_requests"),
        "web_fetch_requests": server_tool_use.get("web_fetch_requests"),
    }


# (response key holding the usage object, per-engine extractor) -- google_ai_mode
# has no usage object of any kind, so it's simply absent from this map.
_EXTRA_USAGE = {
    "openai_chatgpt": ("usage", _extra_openai),
    "google_gemini": ("usage_metadata", _extra_gemini),
    "anthropic_claude": ("usage", _extra_anthropic),
}


def extract_extra_usage_fields(engine: str, response: dict[str, Any] | None) -> dict[str, Any]:
    """Vendor-native usage sub-fields not covered by TokenUsage, keyed by
    their own vendor's field name (never renamed into a shared axis -- see
    module docstring). Missing keys are dropped rather than kept as
    explicit None, so a row with nothing extra reported serializes to `{}`
    (schema.sql's `extra` column: NULL means "nothing extra captured")."""
    entry = _EXTRA_USAGE.get(engine)
    if entry is None:
        return {}
    field_key, extractor = entry
    usage = (response or {}).get(field_key) or {}
    if not usage:
        return {}
    return {k: v for k, v in extractor(usage).items() if v is not None}
