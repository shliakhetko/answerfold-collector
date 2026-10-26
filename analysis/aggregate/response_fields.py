"""Per-engine raw_response_text/model_returned/sources extraction from a
stored run's raw response.

Analysis-layer code (INSTRUMENT.md section 12: analysis/ may import from
collector/, never the reverse). Operates on the plain dict shape a `response`
field has once round-tripped through JSON in data/01_raw/runs/*.jsonl -- NOT the SDK
response objects collector/engines/*.py's adapters see at call time. That is
why this re-implements the extraction with dict lookups instead of importing
the adapters' own `_extract_output_text`/`_extract_sources` helpers, which use
`getattr(sdk_obj, ...)` and would silently return None against a dict --
exactly the pattern analysis.aggregate.usage already established for token
usage and search_fired.

Two vendors have a trap here: `openai`'s `Response.output_text` and
`google-genai`'s `GenerateContentResponse.text` are computed `@property`s on
the live SDK object, absent from `.model_dump()` -- so they are never present
as a top-level key in a stored `response` dict. Reading `response.get("text")`
or `response.get("output_text")` here would silently return None on every
row. Both extractors below only ever walk the underlying `output`/`candidates`
structure -- the same fallback branch each adapter's own `_extract_output_text`
already falls back to -- and never take that shortcut.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import urlparse


def _domain_from_url(url: str) -> str | None:
    try:
        return urlparse(url).netloc or None
    except Exception:
        return None


# --------------------------------------------------------------------------
# raw_response_text
# --------------------------------------------------------------------------

def _text_anthropic(response: dict[str, Any]) -> str | None:
    chunks = [
        block.get("text")
        for block in response.get("content") or []
        if block.get("type") == "text" and block.get("text")
    ]
    return "\n".join(chunks) if chunks else None


def _text_openai(response: dict[str, Any]) -> str | None:
    # Deliberately never reads response.get("output_text") -- see module
    # docstring. Walks `output` exactly like the adapter's own fallback branch.
    chunks: list[str] = []
    for item in response.get("output") or []:
        if item.get("type") != "message":
            continue
        for block in item.get("content") or []:
            if block.get("type") == "output_text" and block.get("text"):
                chunks.append(block["text"])
    return "\n".join(chunks) if chunks else None


def _text_gemini(response: dict[str, Any]) -> str | None:
    # Deliberately never reads response.get("text") -- see module docstring.
    candidates = response.get("candidates") or []
    if not candidates:
        return None
    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    chunks = [part.get("text") for part in parts if part.get("text")]
    return "\n".join(chunks) if chunks else None


def _text_google_ai_mode(response: dict[str, Any]) -> str | None:
    markdown = response.get("reconstructed_markdown")
    if markdown:
        return markdown
    chunks = []
    for block in response.get("text_blocks") or []:
        text = block.get("snippet") or block.get("text")
        if text:
            chunks.append(text)
    return "\n".join(chunks) if chunks else None


RAW_TEXT_EXTRACTORS = {
    "anthropic_claude": _text_anthropic,
    "openai_chatgpt": _text_openai,
    "google_gemini": _text_gemini,
    "google_ai_mode": _text_google_ai_mode,
}


def extract_raw_response_text(engine: str, response: dict[str, Any] | None) -> str | None:
    """Dispatch to the engine-specific text extractor. None for an unknown
    engine or a missing response -- never guessed, same tri-state discipline
    as analysis.aggregate.usage.extract_search_fired."""
    extractor = RAW_TEXT_EXTRACTORS.get(engine)
    if extractor is None or response is None:
        return None
    return extractor(response)


# --------------------------------------------------------------------------
# model_returned
# --------------------------------------------------------------------------

def _model_returned_anthropic(response: dict[str, Any]) -> str | None:
    return response.get("model")


def _model_returned_openai(response: dict[str, Any]) -> str | None:
    return response.get("model")


def _model_returned_gemini(response: dict[str, Any]) -> str | None:
    return response.get("model_version")


def _model_returned_google_ai_mode(response: dict[str, Any]) -> str | None:
    # No model concept for this cell (collector/engines/google_ai_mode.py) --
    # never fabricated.
    return None


MODEL_RETURNED_EXTRACTORS = {
    "anthropic_claude": _model_returned_anthropic,
    "openai_chatgpt": _model_returned_openai,
    "google_gemini": _model_returned_gemini,
    "google_ai_mode": _model_returned_google_ai_mode,
}


def extract_model_returned(engine: str, response: dict[str, Any] | None) -> str | None:
    """Dispatch to the engine-specific model_returned extractor. None for an
    unknown engine or a missing response."""
    extractor = MODEL_RETURNED_EXTRACTORS.get(engine)
    if extractor is None or response is None:
        return None
    return extractor(response)


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------
# Each returns [{"position": int, "url_raw": str, "domain": str | None,
# "title": str | None}, ...], one row per collector.storage.api.insert_sources
# item -- 1-indexed position, order of appearance, matching each adapter's own
# _extract_sources.

def _sources_anthropic(response: dict[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    position = 0
    for block in response.get("content") or []:
        if block.get("type") != "text":
            continue
        for citation in block.get("citations") or []:
            if citation.get("type") != "web_search_result_location":
                continue
            url = citation.get("url")
            if not url:
                continue
            position += 1
            sources.append(
                {
                    "position": position,
                    "url_raw": url,
                    "domain": _domain_from_url(url),
                    "title": citation.get("title"),
                }
            )
    return sources


def _sources_openai(response: dict[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    position = 0
    for item in response.get("output") or []:
        if item.get("type") != "message":
            continue
        for block in item.get("content") or []:
            for annotation in block.get("annotations") or []:
                if annotation.get("type") != "url_citation":
                    continue
                url = annotation.get("url")
                if not url:
                    continue
                position += 1
                sources.append(
                    {
                        "position": position,
                        "url_raw": url,
                        "domain": _domain_from_url(url),
                        "title": annotation.get("title"),
                    }
                )
    return sources


def _sources_gemini(response: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = response.get("candidates") or []
    grounding_metadata = candidates[0].get("grounding_metadata") if candidates else None
    chunks = (grounding_metadata or {}).get("grounding_chunks") or []
    sources: list[dict[str, Any]] = []
    position = 0
    for chunk in chunks:
        web = chunk.get("web")
        if not web:
            continue
        url = web.get("uri")
        if not url:
            continue
        position += 1
        sources.append(
            {
                "position": position,
                "url_raw": url,
                "domain": web.get("domain") or _domain_from_url(url),
                "title": web.get("title"),
            }
        )
    return sources


def _sources_google_ai_mode(response: dict[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for position, item in enumerate(response.get("references") or [], start=1):
        url = item.get("link")
        if not url:
            continue
        sources.append(
            {
                "position": position,
                "url_raw": url,
                "domain": _domain_from_url(url),
                "title": item.get("title"),
            }
        )
    return sources


SOURCES_EXTRACTORS = {
    "anthropic_claude": _sources_anthropic,
    "openai_chatgpt": _sources_openai,
    "google_gemini": _sources_gemini,
    "google_ai_mode": _sources_google_ai_mode,
}


def extract_sources_rows(engine: str, response: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Dispatch to the engine-specific sources extractor. Empty list for an
    unknown engine or a missing response -- matches the adapters' own
    `sources` field, which is always a list, never None."""
    extractor = SOURCES_EXTRACTORS.get(engine)
    if extractor is None or response is None:
        return []
    return extractor(response)
