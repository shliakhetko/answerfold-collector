"""Google AI Mode adapter, reached via the SerpApi third-party scraper
(https://serpapi.com/google-ai-mode-api).

Google AI Mode has no first-party API -- this is a "surface without an API"
per INSTRUMENT.md section 6: the access path (SerpApi) is recorded per row
via `scraper_vendor`/`scraper_endpoint` in place of a model identifier.
`model_requested` is still accepted, to satisfy the uniform adapter call
signature collector/runner/execute.py depends on, but is never sent to
SerpApi and carries no meaning here -- there is no model to pin.

Follows the engine adapter contract, INSTRUMENT.md section 7:
  1. Never raises past this module's boundary -- every failure becomes a
     populated `error` field on an otherwise-complete AdapterResult. This
     includes a logical failure: SerpApi can return HTTP 200 with
     search_metadata.status == "Error" in the body, which must not be
     mistaken for a genuine empty answer (section 7.1).
  2. Never interprets -- `raw_response_text` is passed through unmodified.
  3. search_fired is reported as True unconditionally on a successful parse
     -- Google AI Mode is inherently a search product with no non-search
     mode, unlike the other adapters' tri-state derivation from response
     content. fan_out_queries stays None always: SerpApi does not expose
     which sub-queries AI Mode issued internally, and this adapter does not
     fabricate them (section 6: sub-queries are captured only where a
     vendor actually exposes them).

Storage-ignorant (section 12): this module never imports collector.storage.
Orchestration (run_id, instrument_commit, DB write) lives in
collector/runner/execute.py.

Shape is pinned against the installed `serpapi` SDK and SerpApi's docs, not
a live response -- re-verify before a real run (section 6; README's vendor
behaviour log), especially the `references` item shape (0- vs 1-based
`index`).
"""
from __future__ import annotations

import json
import time
from typing import Any, Protocol
from urllib.parse import urlparse

from collector.config import SerpApiConfig
from collector.engines.types import AdapterResult, Source

# Vendor implementation facts, not experiment parameters. SERPAPI_ENGINE is
# the `engine` query param SerpApi expects; SCRAPER_ENDPOINT is the same
# literal recorded on the row (section 6) -- kept as two names for
# readability at each call site, not two independently-maintained values.
SERPAPI_ENGINE = "google_ai_mode"
SCRAPER_VENDOR = "serpapi"
SCRAPER_ENDPOINT = SERPAPI_ENGINE

# Pinned buyer geography for this study (INSTRUMENT.md section 6): every
# engine is anchored to the same US-based buyer so the methodology can state
# one rule instead of one per vendor. Google's own two-letter country (`gl`)
# and language (`hl`) query params, passed through unchanged to SerpApi --
# without these, Google resolves geography from the request's origin IP,
# which is wherever the instrument happens to run.
SERPAPI_COUNTRY = "us"
SERPAPI_LANGUAGE = "en"


class SearchClient(Protocol):
    """Structural type for the serpapi client's public surface used here --
    lets tests inject a fake object with a `.search()` method, no network
    and no real package import required for the fake path."""
    def search(self, params: dict) -> Any: ...


def _build_client(config: SerpApiConfig) -> Any:
    import serpapi  # local import: a fake `client` injected by tests never
                     # needs the real `serpapi` package importable.
    return serpapi.Client(api_key=config.api_key, timeout=config.timeout_seconds)


def call_google_ai_mode(
    *,
    prompt_text: str,
    model_requested: str,
    config: SerpApiConfig | None = None,
    client: SearchClient | None = None,
) -> AdapterResult:
    """Query Google AI Mode through SerpApi for one prompt. Never raises.

    `model_requested` is accepted only to satisfy the uniform adapter call
    signature -- there is no model concept for this cell (see module
    docstring).

    Setup errors (missing API key -> ConfigError; client construction
    failure) intentionally propagate as exceptions -- they mean collection
    cannot start, which is not the same failure mode as a request that was
    attempted and failed (section 7.1 covers the latter only).
    """
    cfg = config or SerpApiConfig.from_env()
    active_client = client or _build_client(cfg)

    # no_cache=True is not optional: SerpApi serves a cached result for up to
    # 1h when engine+q match a prior call, which would silently collapse
    # same-hour repetitions to one observation and defeat the noise measurement
    # INSTRUMENT.md section 4 relies on reps for. Cached hits are free but a
    # live fetch is billed, so this trades a small per-call cost for reps that
    # are actually independent.
    request_payload = {
        "engine": SERPAPI_ENGINE,
        "q": prompt_text,
        "gl": SERPAPI_COUNTRY,
        "hl": SERPAPI_LANGUAGE,
        "no_cache": True,
    }

    start = time.monotonic()
    try:
        # Pass a COPY, never the dict we store/log: serpapi.Client.request()
        # mutates the params dict it's given in place, injecting the real
        # api_key into it before the HTTP call. Passing request_payload
        # itself would leak the live secret into request_payload (and from
        # there into the JSONL line) as a side effect of the SDK call.
        results = active_client.search(dict(request_payload))
    except Exception as exc:  # contract: never raise past this point
        latency_ms = int((time.monotonic() - start) * 1000)
        return AdapterResult(
            error=f"{type(exc).__name__}: {exc}",
            latency_ms=latency_ms,
            # Unlike model_returned, the scraper's identity is a fact about
            # which adapter ran, not something parsed from a response -- so
            # it is knowable (and recorded) even when the call itself failed.
            scraper_vendor=SCRAPER_VENDOR,
            scraper_endpoint=SCRAPER_ENDPOINT,
            request_payload=request_payload,
        )
    latency_ms = int((time.monotonic() - start) * 1000)

    return _parse_response(results, latency_ms=latency_ms, request_payload=request_payload)


def _parse_response(results: Any, *, latency_ms: int, request_payload: dict[str, Any]) -> AdapterResult:
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

    status = _safe("search_metadata.status", lambda: (results.get("search_metadata") or {}).get("status"))
    vendor_error = _safe("error", lambda: results.get("error"))
    raw_response_envelope = _safe("raw_response_envelope", lambda: _serialize_envelope(results))

    if status == "Error" or vendor_error:
        # A 200 response can still carry a logical failure in the body
        # (section 7.1: a failed call must never look like a genuine empty
        # answer). The envelope is still stored -- the body is real and
        # worth keeping for later recovery even though the result failed
        # (section 8.1's "if a parse is wrong, only the envelope permits
        # recovery" applies to a logical failure too, not only a parse
        # exception).
        return AdapterResult(
            error=vendor_error or f"SerpApi search_metadata.status={status!r}",
            latency_ms=latency_ms,
            raw_response_envelope=raw_response_envelope,
            scraper_vendor=SCRAPER_VENDOR,
            scraper_endpoint=SCRAPER_ENDPOINT,
            request_payload=request_payload,
        )

    raw_response_text = _safe("raw_response_text", lambda: _extract_output_text(results))
    sources = _safe("sources", lambda: _extract_sources(results)) or []

    return AdapterResult(
        raw_response_text=raw_response_text,
        # Unconditional: Google AI Mode is inherently a search product, so a
        # successful parse means search fired -- there is no non-search mode
        # to distinguish, unlike the API-backed adapters (see module docstring).
        search_fired=True,
        # Not fabricated: SerpApi does not expose which sub-queries AI Mode
        # issued internally (see module docstring).
        fan_out_queries=None,
        latency_ms=latency_ms,
        finish_reason=status,
        error="; ".join(parse_errors) or None,
        # Not fabricated: SerpApi bills per search call, not tokens -- there
        # is nothing vendor-reported to put here.
        api_cost_or_tokens=None,
        raw_response_envelope=raw_response_envelope,
        sources=sources,
        scraper_vendor=SCRAPER_VENDOR,
        scraper_endpoint=SCRAPER_ENDPOINT,
        request_payload=request_payload,
    )


def _extract_output_text(results: Any) -> str | None:
    markdown = results.get("reconstructed_markdown")
    if markdown:
        return markdown
    chunks: list[str] = []
    for block in results.get("text_blocks") or []:
        text = block.get("snippet") or block.get("text")
        if text:
            chunks.append(text)
    return "\n".join(chunks) if chunks else None


def _extract_sources(results: Any) -> list[Source]:
    sources: list[Source] = []
    for position, item in enumerate(results.get("references") or [], start=1):
        url = item.get("link")
        if not url:
            continue
        sources.append(
            Source(
                position=position,
                url_raw=url,
                domain=_domain_from_url(url),
                title=item.get("title"),
            )
        )
    return sources


def _serialize_envelope(results: Any) -> str | None:
    """The complete unparsed response, because every other field above is a
    parse of it (section 8.1) -- if a parse is wrong, this stays fixable."""
    return json.dumps(dict(results), default=str)


def _domain_from_url(url: str) -> str | None:
    try:
        return urlparse(url).netloc or None
    except Exception:
        return None
