"""SerpApi Google organic-results wrapper (Task 3).

The study's window instrument only ever reached Google AI Mode via SerpApi
(collector/engines/google_ai_mode.py). Task 3 captures the *organic* SERP for
every prompt string -- a different SerpApi engine ("google") -- so Phase A can
derive the per-company Google-top-10 binary
(analysis/coding/b3b_serp_derivation.py already exists for that join).

Contract, matching the google_ai_mode adapter:
  - never raises past this boundary; a failed call returns a SerpResult with
    `error` set and whatever envelope came back.
  - stores the entire raw JSON response, not a parsed subset.
  - passes a COPY of the params dict to the client (serpapi mutates it in
    place to inject the key -- would otherwise leak the secret).

Per the batch's HARD RULE 5, every call pins: no_cache=true,
location="United States", gl=us, hl=en, device=desktop.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from collector.config import SerpApiConfig

SERPAPI_ENGINE = "google"
SCRAPER_VENDOR = "serpapi"
SCRAPER_ENDPOINT = "google"

# Fixed per HARD RULE 5.
SERP_LOCATION = "United States"
SERP_GL = "us"
SERP_HL = "en"
SERP_DEVICE = "desktop"
SERP_NUM = 10  # top 10 organic

# Rough per-search price for the cost check. SerpApi's Developer plan is
# $75 / 5,000 searches = $0.015; higher tiers are cheaper. Used only for the
# estimate printed before a run.
APPROX_USD_PER_SEARCH = 0.015


class SearchClient(Protocol):
    def search(self, params: dict) -> Any: ...


def _build_client(config: SerpApiConfig) -> Any:
    import serpapi

    return serpapi.Client(api_key=config.api_key, timeout=config.timeout_seconds)


@dataclass
class SerpResult:
    query: str
    ok: bool = False
    error: str | None = None
    elapsed_ms: int = 0
    request_payload: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] | None = None


def build_payload(query: str) -> dict[str, Any]:
    return {
        "engine": SERPAPI_ENGINE,
        "q": query,
        "location": SERP_LOCATION,
        "gl": SERP_GL,
        "hl": SERP_HL,
        "device": SERP_DEVICE,
        "num": SERP_NUM,
        "no_cache": True,
    }


def run_search(query: str, *, config: SerpApiConfig | None = None, client: SearchClient | None = None) -> SerpResult:
    cfg = config or SerpApiConfig.from_env()
    active_client = client or _build_client(cfg)
    payload = build_payload(query)
    result = SerpResult(query=query, request_payload=payload)

    start = time.monotonic()
    try:
        raw = active_client.search(dict(payload))  # copy: see module docstring
    except Exception as exc:  # contract: never raise
        result.error = f"{type(exc).__name__}: {exc}"
        result.elapsed_ms = int((time.monotonic() - start) * 1000)
        return result
    result.elapsed_ms = int((time.monotonic() - start) * 1000)

    try:
        result.raw = dict(raw)
    except Exception as exc:
        result.error = f"envelope_serialize_error: {type(exc).__name__}: {exc}"
        return result

    metadata_status = (result.raw.get("search_metadata") or {}).get("status")
    vendor_error = result.raw.get("error")
    if vendor_error or metadata_status == "Error":
        result.error = vendor_error or f"SerpApi search_metadata.status={metadata_status!r}"
        return result

    result.ok = True
    return result
