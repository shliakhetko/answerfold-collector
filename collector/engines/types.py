"""Shared types for every engine adapter (INSTRUMENT.md section 7: 'one
boundary, identical contract'). An adapter is storage-ignorant: it knows
nothing about run_id, instrument_commit, or collector.storage.api -- it only
produces this shape. collector/runner/ is what wires it into storage.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    """One cited source, order-preserving (INSTRUMENT.md section 8.2).
    Mirrors collector.storage.api.insert_sources' expected item shape."""
    position: int
    url_raw: str
    domain: str | None = None
    title: str | None = None
    extra: dict | None = None


@dataclass
class AdapterResult:
    """Everything an adapter can supply for one runs-table row (section 8.1),
    minus the fields the runner derives (run_id, instrument_commit,
    timestamp, engine, intent/paraphrase/rep/day, prompt_text -- those are
    orchestration, not adapter output).

    `error` is populated, never an exception, on any failure (section 7.1).
    All other fields keep their success-path meaning only when `error` is
    None; on failure they default to None/empty, which is itself the "we do
    not know" signal for a call that never resolved.
    """
    model_returned: str | None = None
    raw_response_text: str | None = None
    # Tri-state on purpose: True/False = determined, None = adapter could not
    # tell (e.g. total parse failure). Never default to False as a stand-in
    # for "unknown" (section 7.3 / schema.sql's search_fired comment).
    search_fired: bool | None = None
    # JSON-encoded string of whatever shape the vendor returned -- "as
    # returned, unreshaped" (section 6) means no cross-vendor normalization,
    # not that it can't be serialized to fit a TEXT column.
    fan_out_queries: str | None = None
    latency_ms: int | None = None
    finish_reason: str | None = None
    error: str | None = None
    api_cost_or_tokens: str | None = None
    raw_response_envelope: str | None = None
    sources: list[Source] = field(default_factory=list)
    # Populated only by adapters for "surfaces without an API" (INSTRUMENT.md
    # section 6) -- recorded in place of model_requested/model_returned, which
    # such a surface has none of. None for every API-backed adapter.
    scraper_vendor: str | None = None
    scraper_endpoint: str | None = None
    # The exact payload the adapter sent (or attempted to send) to the vendor
    # -- populated on both the success and the transport-failure path, since
    # a failed run still recorded what was attempted. Every adapter builds
    # this from the same dict it passes to the vendor client, so it can never
    # drift from what was actually sent.
    request_payload: dict | None = None
    extra: dict | None = None
