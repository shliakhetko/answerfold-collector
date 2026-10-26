"""A4-prime (a3b): search-backed entity verification, between A3 and A4.

A3 resolves surface forms into entities and guesses a homepage only from
the domains cited *alongside* them -- a brand mentioned with no link gets
no URL, and neither A3 nor A4 has an "is this a real, discrete vendor?"
gate. On the trial run that left real vendors (Sembly, Chorus, Bluedot)
stuck at meets_frame=0 purely because A4 had no homepage to fetch, while
platform / feature / publisher noise ("Google Workspace", "G2", "Built-in
Zoom notes") flowed straight into A4.

This stage asks a search-capable model, once per entity, four things:
  - is_real: does this product / company verifiably exist?
  - entity_type: vendor / platform / feature / publisher / directory /
    generic / unknown
  - homepage_url: found via search when A3 had none
  - disambiguation_note: when the surface form is ambiguous

A4 then runs its frozen-frame check only on `is_real and entity_type ==
"vendor"`, using the discovered homepage. Everything else is recorded with
meets_frame = False and no LLM frame call -- still "a count", never a
denominator (the same rule A4 already follows). The full verdict lands in
the `entity_verification` table.

Off with --skip-entity-verification (A4 then behaves exactly as before).
This stage is not in the frozen coding spec -- see the ambiguity ledger.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from analysis.coding import storage
from analysis.coding.cache import LLMCache, cache_key
from collector.config import AnthropicConfig

PROMPT_VERSION = "v0-unfrozen"
VERIFICATION_MODEL = "claude-sonnet-5"  # search-capable premium tier (pinned like every other stage)

# Local copy of the collector's tool id -- kept here so this module does not
# import a collector *engine* just for a constant. Verification is a bounded
# lookup, not open research, so the search count is capped.
WEB_SEARCH_TOOL_TYPE = "web_search_20250305"
MAX_WEB_SEARCHES = 3  # enough to confirm existence / type / homepage; each extra search compounds context tokens
MAX_TOKENS = 1024
DEFAULT_BATCH_MAX_WAIT_SECONDS = 3600.0
DEFAULT_BATCH_POLL_INTERVAL_SECONDS = 30.0
_IN_PROGRESS_STATUS = "in_progress"

ENTITY_TYPES = ("vendor", "platform", "feature", "publisher", "directory", "generic", "unknown")
VENDOR_TYPE = "vendor"

MAX_EVIDENCE_SPANS = 5  # per entity, into the prompt -- enough to show how it was described
MAX_EVIDENCE_SPAN_CHARS = 400


@dataclass(frozen=True)
class VerificationRow:
    is_real: bool
    entity_type: str
    verified_homepage_url: str | None
    disambiguation_note: str | None
    confidence: float


@dataclass
class VerificationResult:
    verification: VerificationRow | None = None
    model_returned: str | None = None
    error: str | None = None
    tokens_input: int | None = None
    tokens_output: int | None = None


class MessagesClient(Protocol):
    """Same structural type as collector/engines/anthropic_claude.py's."""
    messages: Any


def _build_client(config: AnthropicConfig) -> Any:
    import anthropic  # local import: a fake `client` injected by tests never
                       # needs the real `anthropic` package importable.
    return anthropic.Anthropic(
        api_key=config.api_key, timeout=config.timeout_seconds, max_retries=config.max_retries
    )


_PROMPT_TEMPLATE = """You are verifying ONE entity that AI search engines named in answers about AI meeting / call assistants. Use web search to confirm what it actually is -- do not rely only on the snippets below.

ENTITY (best-known name): {canonical_name}
NAME VARIANTS SEEN: {surface_forms}
DOMAINS CITED ALONGSIDE IT: {candidate_domains}
HOW ANSWERS DESCRIBED IT:
{evidence_spans}

Return one JSON object with exactly these keys and nothing else -- no markdown code fence, no commentary:
- "is_real": true/false -- does a product or company by this name verifiably exist (per your searches)? false for a garbled name, a hallucination, or a phrase that is not a product.
- "entity_type": one of "vendor", "platform", "feature", "publisher", "directory", "generic", "unknown".
    - "vendor": a distinct company selling a product a buyer could adopt.
    - "platform": a broad OS / suite / cloud (e.g. Microsoft 365, Google Workspace).
    - "feature": a capability built into a larger product, not sold on its own (e.g. "Zoom AI Companion", "built-in meeting notes").
    - "publisher": a site that reviews or writes about products (e.g. a news site, a blog).
    - "directory": a software listing / marketplace (e.g. G2, Capterra).
    - "generic": a category term, not a specific product.
    - "unknown": cannot tell.
- "homepage_url": the entity's official homepage (full URL). null if it has none or you cannot find it.
- "disambiguation_note": one sentence if the name is ambiguous (multiple real products share it) or you resolved it to something non-obvious; else null.
- "confidence": your confidence in this verdict, 0.0 to 1.0."""


def _format_spans(evidence_spans: list[str]) -> str:
    if not evidence_spans:
        return "(none captured)"
    lines = []
    for span in evidence_spans[:MAX_EVIDENCE_SPANS]:
        clipped = span.strip()[:MAX_EVIDENCE_SPAN_CHARS]
        if clipped:
            lines.append(f"- {clipped}")
    return "\n".join(lines) or "(none captured)"


def _verification_kwargs(
    *, canonical_name: str, surface_forms: list[str], evidence_spans: list[str], candidate_domains: list[str]
) -> dict[str, Any]:
    """The messages.create params for one entity -- shared by the sync call
    and the Batch API request body so they stay identical."""
    return {
        "model": VERIFICATION_MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": _PROMPT_TEMPLATE.format(
                    canonical_name=canonical_name,
                    surface_forms=", ".join(surface_forms) or canonical_name,
                    candidate_domains=", ".join(candidate_domains) or "(none)",
                    evidence_spans=_format_spans(evidence_spans),
                ),
            }
        ],
        "tools": [{"type": WEB_SEARCH_TOOL_TYPE, "name": "web_search", "max_uses": MAX_WEB_SEARCHES}],
    }


def call_entity_verification(
    *,
    canonical_name: str,
    surface_forms: list[str],
    evidence_spans: list[str],
    candidate_domains: list[str],
    config: AnthropicConfig | None = None,
    client: MessagesClient | None = None,
) -> VerificationResult:
    """One search-backed call per entity. Never raises -- mirrors the
    collector adapters' contract."""
    cfg = config or AnthropicConfig.from_env()
    active_client = client or _build_client(cfg)

    kwargs = _verification_kwargs(
        canonical_name=canonical_name, surface_forms=surface_forms,
        evidence_spans=evidence_spans, candidate_domains=candidate_domains,
    )
    try:
        response = active_client.messages.create(**kwargs)
    except Exception as exc:  # contract: never raise past this point
        return VerificationResult(error=f"{type(exc).__name__}: {exc}")

    return _parse_response(response)


def _parse_response(response: Any) -> VerificationResult:
    parse_errors: list[str] = []

    def _safe(label: str, fn):
        try:
            return fn()
        except Exception as exc:
            parse_errors.append(f"{label}: {type(exc).__name__}: {exc}")
            return None

    model_returned = _safe("model_returned", lambda: getattr(response, "model", None))
    text = _safe("text", lambda: _extract_text(response))
    verification = _safe("verification", lambda: _parse_verification_json(text)) if text else None
    usage = _safe("usage", lambda: getattr(response, "usage", None))
    tokens_input = getattr(usage, "input_tokens", None) if usage is not None else None
    tokens_output = getattr(usage, "output_tokens", None) if usage is not None else None

    return VerificationResult(
        verification=verification,
        model_returned=model_returned,
        error="; ".join(parse_errors) or None,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
    )


def _extract_text(response: Any) -> str | None:
    chunks: list[str] = []
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) != "text":
            continue
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)
    return "\n".join(chunks) if chunks else None


def _parse_verification_json(text: str) -> VerificationRow:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    # The model may narrate before the JSON when it searched -- take the last
    # {...} block.
    start = stripped.rfind("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        stripped = stripped[start : end + 1]
    data = json.loads(stripped)

    entity_type = str(data["entity_type"]).strip().lower()
    if entity_type not in ENTITY_TYPES:
        entity_type = "unknown"
    return VerificationRow(
        is_real=bool(data["is_real"]),
        entity_type=entity_type,
        verified_homepage_url=(data.get("homepage_url") or None),
        disambiguation_note=(data.get("disambiguation_note") or None),
        confidence=float(data.get("confidence", 0.0)),
    )


# --------------------------------------------------------------------------
# per-entity context assembly (mirrors a3.build_surface_form_candidates)
# --------------------------------------------------------------------------

@dataclass
class VerificationContext:
    surface_forms: list[str] = field(default_factory=list)
    evidence_spans: list[str] = field(default_factory=list)
    candidate_domains: list[str] = field(default_factory=list)


def build_verification_context(
    entities_rows: list[dict[str, Any]],
    mentions_raw_rows: list[dict[str, Any]],
    sources_by_run: dict[str, list[dict[str, Any]]],
) -> dict[str, VerificationContext]:
    """entity_id -> the surface forms, evidence spans and cited domains for
    that entity, gathered from mentions_raw + the (already Task-5-resolved)
    sources. Pure."""
    sf_to_entity: dict[str, str] = {}
    context: dict[str, VerificationContext] = {}
    for entity in entities_rows:
        entity_id = entity["entity_id"]
        forms = [sf for sf in (entity.get("surface_forms") or "").split(";") if sf]
        context[entity_id] = VerificationContext(surface_forms=forms)
        for sf in forms:
            sf_to_entity[sf] = entity_id

    seen_span: dict[str, set[str]] = {eid: set() for eid in context}
    seen_domain: dict[str, set[str]] = {eid: set() for eid in context}
    for mention in mentions_raw_rows:
        entity_id = sf_to_entity.get(mention["surface_form"])
        if entity_id is None:
            continue
        ctx = context[entity_id]

        span = (mention.get("evidence_span") or "").strip()
        if span and span not in seen_span[entity_id] and len(ctx.evidence_spans) < MAX_EVIDENCE_SPANS:
            seen_span[entity_id].add(span)
            ctx.evidence_spans.append(span)

        for source in sources_by_run.get(mention["run_id"], []):
            domain = source.get("domain")
            if domain and domain not in seen_domain[entity_id]:
                seen_domain[entity_id].add(domain)
                ctx.candidate_domains.append(domain)

    return context


# --------------------------------------------------------------------------
# Batch API path (Anthropic Message Batches -- 50% off tokens; the web-search
# per-search fee is NOT discounted). Same request body as the sync call.
# --------------------------------------------------------------------------

def _verification_cache_key(canonical_name: str, surface_forms: list[str], candidate_domains: list[str]) -> str:
    cache_input = json.dumps(
        {
            "canonical_name": canonical_name,
            "surface_forms": sorted(surface_forms),
            "candidate_domains": sorted(candidate_domains),
        },
        sort_keys=True,
    )
    return cache_key(input_text=cache_input, model=VERIFICATION_MODEL, prompt_version=PROMPT_VERSION)


@dataclass(frozen=True)
class VerificationBatchItem:
    custom_id: str  # entity_id
    canonical_name: str
    surface_forms: list[str]
    evidence_spans: list[str]
    candidate_domains: list[str]


def submit_verification_batch(
    items: list[VerificationBatchItem], *, config: AnthropicConfig | None = None, client: Any | None = None
) -> str:
    cfg = config or AnthropicConfig.from_env()
    active_client = client or _build_client(cfg)
    requests = [
        {
            "custom_id": item.custom_id,
            "params": _verification_kwargs(
                canonical_name=item.canonical_name, surface_forms=item.surface_forms,
                evidence_spans=item.evidence_spans, candidate_domains=item.candidate_domains,
            ),
        }
        for item in items
    ]
    batch = active_client.messages.batches.create(requests=requests)
    return batch.id


def poll_verification_batch(
    batch_id: str,
    *,
    config: AnthropicConfig | None = None,
    client: Any | None = None,
    poll_interval_seconds: float = DEFAULT_BATCH_POLL_INTERVAL_SECONDS,
    max_wait_seconds: float = DEFAULT_BATCH_MAX_WAIT_SECONDS,
) -> str:
    """Blocks until the batch leaves in_progress, or raises TimeoutError
    (the caller keeps the batch id and re-polls later -- no work lost)."""
    cfg = config or AnthropicConfig.from_env()
    active_client = client or _build_client(cfg)
    elapsed = 0.0
    while True:
        batch = active_client.messages.batches.retrieve(batch_id)
        if batch.processing_status != _IN_PROGRESS_STATUS:
            return batch.processing_status
        if elapsed >= max_wait_seconds:
            raise TimeoutError(f"verification batch {batch_id} still {batch.processing_status} after {max_wait_seconds}s")
        time.sleep(poll_interval_seconds)
        elapsed += poll_interval_seconds


def fetch_verification_batch_results(
    batch_id: str, *, config: AnthropicConfig | None = None, client: Any | None = None
) -> dict[str, VerificationResult]:
    cfg = config or AnthropicConfig.from_env()
    active_client = client or _build_client(cfg)
    out: dict[str, VerificationResult] = {}
    for entry in active_client.messages.batches.results(batch_id):
        if getattr(entry.result, "type", None) == "succeeded":
            out[entry.custom_id] = _parse_response(entry.result.message)
        else:
            out[entry.custom_id] = VerificationResult(error=f"batch_result_{getattr(entry.result, 'type', 'unknown')}")
    return out


VERIFICATION_BATCH_FNS = (submit_verification_batch, poll_verification_batch, fetch_verification_batch_results)


# --------------------------------------------------------------------------
# stage entry point
# --------------------------------------------------------------------------

def is_verified_vendor(verification: dict[str, Any] | None) -> bool:
    """True only when this stage positively confirmed a real, discrete
    vendor. Missing verification (None / never ran) is NOT a rejection --
    A4 then proceeds as it did before this stage existed (fail open)."""
    if verification is None:
        return True
    return bool(verification.get("is_real")) and verification.get("entity_type") == VENDOR_TYPE


def _entity_verification_plan(rows, context_by_entity):
    """[(entity_id, canonical_name, surface_forms, evidence_spans,
    candidate_domains, cache_key)] -- one per entity, in table order."""
    plan = []
    for row in rows:
        entity_id = row["entity_id"]
        ctx = context_by_entity.get(entity_id) or VerificationContext(surface_forms=[row["canonical_name"]])
        surface_forms = ctx.surface_forms or [row["canonical_name"]]
        plan.append(
            (
                entity_id, row["canonical_name"], surface_forms, ctx.evidence_spans, ctx.candidate_domains,
                _verification_cache_key(row["canonical_name"], surface_forms, ctx.candidate_domains),
            )
        )
    return plan


def _prefill_verification_cache_via_batch(plan, cache, batch_fns, *, max_wait_seconds):
    """Submit ONE batch for every cache-miss entity, poll, write results
    into the cache. The batch id is parked in
    <cache_dir>/_pending_verification_batch.json so a poll TimeoutError can
    be resumed by re-calling verify_entities (no re-submit / re-charge)."""
    submit_fn, poll_fn, fetch_fn = batch_fns
    state_path = cache.cache_dir / "_pending_verification_batch.json"

    pending = [
        VerificationBatchItem(eid, name, sf, spans, doms)
        for (eid, name, sf, spans, doms, key) in plan
        if cache.get(key) is None
    ]

    if state_path.exists():
        batch_id = json.loads(state_path.read_text(encoding="utf-8"))["batch_id"]
    elif pending:
        cache.cache_dir.mkdir(parents=True, exist_ok=True)
        batch_id = submit_fn(pending)
        state_path.write_text(json.dumps({"batch_id": batch_id, "n": len(pending)}), encoding="utf-8")
    else:
        return  # nothing to do

    poll_fn(batch_id, max_wait_seconds=max_wait_seconds)  # TimeoutError propagates; state_path stays
    fetched = fetch_fn(batch_id)
    key_by_id = {eid: key for (eid, _n, _sf, _sp, _do, key) in plan}
    for eid, key in key_by_id.items():
        result = fetched.get(eid)
        if result is not None:
            cache.set(key, _result_to_cache_dict(result))
    state_path.unlink(missing_ok=True)


def verify_entities(
    *,
    context_by_entity: dict[str, VerificationContext],
    cache: LLMCache,
    call_fn: Callable[..., VerificationResult] = call_entity_verification,
    use_batch: bool = False,
    batch_fns=VERIFICATION_BATCH_FNS,
    batch_max_wait_seconds: float = DEFAULT_BATCH_MAX_WAIT_SECONDS,
    conn: sqlite3.Connection,
) -> list[VerificationResult]:
    rows = storage.list_entities(conn=conn)
    plan = _entity_verification_plan(rows, context_by_entity)

    if use_batch:
        _prefill_verification_cache_via_batch(plan, cache, batch_fns, max_wait_seconds=batch_max_wait_seconds)

    results: list[VerificationResult] = []
    output_rows: list[dict[str, Any]] = []
    for entity_id, canonical_name, surface_forms, evidence_spans, candidate_domains, key in plan:
        cached = cache.get(key)
        if cached is not None:
            result = _result_from_cache_dict(cached)
        else:
            result = call_fn(
                canonical_name=canonical_name,
                surface_forms=surface_forms,
                evidence_spans=evidence_spans,
                candidate_domains=candidate_domains,
            )
            cache.set(key, _result_to_cache_dict(result))
        results.append(result)

        v = result.verification
        output_rows.append(
            {
                "entity_id": entity_id,
                "is_real": None if v is None else v.is_real,
                "entity_type": None if v is None else v.entity_type,
                "verified_homepage_url": None if v is None else v.verified_homepage_url,
                "disambiguation_note": None if v is None else v.disambiguation_note,
                "confidence": None if v is None else v.confidence,
            }
        )

    storage.clear_entity_verification(conn=conn)
    if output_rows:
        storage.insert_entity_verification(output_rows, conn=conn)

    return results


def _result_to_cache_dict(result: VerificationResult) -> dict[str, Any]:
    v = result.verification
    return {
        "verification": (
            None
            if v is None
            else {
                "is_real": v.is_real,
                "entity_type": v.entity_type,
                "verified_homepage_url": v.verified_homepage_url,
                "disambiguation_note": v.disambiguation_note,
                "confidence": v.confidence,
            }
        ),
        "model_returned": result.model_returned,
        "error": result.error,
        "tokens_input": result.tokens_input,
        "tokens_output": result.tokens_output,
    }


def _result_from_cache_dict(data: dict[str, Any]) -> VerificationResult:
    v = data.get("verification")
    return VerificationResult(
        verification=VerificationRow(**v) if v else None,
        model_returned=data.get("model_returned"),
        error=data.get("error"),
        tokens_input=data.get("tokens_input"),
        tokens_output=data.get("tokens_output"),
    )
