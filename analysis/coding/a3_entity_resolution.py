"""A3: entity resolution (Post-Collection Coding Pipeline Spec). One pass
over the surface_forms view (analysis/coding/schema.sql), using each
surface_form's cited source domains (from the `sources` table -- see
analysis/aggregate/response_fields.py) to disambiguate name collisions text
alone can't resolve. Known trap: askrosie.ai and heyrosie.com are different
vendors even when the answer text says only "Rosie" for both -- domain
evidence is what tells them apart.

Output: the entities/entity_surface_forms tables (analysis/coding/storage.py)
-- A4 enriches the same entities rows with classification columns.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Protocol

from analysis.coding import storage
from analysis.coding.cache import LLMCache, cache_key
from collector.config import AnthropicConfig

PROMPT_VERSION = "v0-unfrozen"
RESOLUTION_MODEL = "claude-haiku-4-5-20251001"  # verified current cheap/fast-tier Anthropic model

# The real window produces ~2,200 surface forms with up to ~600 cited
# domains each -- one prompt is ~455k tokens, far over the 200k limit. So
# A3 runs in chunks: cap the disambiguating domain list, sort by normalized
# name so a company's spelling variants land in the same chunk, resolve each
# chunk, then merge entities that share a normalized canonical name or a
# homepage domain across chunk boundaries.
MAX_CANDIDATE_DOMAINS = 8
RESOLUTION_CHUNK_SIZE = 120
# One chunk's output is ~100 entity objects -- comfortably minutes of Haiku
# generation, well over the 60s collector default. A cache-cold chunk that
# errors (timeout / transport) is NOT cached, so the next run retries it.
RESOLUTION_TIMEOUT_SECONDS = 600.0


@dataclass(frozen=True)
class EntityRow:
    entity_id: str
    canonical_name: str
    surface_forms: tuple[str, ...]
    homepage_url: str | None


@dataclass
class EntityResolutionResult:
    entities: list[EntityRow] = field(default_factory=list)
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


_PROMPT_TEMPLATE = """You are grouping company/product name variants into distinct real-world companies ("entities").

Below is a JSON array of surface forms mentioned across AI search engine answers. Each item has:
- "surface_form": one exact spelling as it appeared in an answer.
- "mention_count": how many times it was mentioned.
- "candidate_domains": distinct domains cited alongside this surface_form, in the order they were seen (may be empty).

Group surface forms that refer to the SAME real-world company into one entity. Two surface forms with different candidate_domains almost always refer to DIFFERENT companies even if their names look similar or identical -- e.g. a bare mention of "Rosie" cited alongside askrosie.ai is a different company from a bare mention of "Rosie" cited alongside heyrosie.com, even though the text says the same word both times. Only merge same-spelling surface forms with different domains if you have strong evidence they are the same company.

For each entity, pick:
- "entity_id": a short, stable, lowercase-hyphenated slug (e.g. "goodcall", "askrosie") -- must be unique across entities.
- "canonical_name": the best human-readable name for the company.
- "surface_forms": the list of input surface_form strings that belong to this entity.
- "homepage_url": your best guess at the company's homepage, from candidate_domains if available, else null.

Return a JSON array of these objects and nothing else -- no markdown code fence, no commentary. Every input surface_form must appear in exactly one output entity's surface_forms list.

SURFACE FORMS:
{surface_forms_json}"""


def build_surface_form_candidates(
    surface_forms_rows: list[dict[str, Any]],
    sources_by_run: dict[str, list[dict[str, Any]]],
    *,
    max_domains: int = MAX_CANDIDATE_DOMAINS,
) -> list[dict[str, Any]]:
    """For each surface_forms.csv row, collect the set of distinct cited
    domains across its run_ids (order-preserving) -- the disambiguating
    evidence the prompt above needs for the askrosie.ai/heyrosie.com trap.
    The list is capped at max_domains: the first few cited domains carry
    essentially all the disambiguation signal, and an uncapped list (up to
    ~600 domains for a popular name) is what blows the context window."""
    candidates = []
    for row in surface_forms_rows:
        run_ids = row["run_ids"].split(";") if row["run_ids"] else []
        seen: set[str] = set()
        domains: list[str] = []
        for run_id in run_ids:
            for source in sources_by_run.get(run_id, []):
                domain = source.get("domain")
                if domain and domain not in seen:
                    seen.add(domain)
                    domains.append(domain)
        candidates.append(
            {
                "surface_form": row["surface_form"],
                "mention_count": int(row["mention_count"]),
                "candidate_domains": domains[:max_domains],
            }
        )
    return candidates


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _chunk_candidates(
    candidates: list[dict[str, Any]], chunk_size: int = RESOLUTION_CHUNK_SIZE
) -> list[list[dict[str, Any]]]:
    """Sort by normalized surface form (so "Rosie AI" / "RosieAI" / "rosie"
    are adjacent and land in one chunk), then slice into <=chunk_size groups."""
    ordered = sorted(candidates, key=lambda c: (_norm_name(c["surface_form"]), c["surface_form"]))
    return [ordered[i : i + chunk_size] for i in range(0, len(ordered), chunk_size)] or [[]]


def _domain_of(url: str | None) -> str:
    from urllib.parse import urlparse

    return (urlparse(url).netloc or "").lower().removeprefix("www.") if url else ""


def _pick_representative(members: list[EntityRow]) -> EntityRow:
    """Of the chunk-level entities that merged into one bucket, pick whose
    canonical_name / homepage_url / entity_id the bucket keeps. Prefer a
    member whose normalized name matches its homepage domain (e.g. "Smith.ai"
    <-> smith.ai over "Claire AI" <-> smith.ai), then the one that carried
    the most surface forms, then first seen."""
    def _name_matches_domain(e: EntityRow) -> int:
        d = _domain_of(e.homepage_url)
        n = _norm_name(e.canonical_name)
        return 1 if d and n and (n in _norm_name(d) or _norm_name(d).startswith(n)) else 0

    return max(
        members,
        key=lambda e: (_name_matches_domain(e), len(e.surface_forms), -members.index(e)),
    )


def _merge_entities(entities: list[EntityRow]) -> list[EntityRow]:
    """Union entities across chunks: two entities merge when they share a
    normalized canonical_name or a homepage domain. surface_forms are
    unioned (order-preserving); the bucket's canonical_name / homepage_url /
    entity_id come from _pick_representative. A regenerated fallback id is
    used only where the chosen id would collide with another bucket's."""
    buckets: list[list[EntityRow]] = []
    by_name: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    for e in entities:
        nkey = _norm_name(e.canonical_name)
        dkey = _domain_of(e.homepage_url)
        idx = by_name.get(nkey)
        if idx is None and dkey:
            idx = by_domain.get(dkey)
        if idx is None:
            idx = len(buckets)
            buckets.append([])
        buckets[idx].append(e)
        by_name.setdefault(nkey, idx)
        if dkey:
            by_domain.setdefault(dkey, idx)

    out: list[EntityRow] = []
    used_ids: set[str] = set()
    for members in buckets:
        rep = _pick_representative(members)
        surface_forms: list[str] = []
        for e in members:
            for sf in e.surface_forms:
                if sf not in surface_forms:
                    surface_forms.append(sf)
        homepage_url = rep.homepage_url or next((e.homepage_url for e in members if e.homepage_url), None)

        eid = rep.entity_id or _norm_name(rep.canonical_name) or "entity"
        if eid in used_ids:
            base, n = eid, 2
            while eid in used_ids:
                eid = f"{base}-{n}"
                n += 1
        used_ids.add(eid)
        out.append(
            EntityRow(
                entity_id=eid,
                canonical_name=rep.canonical_name,
                surface_forms=tuple(surface_forms),
                homepage_url=homepage_url,
            )
        )
    return out


def call_entity_resolution(
    *,
    surface_form_candidates: list[dict[str, Any]],
    config: AnthropicConfig | None = None,
    client: MessagesClient | None = None,
) -> EntityResolutionResult:
    """One call for one chunk of surface forms (resolve_entities chunks the
    real-window ~2,200-row set and merges the results). Never raises."""
    cfg = config or AnthropicConfig.from_env()
    if config is None:
        cfg = replace(cfg, timeout_seconds=max(cfg.timeout_seconds, RESOLUTION_TIMEOUT_SECONDS))
    active_client = client or _build_client(cfg)

    kwargs: dict[str, Any] = {
        "model": RESOLUTION_MODEL,
        "max_tokens": 16384,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": _PROMPT_TEMPLATE.format(
                    surface_forms_json=json.dumps(surface_form_candidates)
                ),
            }
        ],
    }
    try:
        response = active_client.messages.create(**kwargs)
    except Exception as exc:  # contract: never raise past this point
        return EntityResolutionResult(error=f"{type(exc).__name__}: {exc}")

    return _parse_response(response)


def _parse_response(response: Any) -> EntityResolutionResult:
    parse_errors: list[str] = []

    def _safe(label: str, fn):
        try:
            return fn()
        except Exception as exc:
            parse_errors.append(f"{label}: {type(exc).__name__}: {exc}")
            return None

    model_returned = _safe("model_returned", lambda: getattr(response, "model", None))
    text = _safe("text", lambda: _extract_text(response))
    entities = (_safe("entities", lambda: _parse_entities_json(text)) if text else []) or []
    usage = _safe("usage", lambda: getattr(response, "usage", None))
    tokens_input = getattr(usage, "input_tokens", None) if usage is not None else None
    tokens_output = getattr(usage, "output_tokens", None) if usage is not None else None

    return EntityResolutionResult(
        entities=entities,
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


_REQUIRED_ENTITY_KEYS = ("entity_id", "canonical_name", "surface_forms")


def _parse_entities_json(text: str) -> list[EntityRow]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        # Output truncated at max_tokens mid-array: salvage the leading
        # complete objects so a large chunk still contributes.
        data = _salvage_entity_objects(stripped)
        if not data:
            raise
    return [
        EntityRow(
            entity_id=item["entity_id"],
            canonical_name=item["canonical_name"],
            surface_forms=tuple(item["surface_forms"]),
            homepage_url=item.get("homepage_url"),
        )
        for item in data
        if all(k in item for k in _REQUIRED_ENTITY_KEYS)
    ]


def _salvage_entity_objects(s: str) -> list[dict]:
    start = s.find("[")
    if start == -1:
        return []
    decoder = json.JSONDecoder()
    out: list[dict] = []
    i = start + 1
    while i < len(s):
        while i < len(s) and s[i] in " \t\r\n,":
            i += 1
        if i >= len(s) or s[i] == "]":
            break
        try:
            obj, end = decoder.raw_decode(s, i)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            out.append(obj)
        i = end
    return out


def _result_to_cache_dict(result: EntityResolutionResult) -> dict[str, Any]:
    return {
        "entities": [
            {
                "entity_id": e.entity_id,
                "canonical_name": e.canonical_name,
                "surface_forms": list(e.surface_forms),
                "homepage_url": e.homepage_url,
            }
            for e in result.entities
        ],
        "model_returned": result.model_returned,
        "error": result.error,
        "tokens_input": result.tokens_input,
        "tokens_output": result.tokens_output,
    }


def _result_from_cache_dict(data: dict[str, Any]) -> EntityResolutionResult:
    return EntityResolutionResult(
        entities=[
            EntityRow(
                entity_id=e["entity_id"],
                canonical_name=e["canonical_name"],
                surface_forms=tuple(e["surface_forms"]),
                homepage_url=e.get("homepage_url"),
            )
            for e in data["entities"]
        ],
        model_returned=data.get("model_returned"),
        error=data.get("error"),
        tokens_input=data.get("tokens_input"),
        tokens_output=data.get("tokens_output"),
    )


def resolve_entities(
    *,
    sources_by_run: dict[str, list[dict[str, Any]]],
    cache: LLMCache,
    call_fn: Callable[..., EntityResolutionResult] = call_entity_resolution,
    conn: sqlite3.Connection,
) -> EntityResolutionResult:
    rows = storage.list_surface_forms(conn=conn)
    candidates = build_surface_form_candidates(rows, sources_by_run)
    chunks = _chunk_candidates(candidates, RESOLUTION_CHUNK_SIZE)

    chunk_entities: list[EntityRow] = []
    errors: list[str] = []
    for n, chunk in enumerate(chunks):
        if not chunk:
            continue
        key = cache_key(
            input_text=json.dumps(chunk, sort_keys=True),
            model=RESOLUTION_MODEL,
            prompt_version=f"{PROMPT_VERSION}:chunk",
        )
        cached = cache.get(key)
        if cached is not None and not cached.get("error"):
            chunk_result = _result_from_cache_dict(cached)
        else:
            chunk_result = call_fn(surface_form_candidates=chunk)
            # Only cache a clean result -- an errored chunk (timeout /
            # transport / truncation past salvage) must be retried on the
            # next run, not frozen into the cache like a real answer.
            if not chunk_result.error:
                cache.set(key, _result_to_cache_dict(chunk_result))
        chunk_entities.extend(chunk_result.entities)
        if chunk_result.error:
            errors.append(f"chunk {n}: {chunk_result.error}")

    if errors:
        # Partial resolution -- do NOT write a half entity set that A4-prime
        # would then verify (and bill for). The failed chunks aren't cached,
        # so re-running resolve_entities retries just them; raise the same
        # "not done, resume later" signal the batch stages use on timeout.
        conn.commit()
        raise TimeoutError(
            f"A3 entity resolution incomplete: {len(errors)}/{len(chunks)} chunks failed "
            f"({'; '.join(errors)[:300]}). Re-run to retry the failed chunks."
        )

    entities = _merge_entities(chunk_entities)
    result = EntityResolutionResult(entities=entities, error=None)

    storage.clear_entities(conn=conn)
    if result.entities:
        storage.insert_entities(
            [
                {
                    "entity_id": e.entity_id,
                    "canonical_name": e.canonical_name,
                    "homepage_url": e.homepage_url,
                    "surface_forms": list(e.surface_forms),
                }
                for e in result.entities
            ],
            conn=conn,
        )

    return result
