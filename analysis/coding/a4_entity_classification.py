"""A4: entity classification (Post-Collection Coding Pipeline Spec). This
stage builds the population: meets_frame is what defines it. Classify from
the company's homepage, never from raw_response_text -- classifying from the
answer measures how the engine described them, a different variable.

meets_frame is a 4-way outcome (config/inclusion_frame.yaml, frozen
2026-09-05), not a bool: "in" | "out" | "borderline" | "unresolved". Only
"in" counts toward any denominator (b1_per_run_company.in_frame_entities) --
"borderline"/"unresolved" are excluded too, same as "out": an unresolved
call is not evidence of membership. "out" carries a frame_reason (one of
OUT_REASON_CODES); "borderline" carries a one-line explanation in the same
column.

Enriches the entities table (analysis/coding/schema.sql, written by A3)
with meets_frame/in_seed_list/segment/evidence_url/frame_reason.
HomepageFetcher stays a Protocol so this stage is tested against
FakeHomepageFetcher; the real implementation is
analysis/coding/homepage_fetch.py::RealHomepageFetcher (stdlib-only
urllib + html.parser fetch).
"""
from __future__ import annotations

import csv
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import yaml

from analysis.coding import storage
from analysis.coding.a3b_entity_verification import is_verified_vendor
from analysis.coding.cache import LLMCache, cache_key
from collector.config import AnthropicConfig

from collector import paths
DEFAULT_INCLUSION_FRAME_PATH = paths.INCLUSION_FRAME_YAML
DEFAULT_SEED_LIST_PATH = paths.SEED_VENDOR_LIST_CSV

PROMPT_VERSION = "v0-unfrozen"
CLASSIFICATION_MODEL = "claude-haiku-4-5-20251001"  # verified current cheap/fast-tier Anthropic model
MAX_TOKENS = 1024

# Batch API (Anthropic Message Batches) -- 50% cheaper, async. Same
# resumable "poll inline up to max_wait, else raise TimeoutError and park
# the batch id" contract as a3b_entity_verification.
DEFAULT_BATCH_MAX_WAIT_SECONDS = 3600.0
DEFAULT_BATCH_POLL_INTERVAL_SECONDS = 30.0
_IN_PROGRESS_STATUS = "in_progress"

MEETS_FRAME_VALUES = ("in", "out", "borderline", "unresolved")
OUT_REASON_CODES = ("not_a_company", "not_a_vendor", "end_user", "unrelated", "adjacent_product")
SEGMENTS = ("done-for-you", "dev platform", "no-code", "vertical-specific", "human_staffed")

# a3b_entity_verification's entity_type -> the A4-prime gate's OUT reason,
# for an entity that fails is_verified_vendor() and so never reaches the
# real frame-classification call. "unknown" and a missing/failed
# verification are not confident enough to call OUT -- see _gate_outcome.
_ENTITY_TYPE_OUT_REASON = {
    "generic": "not_a_company",
    "directory": "not_a_vendor",
    "publisher": "not_a_vendor",
    "platform": "unrelated",
    "feature": "adjacent_product",
}


def _gate_outcome(verification: dict[str, Any]) -> tuple[str, str | None]:
    """(meets_frame, frame_reason) for an entity the A4-prime gate excludes
    before it ever reaches the LLM frame check. Only called when
    is_verified_vendor(verification) is False, i.e. verification is not
    None -- a missing verification fails open (see classify_entities)."""
    is_real = verification.get("is_real")
    entity_type = verification.get("entity_type")
    if is_real is None:
        return "unresolved", None
    if is_real is False:
        return "out", "not_a_company"
    if entity_type == "unknown":
        return "unresolved", None
    return "out", _ENTITY_TYPE_OUT_REASON.get(entity_type, "not_a_vendor")


class HomepageFetcher(Protocol):
    def fetch(self, homepage_url: str) -> str: ...


@dataclass(frozen=True)
class ClassificationRow:
    meets_frame: str
    frame_reason: str | None
    segment: str | None
    evidence_url: str | None


@dataclass
class ClassificationResult:
    classification: ClassificationRow | None = None
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


_PROMPT_TEMPLATE = """{inclusion_frame_text}

---

Now apply the frame above to ONE company, using ONLY its homepage content
below -- never infer anything from how some AI search engine may have
described this company elsewhere, and never from mention counts.

COMPANY: {canonical_name}
HOMEPAGE URL: {homepage_url}
HOMEPAGE TEXT:
{homepage_text}

Return one JSON object with exactly these keys and nothing else -- no markdown code fence, no commentary:
- "meets_frame": one of "in", "out", "borderline", "unresolved".
- "frame_reason": if meets_frame is "out", one of not_a_company | not_a_vendor | end_user | unrelated | adjacent_product. If meets_frame is "borderline", a one-line explanation of the ambiguity. Otherwise null.
- "segment": one of "done-for-you", "dev platform", "no-code", "vertical-specific", "human_staffed" -- whichever best describes this company's offering. null unless meets_frame is "in".
- "evidence_url": the URL (homepage or a specific page on it) that best supports your decision."""


def _classification_kwargs(
    *, canonical_name: str, homepage_url: str, homepage_text: str, inclusion_frame_text: str
) -> dict[str, Any]:
    """messages.create params for one entity -- shared by the sync call and
    the Batch API request body so they stay identical."""
    return {
        "model": CLASSIFICATION_MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": _PROMPT_TEMPLATE.format(
                    inclusion_frame_text=inclusion_frame_text,
                    canonical_name=canonical_name,
                    homepage_url=homepage_url,
                    homepage_text=homepage_text,
                ),
            }
        ],
    }


def _classification_cache_key(
    canonical_name: str, homepage_url: str, homepage_text: str, inclusion_frame_text: str
) -> str:
    cache_input = json.dumps(
        {"canonical_name": canonical_name, "homepage_url": homepage_url,
         "homepage_text": homepage_text, "inclusion_frame_text": inclusion_frame_text},
        sort_keys=True,
    )
    return cache_key(input_text=cache_input, model=CLASSIFICATION_MODEL, prompt_version=PROMPT_VERSION)


def call_entity_classification(
    *,
    canonical_name: str,
    homepage_url: str,
    homepage_text: str,
    inclusion_frame_text: str,
    config: AnthropicConfig | None = None,
    client: MessagesClient | None = None,
) -> ClassificationResult:
    """Never raises -- mirrors collector/engines/*.py's adapter contract."""
    cfg = config or AnthropicConfig.from_env()
    active_client = client or _build_client(cfg)

    try:
        response = active_client.messages.create(
            **_classification_kwargs(
                canonical_name=canonical_name, homepage_url=homepage_url,
                homepage_text=homepage_text, inclusion_frame_text=inclusion_frame_text,
            )
        )
    except Exception as exc:  # contract: never raise past this point
        return ClassificationResult(error=f"{type(exc).__name__}: {exc}")

    return _parse_response(response)


def _parse_response(response: Any) -> ClassificationResult:
    parse_errors: list[str] = []

    def _safe(label: str, fn):
        try:
            return fn()
        except Exception as exc:
            parse_errors.append(f"{label}: {type(exc).__name__}: {exc}")
            return None

    model_returned = _safe("model_returned", lambda: getattr(response, "model", None))
    text = _safe("text", lambda: _extract_text(response))
    classification = _safe("classification", lambda: _parse_classification_json(text)) if text else None
    usage = _safe("usage", lambda: getattr(response, "usage", None))
    tokens_input = getattr(usage, "input_tokens", None) if usage is not None else None
    tokens_output = getattr(usage, "output_tokens", None) if usage is not None else None

    return ClassificationResult(
        classification=classification,
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


def _parse_classification_json(text: str) -> ClassificationRow:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    # raw_decode (not json.loads) so trailing content after the JSON object
    # -- a stray note, a repeated block -- doesn't fail the whole parse. On
    # the real run ~22% of A4 responses had exactly this shape.
    data, _end = json.JSONDecoder().raw_decode(stripped)
    meets_frame = str(data["meets_frame"]).strip().lower()
    if meets_frame not in MEETS_FRAME_VALUES:
        raise ValueError(f"meets_frame {meets_frame!r} not one of {MEETS_FRAME_VALUES}")
    segment = data.get("segment")
    if segment is not None and segment not in SEGMENTS:
        raise ValueError(f"segment {segment!r} not one of {SEGMENTS}")
    return ClassificationRow(
        meets_frame=meets_frame,
        frame_reason=data.get("frame_reason"),
        segment=segment if meets_frame == "in" else None,
        evidence_url=data.get("evidence_url"),
    )


def load_seed_list(seed_list_path: Path) -> set[str]:
    """Lowercased canonical_name/homepage_domain values -- either matching
    an entity into the seed list (spec: "used for recall validation only")."""
    values: set[str] = set()
    with seed_list_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(filter(lambda line: not line.startswith("#"), f)):
            if row.get("canonical_name"):
                values.add(row["canonical_name"].strip().lower())
            if row.get("homepage_domain"):
                values.add(row["homepage_domain"].strip().lower())
    return values


def load_inclusion_frame_text(inclusion_frame_path: Path) -> str:
    """config/inclusion_frame.yaml's `frame:` block scalar -- the actual
    classifier instructions -- without the version/frozen_date bookkeeping
    around it, so the prompt doesn't carry YAML-file noise. Falls back to
    the raw file text for a stub/legacy file with no `frame` key."""
    raw = inclusion_frame_path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw
    frame = data.get("frame") if isinstance(data, dict) else None
    return frame.strip() if isinstance(frame, str) else raw


def is_in_seed_list(*, canonical_name: str, homepage_url: str | None, seed_list: set[str]) -> bool:
    if canonical_name.strip().lower() in seed_list:
        return True
    if homepage_url:
        from urllib.parse import urlparse

        domain = (urlparse(homepage_url).netloc or "").lower().removeprefix("www.")
        if domain in seed_list:
            return True
    return False


# --------------------------------------------------------------------------
# Batch API path -- mirrors a3b_entity_verification's submit/poll/fetch +
# resumable prefill. custom_id is the entity's own cache key (a 64-char
# sha256 hex -> a valid Anthropic custom_id, already deduped per input).
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EntityClassificationBatchItem:
    custom_id: str
    canonical_name: str
    homepage_url: str
    homepage_text: str
    inclusion_frame_text: str


def submit_entity_classification_batch(
    items: list[EntityClassificationBatchItem], *, config: AnthropicConfig | None = None, client: Any | None = None
) -> str:
    cfg = config or AnthropicConfig.from_env()
    active_client = client or _build_client(cfg)
    requests = [
        {
            "custom_id": item.custom_id,
            "params": _classification_kwargs(
                canonical_name=item.canonical_name, homepage_url=item.homepage_url,
                homepage_text=item.homepage_text, inclusion_frame_text=item.inclusion_frame_text,
            ),
        }
        for item in items
    ]
    batch = active_client.messages.batches.create(requests=requests)
    return batch.id


def poll_entity_classification_batch(
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
            raise TimeoutError(
                f"entity-classification batch {batch_id} still {batch.processing_status} after {max_wait_seconds}s"
            )
        time.sleep(poll_interval_seconds)
        elapsed += poll_interval_seconds


def fetch_entity_classification_batch_results(
    batch_id: str, *, config: AnthropicConfig | None = None, client: Any | None = None
) -> dict[str, ClassificationResult]:
    cfg = config or AnthropicConfig.from_env()
    active_client = client or _build_client(cfg)
    out: dict[str, ClassificationResult] = {}
    for entry in active_client.messages.batches.results(batch_id):
        if getattr(entry.result, "type", None) == "succeeded":
            out[entry.custom_id] = _parse_response(entry.result.message)
        else:
            out[entry.custom_id] = ClassificationResult(
                error=f"batch_result_{getattr(entry.result, 'type', 'unknown')}"
            )
    return out


ENTITY_CLASSIFICATION_BATCH_FNS = (
    submit_entity_classification_batch,
    poll_entity_classification_batch,
    fetch_entity_classification_batch_results,
)


def _prefill_entity_classification_cache_via_batch(plan, cache, batch_fns, *, max_wait_seconds) -> None:
    """Submit ONE batch for every cache-miss entity, poll, write results
    into the cache. The batch id is parked in
    <cache_dir>/_pending_entity_classification_batch.json so a poll
    TimeoutError can be resumed by re-calling classify_entities (no
    re-submit / re-charge)."""
    submit_fn, poll_fn, fetch_fn = batch_fns
    state_path = cache.cache_dir / "_pending_entity_classification_batch.json"

    pending_by_id: dict[str, EntityClassificationBatchItem] = {}
    for _entity_id, canonical_name, homepage_url, homepage_text, frame_text, key in plan:
        if cache.get(key) is None:
            pending_by_id.setdefault(
                key,
                EntityClassificationBatchItem(key, canonical_name, homepage_url, homepage_text, frame_text),
            )

    if state_path.exists():
        batch_id = json.loads(state_path.read_text(encoding="utf-8"))["batch_id"]
    elif pending_by_id:
        cache.cache_dir.mkdir(parents=True, exist_ok=True)
        batch_id = submit_fn(list(pending_by_id.values()))
        state_path.write_text(json.dumps({"batch_id": batch_id, "n": len(pending_by_id)}), encoding="utf-8")
    else:
        return  # nothing to do

    poll_fn(batch_id, max_wait_seconds=max_wait_seconds)  # TimeoutError propagates; state_path stays
    fetched = fetch_fn(batch_id)
    for key, result in fetched.items():
        cache.set(key, _result_to_cache_dict(result))
    state_path.unlink(missing_ok=True)


def classify_entities(
    *,
    inclusion_frame_path: Path = DEFAULT_INCLUSION_FRAME_PATH,
    seed_list_path: Path = DEFAULT_SEED_LIST_PATH,
    fetcher: HomepageFetcher,
    cache: LLMCache,
    call_fn: Callable[..., ClassificationResult] = call_entity_classification,
    verification_by_entity: dict[str, dict[str, Any]] | None = None,
    use_batch: bool = False,
    batch_fns=ENTITY_CLASSIFICATION_BATCH_FNS,
    batch_max_wait_seconds: float = DEFAULT_BATCH_MAX_WAIT_SECONDS,
    conn: sqlite3.Connection,
) -> list[ClassificationResult]:
    inclusion_frame_text = load_inclusion_frame_text(inclusion_frame_path)
    seed_list = load_seed_list(seed_list_path)
    verification_by_entity = verification_by_entity or {}

    rows = storage.list_entities(conn=conn)

    def _homepage_url(row, verification) -> str:
        return row.get("homepage_url") or (verification or {}).get("verified_homepage_url") or ""

    # Plan the entities that get an LLM call: verified vendors only (the
    # A4-prime gate). Fetch each homepage once here; the loop reuses it.
    plan: list[tuple[str, str, str, str, str, str]] = []
    text_by_entity: dict[str, tuple[str, str]] = {}  # entity_id -> (homepage_url, cache_key)
    for row in rows:
        verification = verification_by_entity.get(row["entity_id"])
        if not is_verified_vendor(verification):
            continue
        homepage_url = _homepage_url(row, verification)
        homepage_text = fetcher.fetch(homepage_url) if homepage_url else ""
        key = _classification_cache_key(row["canonical_name"], homepage_url, homepage_text, inclusion_frame_text)
        plan.append((row["entity_id"], row["canonical_name"], homepage_url, homepage_text, inclusion_frame_text, key))
        text_by_entity[row["entity_id"]] = (homepage_url, key)

    if use_batch:
        _prefill_entity_classification_cache_via_batch(
            plan, cache, batch_fns, max_wait_seconds=batch_max_wait_seconds
        )

    call_by_key: dict[str, ClassificationResult] = {}
    for entity_id, canonical_name, homepage_url, homepage_text, _frame_text, key in plan:
        if key in call_by_key:
            continue
        cached = cache.get(key)
        if cached is not None:
            call_by_key[key] = _result_from_cache_dict(cached)
        else:
            result = call_fn(
                canonical_name=canonical_name, homepage_url=homepage_url, homepage_text=homepage_text,
                inclusion_frame_text=inclusion_frame_text,
            )
            cache.set(key, _result_to_cache_dict(result))
            call_by_key[key] = result

    results: list[ClassificationResult] = []
    for row in rows:
        verification = verification_by_entity.get(row["entity_id"])

        # A4-prime gate: an entity this stage confirmed is NOT a real,
        # discrete vendor never reaches the frozen-frame check -- record the
        # gate's own outcome (never "in": still "a count", never a
        # denominator) and skip the fetch + LLM call. Missing verification
        # = proceed to the real classification below.
        if not is_verified_vendor(verification):
            gate_meets_frame, gate_reason = _gate_outcome(verification)
            storage.update_entity_classification(
                row["entity_id"],
                meets_frame=gate_meets_frame,
                in_seed_list=is_in_seed_list(
                    canonical_name=row["canonical_name"],
                    homepage_url=_homepage_url(row, verification),
                    seed_list=seed_list,
                ),
                segment=None,
                evidence_url=(verification or {}).get("verified_homepage_url"),
                frame_reason=gate_reason,
                conn=conn,
            )
            results.append(ClassificationResult(classification=None))
            continue

        homepage_url, key = text_by_entity[row["entity_id"]]
        result = call_by_key[key]
        results.append(result)

        classification = result.classification
        storage.update_entity_classification(
            row["entity_id"],
            meets_frame=classification.meets_frame if classification else "unresolved",
            in_seed_list=is_in_seed_list(
                canonical_name=row["canonical_name"], homepage_url=homepage_url, seed_list=seed_list
            ),
            segment=classification.segment if classification else None,
            evidence_url=classification.evidence_url if classification else None,
            frame_reason=classification.frame_reason if classification else result.error,
            conn=conn,
        )

    return results


def _result_to_cache_dict(result: ClassificationResult) -> dict[str, Any]:
    return {
        "classification": (
            {
                "meets_frame": result.classification.meets_frame,
                "frame_reason": result.classification.frame_reason,
                "segment": result.classification.segment,
                "evidence_url": result.classification.evidence_url,
            }
            if result.classification
            else None
        ),
        "model_returned": result.model_returned,
        "error": result.error,
        "tokens_input": result.tokens_input,
        "tokens_output": result.tokens_output,
    }


def _result_from_cache_dict(data: dict[str, Any]) -> ClassificationResult:
    c = data.get("classification")
    return ClassificationResult(
        classification=ClassificationRow(**c) if c else None,
        model_returned=data.get("model_returned"),
        error=data.get("error"),
        tokens_input=data.get("tokens_input"),
        tokens_output=data.get("tokens_output"),
    )
