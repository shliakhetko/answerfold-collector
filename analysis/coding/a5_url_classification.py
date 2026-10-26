"""A5: URL classification (Post-Collection Coding Pipeline Spec).
Deduplicates the sources table on url_normalized, classifies each unique
URL once. Output: the url_registry table (analysis/coding/storage.py).

The orchestrator hands this stage the Task 5-resolved sources
(analysis/coding/cited_url_resolution.py), so Gemini's grounding-redirect
wrappers arrive already collapsed to their real destination and a citation
flagged `resolution_dead` is recorded as "other" with no LLM call.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from analysis.coding import storage
from analysis.coding.cache import LLMCache, cache_key
from analysis.coding.url_normalize import normalize_url
from collector.config import AnthropicConfig

PROMPT_VERSION = "v0-unfrozen"
CLASSIFICATION_MODEL = "claude-haiku-4-5-20251001"  # verified current cheap/fast-tier Anthropic model
MAX_TOKENS = 512

# Batch API (Anthropic Message Batches) -- 50% cheaper, async. Same
# resumable "poll inline up to max_wait, else raise TimeoutError and park
# the batch id" contract as a3b_entity_verification.
DEFAULT_BATCH_MAX_WAIT_SECONDS = 3600.0
DEFAULT_BATCH_POLL_INTERVAL_SECONDS = 30.0
_IN_PROGRESS_STATUS = "in_progress"

URL_CLASSES = ("vendor page", "vendor listicle", "review site", "forum", "news", "other")


@dataclass(frozen=True)
class URLClassificationRow:
    url_class: str
    publisher_is_in_category: bool
    classifier_confidence: float


@dataclass
class URLClassificationResult:
    classification: URLClassificationRow | None = None
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


_PROMPT_TEMPLATE = """You are classifying ONE cited URL from an AI search engine's answer about AI receptionist / AI voice agent products.

URL: {url_normalized}
TITLE (as cited): {title}

Return one JSON object with exactly these keys and nothing else -- no markdown code fence, no commentary:
- "url_class": one of "vendor page", "vendor listicle", "review site", "forum", "news", "other".
- "publisher_is_in_category": true if the site publishing this URL is itself a company selling an AI receptionist / AI voice agent product (regardless of url_class label), else false.
- "classifier_confidence": your confidence in this classification, a number from 0.0 to 1.0."""

# Frozen 2026-09-12 in config/url_classification_prompt.yaml, by content
# hash rather than by bumping PROMPT_VERSION -- the real 2026-09-05 run
# already used this exact text (confirmed: it never changed across this
# file's whole git history), and renaming PROMPT_VERSION would invalidate
# every one of its 3,966 cache keys for zero benefit. If this template ever
# needs to change, that edit must come with a new frozen_date + sha256 in
# the yaml file, a bumped PROMPT_VERSION, and an ambiguity-ledger entry --
# this assertion exists to make a silent edit impossible to miss.
PROMPT_SHA256 = "5da5b9b9bca02a0897008a7be3dae98f1c1b568efa2232cc6afb6a13778c8578"

if hashlib.sha256(_PROMPT_TEMPLATE.encode("utf-8")).hexdigest() != PROMPT_SHA256:
    raise RuntimeError(
        "A5 prompt template has drifted from the frozen copy in "
        "config/url_classification_prompt.yaml -- PROMPT_VERSION is "
        '"v0-unfrozen" and every cache key assumes this exact text. Either '
        "revert the prompt or freeze a new version (bump PROMPT_VERSION, "
        "update PROMPT_SHA256 here and sha256 in the yaml file, and add a "
        "new frozen_date + ambiguity-ledger entry)."
    )


def _classification_kwargs(url_normalized: str, title: str | None) -> dict[str, Any]:
    """messages.create params for one URL -- shared by the sync call and the
    Batch API request body so they stay identical."""
    return {
        "model": CLASSIFICATION_MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "messages": [
            {"role": "user", "content": _PROMPT_TEMPLATE.format(url_normalized=url_normalized, title=title or "")}
        ],
    }


def _url_cache_key(url_normalized: str, title: str | None) -> str:
    return cache_key(
        input_text=f"{url_normalized}\x00{title or ''}", model=CLASSIFICATION_MODEL, prompt_version=PROMPT_VERSION
    )


def call_url_classification(
    *, url_normalized: str, title: str | None, config: AnthropicConfig | None = None, client: MessagesClient | None = None
) -> URLClassificationResult:
    """Never raises -- mirrors collector/engines/*.py's adapter contract."""
    cfg = config or AnthropicConfig.from_env()
    active_client = client or _build_client(cfg)

    try:
        response = active_client.messages.create(**_classification_kwargs(url_normalized, title))
    except Exception as exc:  # contract: never raise past this point
        return URLClassificationResult(error=f"{type(exc).__name__}: {exc}")

    return _parse_response(response)


def _parse_response(response: Any) -> URLClassificationResult:
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

    return URLClassificationResult(
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


def _parse_classification_json(text: str) -> URLClassificationRow:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    data = json.loads(stripped)
    return URLClassificationRow(
        url_class=data["url_class"],
        publisher_is_in_category=bool(data["publisher_is_in_category"]),
        classifier_confidence=float(data["classifier_confidence"]),
    )


def dedupe_sources_by_normalized_url(sources_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """url_normalized -> first-seen source row (for its title). Sources
    without a usable url_raw are skipped."""
    deduped: dict[str, dict[str, Any]] = {}
    for source in sources_rows:
        url_raw = source.get("url_raw")
        if not url_raw:
            continue
        normalized = normalize_url(url_raw)
        deduped.setdefault(normalized, source)
    return deduped


# --------------------------------------------------------------------------
# Batch API path -- mirrors a3b_entity_verification's submit/poll/fetch +
# resumable prefill. custom_id is the URL's own cache key (a 64-char sha256
# hex -> a valid Anthropic custom_id, and already deduped per unique input).
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class URLClassificationBatchItem:
    custom_id: str
    url_normalized: str
    title: str | None


def submit_url_classification_batch(
    items: list[URLClassificationBatchItem], *, config: AnthropicConfig | None = None, client: Any | None = None
) -> str:
    cfg = config or AnthropicConfig.from_env()
    active_client = client or _build_client(cfg)
    requests = [
        {"custom_id": item.custom_id, "params": _classification_kwargs(item.url_normalized, item.title)}
        for item in items
    ]
    batch = active_client.messages.batches.create(requests=requests)
    return batch.id


def poll_url_classification_batch(
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
                f"url-classification batch {batch_id} still {batch.processing_status} after {max_wait_seconds}s"
            )
        time.sleep(poll_interval_seconds)
        elapsed += poll_interval_seconds


def fetch_url_classification_batch_results(
    batch_id: str, *, config: AnthropicConfig | None = None, client: Any | None = None
) -> dict[str, URLClassificationResult]:
    cfg = config or AnthropicConfig.from_env()
    active_client = client or _build_client(cfg)
    out: dict[str, URLClassificationResult] = {}
    for entry in active_client.messages.batches.results(batch_id):
        if getattr(entry.result, "type", None) == "succeeded":
            out[entry.custom_id] = _parse_response(entry.result.message)
        else:
            out[entry.custom_id] = URLClassificationResult(
                error=f"batch_result_{getattr(entry.result, 'type', 'unknown')}"
            )
    return out


URL_CLASSIFICATION_BATCH_FNS = (
    submit_url_classification_batch,
    poll_url_classification_batch,
    fetch_url_classification_batch_results,
)


def _prefill_url_classification_cache_via_batch(plan, cache, batch_fns, *, max_wait_seconds) -> None:
    """Submit ONE batch for every cache-miss URL, poll, write results into
    the cache. The batch id is parked in
    <cache_dir>/_pending_url_classification_batch.json so a poll TimeoutError
    can be resumed by re-calling classify_urls (no re-submit / re-charge)."""
    submit_fn, poll_fn, fetch_fn = batch_fns
    state_path = cache.cache_dir / "_pending_url_classification_batch.json"

    pending_by_id: dict[str, URLClassificationBatchItem] = {}
    for url_normalized, title, key in plan:
        if cache.get(key) is None:
            pending_by_id.setdefault(key, URLClassificationBatchItem(key, url_normalized, title))

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


def classify_urls(
    sources_rows: list[dict[str, Any]],
    *,
    cache: LLMCache,
    call_fn: Callable[..., URLClassificationResult] = call_url_classification,
    use_batch: bool = False,
    batch_fns=URL_CLASSIFICATION_BATCH_FNS,
    batch_max_wait_seconds: float = DEFAULT_BATCH_MAX_WAIT_SECONDS,
    conn: sqlite3.Connection,
) -> list[URLClassificationResult]:
    deduped = dedupe_sources_by_normalized_url(sources_rows)

    if use_batch:
        plan = [
            (url_normalized, deduped[url_normalized].get("title"),
             _url_cache_key(url_normalized, deduped[url_normalized].get("title")))
            for url_normalized in sorted(deduped)
            if not deduped[url_normalized].get("resolution_dead")
        ]
        _prefill_url_classification_cache_via_batch(plan, cache, batch_fns, max_wait_seconds=batch_max_wait_seconds)

    results: list[URLClassificationResult] = []
    output_rows: list[dict[str, Any]] = []
    for url_normalized in sorted(deduped):
        source = deduped[url_normalized]
        title = source.get("title")

        # Task 5 overlay (analysis/coding/cited_url_resolution.py): a citation
        # that resolved to a 404 / 410 / dead host / timeout / redirect loop
        # points nowhere to classify -- record it as "other" without paying
        # for an LLM call.
        if source.get("resolution_dead"):
            result = URLClassificationResult(
                classification=URLClassificationRow(
                    url_class="other", publisher_is_in_category=False, classifier_confidence=0.0
                )
            )
            results.append(result)
            output_rows.append(
                {
                    "url_normalized": url_normalized,
                    "url_class": "other",
                    "publisher_is_in_category": False,
                    "classifier_confidence": 0.0,
                }
            )
            continue

        key = _url_cache_key(url_normalized, title)
        cached = cache.get(key)
        if cached is not None:
            result = _result_from_cache_dict(cached)
        else:
            result = call_fn(url_normalized=url_normalized, title=title)
            cache.set(key, _result_to_cache_dict(result))
        results.append(result)

        c = result.classification
        output_rows.append(
            {
                "url_normalized": url_normalized,
                "url_class": c.url_class if c else None,
                "publisher_is_in_category": c.publisher_is_in_category if c else None,
                "classifier_confidence": c.classifier_confidence if c else None,
            }
        )

    storage.clear_url_registry(conn=conn)
    if output_rows:
        storage.insert_url_registry(output_rows, conn=conn)

    return results


def _result_to_cache_dict(result: URLClassificationResult) -> dict[str, Any]:
    return {
        "classification": (
            {
                "url_class": result.classification.url_class,
                "publisher_is_in_category": result.classification.publisher_is_in_category,
                "classifier_confidence": result.classification.classifier_confidence,
            }
            if result.classification
            else None
        ),
        "model_returned": result.model_returned,
        "error": result.error,
        "tokens_input": result.tokens_input,
        "tokens_output": result.tokens_output,
    }


def _result_from_cache_dict(data: dict[str, Any]) -> URLClassificationResult:
    c = data.get("classification")
    return URLClassificationResult(
        classification=URLClassificationRow(**c) if c else None,
        model_returned=data.get("model_returned"),
        error=data.get("error"),
        tokens_input=data.get("tokens_input"),
        tokens_output=data.get("tokens_output"),
    )
