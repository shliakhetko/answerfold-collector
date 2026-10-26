"""A1 extractor, Anthropic backend. See extractors/base.py's module docstring
for the shared contract.

Self-judging disclosure (spec, "Calibration and checks"): this extractor's
vendor (Anthropic) also appears among the engines under test
(anthropic_claude in config/experiment.yaml) -- unavoidable given all 3 extractors
reuse this project's already-installed SDKs (analysis.coding.extractors,
one per collector/engines/*.py vendor). Written up in full in
validation_report.md, not resolved by swapping vendors here.
"""
from __future__ import annotations

import time
from typing import Any, Protocol

from analysis.coding.extractors.base import BatchItem, ExtractionResult, parse_mentions_json
from analysis.coding.extractors.models import EXTRACTOR_MODELS
from analysis.coding.extractors.prompts import build_extraction_prompt
from collector.config import AnthropicConfig

MAX_TOKENS = 2048  # generous ceiling for a JSON mentions array; not a study parameter

DEFAULT_POLL_INTERVAL_SECONDS = 30.0
DEFAULT_MAX_WAIT_SECONDS = 3600.0

# Anthropic's Message Batches API reports `processing_status` as
# "in_progress" while running, "ended" once done -- both confirmed via a
# live smoke test (2026-08-26), not just inferred from public docs.
_ANTHROPIC_IN_PROGRESS_STATUS = "in_progress"


class MessagesClient(Protocol):
    """Same structural type as collector/engines/anthropic_claude.py's --
    lets tests inject a fake object with a `.messages.create()` method."""
    messages: Any


def _build_client(config: AnthropicConfig) -> Any:
    import anthropic  # local import: a fake `client` injected by tests never
                       # needs the real `anthropic` package importable.
    return anthropic.Anthropic(
        api_key=config.api_key,
        timeout=config.timeout_seconds,
        max_retries=config.max_retries,
    )


def call_anthropic_extractor(
    *,
    raw_response_text: str,
    config: AnthropicConfig | None = None,
    client: MessagesClient | None = None,
) -> ExtractionResult:
    """Ask the Anthropic extractor model for every company/product mention
    in raw_response_text. Never raises -- mirrors collector/engines/*.py's
    adapter contract."""
    cfg = config or AnthropicConfig.from_env()
    active_client = client or _build_client(cfg)

    kwargs: dict[str, Any] = {
        "model": EXTRACTOR_MODELS["anthropic"],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,  # spec: "All LLM calls: temperature 0"
        "messages": [{"role": "user", "content": build_extraction_prompt(raw_response_text)}],
    }

    try:
        response = active_client.messages.create(**kwargs)
    except Exception as exc:  # contract: never raise past this point
        return ExtractionResult(error=f"{type(exc).__name__}: {exc}")

    return _parse_response(response)


def _parse_response(response: Any) -> ExtractionResult:
    parse_errors: list[str] = []

    def _safe(label: str, fn):
        try:
            return fn()
        except Exception as exc:
            parse_errors.append(f"{label}: {type(exc).__name__}: {exc}")
            return None

    model_returned = _safe("model_returned", lambda: getattr(response, "model", None))
    text = _safe("text", lambda: _extract_text(response))
    mentions = (_safe("mentions", lambda: parse_mentions_json(text)) if text else []) or []
    usage = _safe("usage", lambda: getattr(response, "usage", None))
    tokens_input = getattr(usage, "input_tokens", None) if usage is not None else None
    tokens_output = getattr(usage, "output_tokens", None) if usage is not None else None

    return ExtractionResult(
        mentions=mentions,
        model_returned=model_returned,
        error="; ".join(parse_errors) or None,
        raw_envelope=text,
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


def _extraction_params(raw_response_text: str) -> dict[str, Any]:
    return {
        "model": EXTRACTOR_MODELS["anthropic"],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "messages": [{"role": "user", "content": build_extraction_prompt(raw_response_text)}],
    }


def submit_anthropic_batch(
    items: list[BatchItem], *, config: AnthropicConfig | None = None, client: Any | None = None
) -> str:
    """One Anthropic Message Batch covering every item -- returns the batch
    id to poll/fetch. items' custom_id must be unique within the batch
    (Anthropic's own requirement); this repo uses run_id for extraction
    batches."""
    cfg = config or AnthropicConfig.from_env()
    active_client = client or _build_client(cfg)
    requests = [
        {"custom_id": item.custom_id, "params": _extraction_params(item.raw_response_text)} for item in items
    ]
    batch = active_client.messages.batches.create(requests=requests)
    return batch.id


def poll_anthropic_batch(
    batch_id: str,
    *,
    config: AnthropicConfig | None = None,
    client: Any | None = None,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
) -> str:
    """Blocks until the batch leaves "in_progress" (or max_wait_seconds is
    exceeded). Raises TimeoutError rather than hanging forever -- a rerun
    afterward can poll the same batch_id again, no work is lost."""
    cfg = config or AnthropicConfig.from_env()
    active_client = client or _build_client(cfg)
    elapsed = 0.0
    while True:
        batch = active_client.messages.batches.retrieve(batch_id)
        if batch.processing_status != _ANTHROPIC_IN_PROGRESS_STATUS:
            return batch.processing_status
        if elapsed >= max_wait_seconds:
            raise TimeoutError(
                f"Anthropic batch {batch_id} still {batch.processing_status} after {max_wait_seconds}s"
            )
        time.sleep(poll_interval_seconds)
        elapsed += poll_interval_seconds


def fetch_anthropic_batch_results(
    batch_id: str, *, config: AnthropicConfig | None = None, client: Any | None = None
) -> dict[str, ExtractionResult]:
    """Keyed by custom_id -- results are NOT guaranteed to come back in
    submission order (Anthropic's own documented behavior)."""
    cfg = config or AnthropicConfig.from_env()
    active_client = client or _build_client(cfg)
    results: dict[str, ExtractionResult] = {}
    for entry in active_client.messages.batches.results(batch_id):
        result_type = getattr(entry.result, "type", None)
        if result_type == "succeeded":
            results[entry.custom_id] = _parse_response(entry.result.message)
        else:
            results[entry.custom_id] = ExtractionResult(error=f"batch_result_{result_type}")
    return results
