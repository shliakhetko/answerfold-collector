"""A1 extractor, OpenAI backend. See extractors/base.py's module docstring
for the shared contract.

Self-judging disclosure (spec, "Calibration and checks"): this extractor's
vendor (OpenAI) also appears among the engines under test (openai_chatgpt in
config/experiment.yaml) -- unavoidable given all 3 extractors reuse this project's
already-installed SDKs. Written up in full in validation_report.md, not
resolved by swapping vendors here.
"""
from __future__ import annotations

import json
import time
from typing import Any, Protocol

from analysis.coding.extractors.arbiter import ArbiterResult, parse_votes_json
from analysis.coding.extractors.base import BatchItem, ExtractionResult, dict_to_namespace, parse_mentions_json
from analysis.coding.extractors.models import EXTRACTOR_MODELS
from analysis.coding.extractors.prompts import build_arbiter_prompt, build_extraction_prompt
from collector.config import OpenAIConfig

DEFAULT_POLL_INTERVAL_SECONDS = 30.0
DEFAULT_MAX_WAIT_SECONDS = 3600.0

# openai.types.Batch.status values while a batch is still running --
# "completed" confirmed as the terminal status via a live smoke test
# (2026-08-26) against SDK version 3.0.0; the in-progress values themselves
# are inferred from OpenAI's documented state machine, not yet individually
# observed (the smoke test's tiny batch moved through them too fast to
# catch). Anything not in this set is treated as terminal.
_OPENAI_IN_PROGRESS_STATUSES = {"validating", "in_progress", "finalizing"}


class ResponsesClient(Protocol):
    """Same structural type as collector/engines/openai_chatgpt.py's --
    lets tests inject a fake object with a `.responses.create()` method."""
    responses: Any


def _build_client(config: OpenAIConfig) -> Any:
    import openai  # local import: a fake `client` injected by tests never
                    # needs the real `openai` package importable.
    return openai.OpenAI(
        api_key=config.api_key,
        timeout=config.timeout_seconds,
        max_retries=config.max_retries,
    )


def call_openai_extractor(
    *,
    raw_response_text: str,
    config: OpenAIConfig | None = None,
    client: ResponsesClient | None = None,
) -> ExtractionResult:
    """Ask the OpenAI extractor model for every company/product mention in
    raw_response_text. Never raises -- mirrors collector/engines/*.py's
    adapter contract."""
    cfg = config or OpenAIConfig.from_env()
    active_client = client or _build_client(cfg)

    kwargs: dict[str, Any] = {
        "model": EXTRACTOR_MODELS["openai"],
        "input": build_extraction_prompt(raw_response_text),
        # No `temperature` here (spec wants "temperature 0" for every LLM
        # call, but verified live against EXTRACTOR_MODELS["openai"]: this
        # reasoning-tier model rejects the parameter outright --
        # BadRequestError "Unsupported parameter: 'temperature' is not
        # supported with this model." Same reason
        # collector/engines/openai_chatgpt.py's adapter never sets it
        # either. If a future non-reasoning model replaces this one,
        # re-add temperature=0.
    }

    try:
        response = active_client.responses.create(**kwargs)
    except Exception as exc:  # contract: never raise past this point
        return ExtractionResult(error=f"{type(exc).__name__}: {exc}")

    return _parse_response(response)


def call_openai_arbiter_vote(
    *,
    raw_response_text: str,
    candidates: list[dict[str, str]],
    config: OpenAIConfig | None = None,
    client: ResponsesClient | None = None,
) -> ArbiterResult:
    """OpenAI implementation of a1_answer_coding.py's ARBITER_EXTRACTOR role
    (see arbiter.py for the Anthropic one + shared types). One call per run
    covers that run's whole disputed set. Never raises -- same contract as
    call_arbiter_vote. Synchronous only: OpenAI's Batch API is unusable on
    this account, and the arbiter only ever touches the disputed runs."""
    if not candidates:
        return ArbiterResult()  # nothing to adjudicate -- never call the model for an empty set

    cfg = config or OpenAIConfig.from_env()
    active_client = client or _build_client(cfg)

    candidates_json = json.dumps(
        [{"surface_form": c["surface_form"], "evidence_span": c["evidence_span"]} for c in candidates]
    )
    try:
        response = active_client.responses.create(
            model=EXTRACTOR_MODELS["openai"],
            input=build_arbiter_prompt(raw_response_text, candidates_json),
        )
    except Exception as exc:  # contract: never raise past this point
        return ArbiterResult(error=f"{type(exc).__name__}: {exc}")

    parse_errors: list[str] = []

    def _safe(label: str, fn):
        try:
            return fn()
        except Exception as exc:
            parse_errors.append(f"{label}: {type(exc).__name__}: {exc}")
            return None

    model_returned = _safe("model_returned", lambda: getattr(response, "model", None))
    text = _safe("text", lambda: _extract_text(response))
    votes = (_safe("votes", lambda: parse_votes_json(text)) if text else []) or []
    usage = _safe("usage", lambda: getattr(response, "usage", None))
    return ArbiterResult(
        votes=votes,
        model_returned=model_returned,
        error="; ".join(parse_errors) or None,
        raw_envelope=text,
        tokens_input=getattr(usage, "input_tokens", None) if usage is not None else None,
        tokens_output=getattr(usage, "output_tokens", None) if usage is not None else None,
    )


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
    # Deliberately never reads response.output_text as a shortcut here
    # either (see analysis/aggregate/response_fields.py's docstring for why
    # that's a computed property, not safe to rely on for a dict-shaped
    # cache/replay path) -- walks `output` directly, same fallback branch
    # collector/engines/openai_chatgpt.py's own extractor falls back to.
    chunks: list[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for block in getattr(item, "content", None) or []:
            if getattr(block, "type", None) != "output_text":
                continue
            block_text = getattr(block, "text", None)
            if block_text:
                chunks.append(block_text)
    return "\n".join(chunks) if chunks else None


def submit_openai_batch(
    items: list[BatchItem], *, config: OpenAIConfig | None = None, client: Any | None = None
) -> str:
    """Uploads a JSONL request file (one line per item, OpenAI's documented
    batch request format) then creates the batch job against it. Returns
    the batch id to poll/fetch."""
    cfg = config or OpenAIConfig.from_env()
    active_client = client or _build_client(cfg)
    lines = [
        json.dumps({
            "custom_id": item.custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": {"model": EXTRACTOR_MODELS["openai"], "input": build_extraction_prompt(item.raw_response_text)},
        })
        for item in items
    ]
    jsonl_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    file_id = _upload_batch_input(active_client, jsonl_bytes)
    _wait_for_file_processed(active_client, file_id)
    batch = active_client.batches.create(input_file_id=file_id, endpoint="/v1/responses", completion_window="24h")
    return batch.id


def _upload_batch_input(client: Any, jsonl_bytes: bytes) -> str:
    """Upload the batch-input JSONL, returning its file id.

    Uses the chunked Uploads API (POST /uploads + parts + complete) instead
    of a single files.create() multipart POST. A ~16 MB batch input uploaded
    in one shot reported status "processed" but was then unresolvable by the
    Batch service: every batch against it failed validation with "Cannot
    find file <id>, or organization does not have access to it", zero
    requests run (observed 3x on the real A1 input, 2026-09-01/02). The
    <200 KB trial inputs never hit this. Falls back to files.create for a
    fake client with no uploads resource (tests).
    """
    chunked = getattr(getattr(client, "uploads", None), "upload_file_chunked", None)
    if chunked is not None:
        upload = chunked(
            file=jsonl_bytes,
            mime_type="application/jsonl",
            purpose="batch",
            filename="batch_input.jsonl",
            bytes=len(jsonl_bytes),
        )
        return upload.file.id
    return client.files.create(file=("batch_input.jsonl", jsonl_bytes), purpose="batch").id


def _wait_for_file_processed(
    client: Any, file_id: str, *, poll_interval_seconds: float = 3.0, max_wait_seconds: float = 300.0
) -> None:
    """Block until an uploaded batch-input file leaves status "pending".

    Belt-and-braces alongside _upload_batch_input's chunked upload: a batch
    created against a still-"pending" file fails validation with "Cannot
    find file <id>, or organization does not have access to it" and the
    whole batch is marked failed (observed on the ~16 MB A1 input,
    2026-09-01/02, zero requests run). No-op for a fake client without
    files.retrieve.
    """
    retrieve = getattr(getattr(client, "files", None), "retrieve", None)
    if retrieve is None:
        return
    elapsed = 0.0
    while True:
        status = getattr(retrieve(file_id), "status", "processed")
        if status == "processed":
            return
        if status == "error":
            raise RuntimeError(f"OpenAI batch-input file {file_id} failed processing (status=error)")
        if elapsed >= max_wait_seconds:
            raise TimeoutError(f"OpenAI batch-input file {file_id} still {status} after {max_wait_seconds}s")
        time.sleep(poll_interval_seconds)
        elapsed += poll_interval_seconds


def poll_openai_batch(
    batch_id: str,
    *,
    config: OpenAIConfig | None = None,
    client: Any | None = None,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
) -> str:
    """Blocks until the batch leaves _OPENAI_IN_PROGRESS_STATUSES (or
    max_wait_seconds is exceeded). Raises TimeoutError rather than hanging
    forever -- a rerun afterward can poll the same batch_id again."""
    cfg = config or OpenAIConfig.from_env()
    active_client = client or _build_client(cfg)
    elapsed = 0.0
    while True:
        batch = active_client.batches.retrieve(batch_id)
        if batch.status not in _OPENAI_IN_PROGRESS_STATUSES:
            return batch.status
        if elapsed >= max_wait_seconds:
            raise TimeoutError(f"OpenAI batch {batch_id} still {batch.status} after {max_wait_seconds}s")
        time.sleep(poll_interval_seconds)
        elapsed += poll_interval_seconds


def fetch_openai_batch_results(
    batch_id: str, *, config: OpenAIConfig | None = None, client: Any | None = None
) -> dict[str, ExtractionResult]:
    """Keyed by custom_id. Reads output_file_id (succeeded requests) and
    error_file_id (failed requests) separately -- OpenAI's documented split,
    a batch can have both. Each output line's response.body is a plain
    dict (parsed JSON, not an SDK object) -- dict_to_namespace() lets the
    same _parse_response()/_extract_text() the sync path uses read it via
    the identical getattr()-based access."""
    cfg = config or OpenAIConfig.from_env()
    active_client = client or _build_client(cfg)
    batch = active_client.batches.retrieve(batch_id)
    results: dict[str, ExtractionResult] = {}

    if batch.output_file_id:
        content = active_client.files.content(batch.output_file_id).text
        for line in content.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            body = (record.get("response") or {}).get("body")
            if body is not None:
                results[record["custom_id"]] = _parse_response(dict_to_namespace(body))
            else:
                results[record["custom_id"]] = ExtractionResult(error=f"batch_error: {record.get('error')}")

    if batch.error_file_id:
        content = active_client.files.content(batch.error_file_id).text
        for line in content.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            results.setdefault(record["custom_id"], ExtractionResult(error=f"batch_error: {record.get('error')}"))

    return results
