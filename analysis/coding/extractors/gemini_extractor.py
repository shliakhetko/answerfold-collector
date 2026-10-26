"""A1 extractor, Gemini backend. See extractors/base.py's module docstring
for the shared contract.

Self-judging disclosure (spec, "Calibration and checks"): this extractor's
vendor (Google) also appears among the engines under test (google_gemini and
google_ai_mode in config/experiment.yaml -- doubly so) -- unavoidable given all 3
extractors reuse this project's already-installed SDKs. Written up in full
in validation_report.md, not resolved by swapping vendors here.
"""
from __future__ import annotations

import time
from typing import Any, Protocol

from analysis.coding.extractors.base import BatchItem, ExtractionResult, parse_mentions_json
from analysis.coding.extractors.models import EXTRACTOR_MODELS
from analysis.coding.extractors.prompts import build_extraction_prompt
from collector.config import GeminiConfig

DEFAULT_POLL_INTERVAL_SECONDS = 30.0
DEFAULT_MAX_WAIT_SECONDS = 3600.0

# google-genai's batch job states while still running -- terminal states
# (JOB_STATE_SUCCEEDED/FAILED/CANCELLED/PAUSED) confirmed via context7
# against this SDK's docs, and JOB_STATE_SUCCEEDED confirmed again via a
# live smoke test (2026-08-26; job.state came back as a JobState enum
# member, not a plain str, but compares equal to these string values --
# str-subclassing enum, confirmed working, no change needed). Anything not
# in this set is treated as terminal.
_GEMINI_IN_PROGRESS_STATES = {"JOB_STATE_PENDING", "JOB_STATE_RUNNING", "JOB_STATE_QUEUED"}


class GenerativeClient(Protocol):
    """Same structural type as collector/engines/google_gemini.py's -- lets
    tests inject a fake object with a `.models.generate_content()` method."""
    models: Any


def _build_client(config: GeminiConfig) -> Any:
    from google import genai  # local import: a fake `client` injected by
                               # tests never needs the real package importable.
    from google.genai import types

    http_options = types.HttpOptions(
        timeout=int(config.timeout_seconds * 1000),  # SDK takes milliseconds
        retry_options=types.HttpRetryOptions(attempts=config.max_retries),
    )
    return genai.Client(api_key=config.api_key, http_options=http_options)


def call_gemini_extractor(
    *,
    raw_response_text: str,
    config: GeminiConfig | None = None,
    client: GenerativeClient | None = None,
) -> ExtractionResult:
    """Ask the Gemini extractor model for every company/product mention in
    raw_response_text. Never raises -- mirrors collector/engines/*.py's
    adapter contract."""
    cfg = config or GeminiConfig.from_env()
    active_client = client or _build_client(cfg)

    from google.genai import types

    try:
        response = active_client.models.generate_content(
            model=EXTRACTOR_MODELS["gemini"],
            contents=build_extraction_prompt(raw_response_text),
            config=types.GenerateContentConfig(temperature=0),  # spec: temperature 0
        )
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

    model_returned = _safe("model_returned", lambda: getattr(response, "model_version", None))
    text = _safe("text", lambda: _extract_text(response))
    mentions = (_safe("mentions", lambda: parse_mentions_json(text)) if text else []) or []
    usage = _safe("usage", lambda: getattr(response, "usage_metadata", None))
    tokens_input = getattr(usage, "prompt_token_count", None) if usage is not None else None
    tokens_output = getattr(usage, "candidates_token_count", None) if usage is not None else None

    return ExtractionResult(
        mentions=mentions,
        model_returned=model_returned,
        error="; ".join(parse_errors) or None,
        raw_envelope=text,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
    )


def _extract_text(response: Any) -> str | None:
    # Deliberately never reads response.text as a shortcut here either (see
    # analysis/aggregate/response_fields.py's docstring) -- walks
    # candidates[0].content.parts directly.
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    chunks = [getattr(part, "text", None) for part in parts or [] if getattr(part, "text", None)]
    return "\n".join(chunks) if chunks else None


def submit_gemini_batch(
    items: list[BatchItem], *, config: GeminiConfig | None = None, client: Any | None = None
) -> str:
    """One Gemini batch prediction job covering every item, inline (not
    file-uploaded -- fine at this repo's scale). Returns the job name to
    poll/fetch."""
    cfg = config or GeminiConfig.from_env()
    active_client = client or _build_client(cfg)
    src = [
        {
            "contents": [{"parts": [{"text": build_extraction_prompt(item.raw_response_text)}], "role": "user"}],
            "config": {"temperature": 0},
        }
        for item in items
    ]
    job = active_client.batches.create(model=EXTRACTOR_MODELS["gemini"], src=src)
    return job.name


def poll_gemini_batch(
    job_name: str,
    *,
    config: GeminiConfig | None = None,
    client: Any | None = None,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
) -> str:
    """Blocks until the job leaves _GEMINI_IN_PROGRESS_STATES (or
    max_wait_seconds is exceeded). Raises TimeoutError rather than hanging
    forever -- a rerun afterward can poll the same job_name again."""
    cfg = config or GeminiConfig.from_env()
    active_client = client or _build_client(cfg)
    elapsed = 0.0
    while True:
        job = active_client.batches.get(name=job_name)
        if job.state not in _GEMINI_IN_PROGRESS_STATES:
            return job.state
        if elapsed >= max_wait_seconds:
            raise TimeoutError(f"Gemini batch {job_name} still {job.state} after {max_wait_seconds}s")
        time.sleep(poll_interval_seconds)
        elapsed += poll_interval_seconds


def _get(obj: Any, name: str) -> Any:
    """dict-or-attribute access -- inline batch results may come back as
    plain dicts or as SDK objects depending on google-genai version; see
    fetch_gemini_batch_results' docstring."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def fetch_gemini_batch_results(
    job_name: str, items: list[BatchItem], *, config: GeminiConfig | None = None, client: Any | None = None
) -> dict[str, ExtractionResult]:
    """items must be the SAME list, in the SAME order, passed to
    submit_gemini_batch. Both the field name (job.dest.inlined_responses)
    and the strict 1:1 positional-order correlation (no explicit per-item
    key exists) were confirmed via a live smoke test (2026-08-26) -- a
    2-item batch came back in submission order, matched correctly by index.
    Still raises rather than silently mis-attributing a result to the wrong
    run if the counts ever don't match (e.g. a future SDK version changing
    this behavior)."""
    cfg = config or GeminiConfig.from_env()
    active_client = client or _build_client(cfg)
    job = active_client.batches.get(name=job_name)
    dest = _get(job, "dest")
    raw_responses = (_get(dest, "inlined_responses") or []) if dest is not None else []
    if len(raw_responses) != len(items):
        raise ValueError(
            f"Gemini batch {job_name}: got {len(raw_responses)} results for {len(items)} submitted items -- "
            "positional correlation assumption broken, needs live re-verification (see this function's docstring)"
        )

    results: dict[str, ExtractionResult] = {}
    for item, raw in zip(items, raw_responses):
        response = _get(raw, "response")
        if response is None:
            results[item.custom_id] = ExtractionResult(error=f"batch_error: {_get(raw, 'error')}")
            continue
        results[item.custom_id] = _parse_response(response)
    return results
