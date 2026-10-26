"""Orchestration: entry data in, one appended JSONL line out.

Derives what the instrument (not the caller) must supply -- run_id (section
2), a UTC timestamp -- then calls an adapter and appends exactly one line to
data/01_raw/runs/YYYY-MM-DD.jsonl via collector.storage.jsonl. Engine-agnostic: any
adapter returning collector.engines.types.AdapterResult plugs in here, this
file never special-cases one engine (section 12).

Storage backend: append-only JSONL, not SQLite (collector.storage.api's
runs/sources tables stay defined but are not called from here -- the only
current SQLite consumer is analysis/aggregate/load_runs.py, a post-collection
pass, see INSTRUMENT.md section 12's one-way collector -> analysis rule). No
field extraction happens here: the adapter's exact request payload and
untouched response envelope pass through as-is (sections 3/7.2).
"""
from __future__ import annotations

import functools
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from collector.engines.types import AdapterResult
from collector.storage import jsonl

RUN_ID_SEPARATOR = "-"  # single named definition of the run_id shape


class InstrumentCommitError(RuntimeError):
    """Raised when the git commit that produced this row cannot be
    determined. Fails loudly rather than writing a placeholder -- a missing
    commit hash would silently invalidate the section 2 reproducibility
    claim, if/when this gets wired into the JSONL line."""


@functools.lru_cache(maxsize=1)
def get_instrument_commit() -> str:
    """`git rev-parse HEAD` for the running tree, cached per process. Not
    currently called by execute_run -- the pilot's JSONL line schema has no
    slot for it yet -- but kept intact since INSTRUMENT.md section 2 still
    requires an instrument commit be recoverable per row, and this is ready
    to wire in the moment that field is added back."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise InstrumentCommitError(
            "Could not determine the instrument's git commit (`git rev-parse "
            "HEAD` failed). Collection must run from inside a git checkout so "
            "every row traces back to the code that produced it "
            "(INSTRUMENT.md section 2)."
        ) from exc
    commit = result.stdout.strip()
    if not commit:
        raise InstrumentCommitError("`git rev-parse HEAD` returned an empty commit hash.")
    return commit


def build_run_id(
    *,
    engine: str,
    intent_id: int,
    paraphrase_id: int,
    day_index: int,
    rep_index: int,
) -> str:
    """Deterministic id encoding intent/paraphrase/engine/day/repetition
    (section 2). Same inputs -> same id, so a retry appends a second line
    carrying the same run_id rather than a silently unrelated one -- any
    reader takes the last line for a given run_id as authoritative
    (collector.storage.jsonl)."""
    parts = [engine, f"i{intent_id}", f"p{paraphrase_id}", f"d{day_index}", f"r{rep_index}"]
    return RUN_ID_SEPARATOR.join(parts)


def current_timestamp_utc() -> str:
    """Full ISO 8601 in UTC -- never date-only, never a naive/local offset."""
    return datetime.now(timezone.utc).isoformat()


AdapterCallable = Callable[..., AdapterResult]


def execute_run(
    *,
    adapter: AdapterCallable,
    engine: str,
    model_requested: str,
    intent_id: int,
    paraphrase_id: int,
    prompt_text: str,
    rep_index: int,
    day_index: int,
    prompt_order_seed: int,
    position_in_pass: int,
    runs_dir: str | Path | None = None,
    timeout_seconds: float | None = None,
    **adapter_kwargs: Any,
) -> str:
    """Call `adapter(prompt_text=..., model_requested=..., **adapter_kwargs)`
    and append exactly one JSONL line. Writes only after the call resolves,
    success or error -- never a pending-then-updated line (sections 3/10:
    nothing is edited in place).

    `adapter_kwargs` passes engine-specific extras through untouched (e.g.
    `config=`, `client=` for call_chatgpt) -- this function never inspects
    them, keeping it usable by any future adapter with the same call shape.

    `timeout_seconds`, when given, is a hard ceiling on the adapter call on
    top of whatever timeout/retry behavior the adapter's own SDK client
    already has (collector/config.py's shared 60s/2-retries defaults) -- a
    backstop for scripts/collect/run_collection.py's barrier-synchronized loop,
    where one stuck engine must not stall the others past a bounded wait.
    Enforced by running the call in a one-shot helper thread and giving up
    on `future.result(timeout=...)`: Python cannot cancel a running thread,
    so a timed-out call keeps running detached in the background rather than
    being killed -- this function never retries it, just stops waiting and
    writes a row saying so. On timeout, `request` is None: the adapter never
    returned far enough to hand back the payload it attempted to send, and
    that's the one thing about this call that truly cannot be recovered.
    """
    run_id = build_run_id(
        engine=engine,
        intent_id=intent_id,
        paraphrase_id=paraphrase_id,
        day_index=day_index,
        rep_index=rep_index,
    )
    ts_utc = current_timestamp_utc()

    if timeout_seconds is None:
        result = adapter(prompt_text=prompt_text, model_requested=model_requested, **adapter_kwargs)
    else:
        # No `with` block: ThreadPoolExecutor.__exit__ calls shutdown(wait=True),
        # which would block on the very call we're trying to time out. A bare
        # shutdown(wait=False) below lets the pool -- and the adapter call
        # still running inside it -- keep going on its own in the background.
        one_shot = ThreadPoolExecutor(max_workers=1)
        future = one_shot.submit(adapter, prompt_text=prompt_text, model_requested=model_requested, **adapter_kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            result = AdapterResult(error=f"timed out after {timeout_seconds}s")
        finally:
            one_shot.shutdown(wait=False)

    response = json.loads(result.raw_response_envelope) if (result.error is None and result.raw_response_envelope) else None

    jsonl.append_run_line(
        run_id=run_id,
        ts_utc=ts_utc,
        engine=engine,
        intent_id=intent_id,
        paraphrase_id=paraphrase_id,
        rep_index=rep_index,
        day_index=day_index,
        prompt_order_seed=prompt_order_seed,
        position_in_pass=position_in_pass,
        request=result.request_payload,
        response=response,
        error=result.error,
        runs_dir=runs_dir,
    )

    return run_id
