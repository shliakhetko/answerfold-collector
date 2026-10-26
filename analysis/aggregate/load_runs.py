"""Load data/01_raw/runs/*.jsonl into the collector's SQLite `runs` table.

Analysis-layer code (INSTRUMENT.md section 12: analysis/ may import from
collector/, never the reverse) -- this is the post-collection pass that
finally calls collector.storage.api.insert_run, which collector/runner/
execute.py deliberately does not (the pilot's live write path is JSONL-only,
collector/storage/jsonl.py).

Scope: this pass maps the fields that are either a direct copy from the
JSONL line, a token/search-fired/finish-reason extraction
(analysis.aggregate.usage plus this module's own _extract_finish_reason), or
a raw_response_text/model_returned/sources extraction
(analysis.aggregate.response_fields). raw_response_envelope keeps the
original response recoverable regardless, in case any of these parses is
later found to be wrong.
reasoning_tokens/tool_tokens and the vendor-native usage sub-fields
from analysis.aggregate.usage.extract_extra_usage_fields (cache hits/writes,
tool-use counts) each have their own schema.sql column -- populated by
whichever engine reports them, NULL for the rest, never packed into a JSON
blob (raw_response_envelope is the only JSON column on this table).

instrument_commit: data/01_raw/runs/*.jsonl does not carry this field (a pre-existing
gap against INSTRUMENT.md section 2, out of scope to fix here -- see
collector/runner/execute.py's get_instrument_commit docstring). Callers must
supply it explicitly; it cannot be reconstructed from an already-written
line, and this loader never guesses one.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analysis.aggregate.response_fields import (
    extract_model_returned,
    extract_raw_response_text,
    extract_sources_rows,
)
from analysis.aggregate.usage import extract_extra_usage_fields, extract_search_fired, extract_usage
from collector.storage import api
from collector.storage.db import get_connection, init_db

from collector import paths
DEFAULT_RUNS_DIR = paths.RUNS_DIR

# request[...] key holding the prompt text sent, per engine -- see each
# engine's request_payload construction in collector/engines/*.py.
_PROMPT_KEY = {
    "openai_chatgpt": lambda request: request.get("input"),
    "google_gemini": lambda request: request.get("contents"),
    "anthropic_claude": lambda request: (request.get("messages") or [{}])[0].get("content"),
    "google_ai_mode": lambda request: request.get("q"),
}


# Mirrors the key path each adapter reads off the *live* SDK/response object
# (collector/engines/anthropic_claude.py, openai_chatgpt.py, google_gemini.py,
# google_ai_mode.py) -- here read off the already-stored dict form instead
# (raw_response_envelope is response.model_dump() JSON for the three SDK
# engines, and SerpApi's own raw JSON for google_ai_mode), since this loader
# only ever sees the serialized JSONL line, never the live SDK object.
def _extract_finish_reason(engine: str, response: dict[str, Any] | None) -> str | None:
    if not response:
        return None
    if engine == "anthropic_claude":
        return response.get("stop_reason")
    if engine == "openai_chatgpt":
        incomplete = response.get("incomplete_details")
        if incomplete:
            return incomplete.get("reason")
        return response.get("status")
    if engine == "google_gemini":
        candidates = response.get("candidates") or []
        return candidates[0].get("finish_reason") if candidates else None
    if engine == "google_ai_mode":
        return (response.get("search_metadata") or {}).get("status")
    return None


# Per-engine truncation signal -- the vendor-specific finish_reason value
# that means "the response was cut off by a token cap," not a genuine stop.
# google_ai_mode has no token concept (SerpApi bills per search), so it can
# never be truncated this way.
_MAX_TOKENS_FINISH_REASONS = {
    "anthropic_claude": "max_tokens",
    "openai_chatgpt": "max_output_tokens",
    "google_gemini": "MAX_TOKENS",
}


def is_max_tokens_truncated(engine: str, finish_reason: str | None) -> bool:
    """True if this row's answer was cut off by a token cap rather than
    stopping naturally. Per INSTRUMENT.md section 3, collection stays
    raw-only -- this is a post-collection eligibility check, same tier as
    excluding rows with `error is not None` from denominators (section 8.1),
    not a filter applied while writing. A truncated row undercounts whatever
    would have appeared after the cutoff, so it must never be analyzed as if
    it were a complete answer."""
    return finish_reason is not None and finish_reason == _MAX_TOKENS_FINISH_REASONS.get(engine)


def _prompt_text(engine: str, request: dict[str, Any] | None) -> str:
    request = request or {}
    getter = _PROMPT_KEY.get(engine)
    text = getter(request) if getter else None
    # NOT NULL in schema.sql -- an engine this loader doesn't recognize yet,
    # or a request payload missing the expected key, must surface as an
    # empty string rather than crash the whole load, matching "an adapter
    # that cannot determine this must say so" (schema.sql's search_fired
    # comment) applied to prompt_text.
    return text if isinstance(text, str) else ""


def parse_run_record(record: dict[str, Any], *, instrument_commit: str) -> dict[str, Any]:
    """Map one data/01_raw/runs/*.jsonl line to collector.storage.api.insert_run kwargs."""
    engine = record["engine"]
    response = record.get("response")
    usage = extract_usage(engine, response)
    extra_fields = extract_extra_usage_fields(engine, response)

    scraper_vendor = scraper_endpoint = None
    if engine == "google_ai_mode":
        scraper_vendor, scraper_endpoint = "serpapi", "google_ai_mode"

    return dict(
        run_id=record["run_id"],
        instrument_commit=instrument_commit,
        timestamp=record["ts_utc"],
        engine=engine,
        intent_id=record["intent_id"],
        paraphrase_id=record["paraphrase_id"],
        prompt_text=_prompt_text(engine, record.get("request")),
        rep_index=record["rep_index"],
        day_index=record["day_index"],
        model_requested=(record.get("request") or {}).get("model"),
        model_returned=extract_model_returned(engine, response),
        raw_response_text=extract_raw_response_text(engine, response),
        error=record.get("error"),
        finish_reason=_extract_finish_reason(engine, response),
        search_fired=extract_search_fired(engine, response),
        tokens_input=usage.input,
        tokens_output=usage.output,
        tokens_total=usage.total,
        reasoning_tokens=usage.reasoning_tokens,
        tool_tokens=usage.tool_tokens,
        cached_tokens=extra_fields.get("cached_tokens"),
        cache_write_tokens=extra_fields.get("cache_write_tokens"),
        cached_content_tokens=extra_fields.get("cached_content_tokens"),
        cache_creation_tokens=extra_fields.get("cache_creation_tokens"),
        cache_read_tokens=extra_fields.get("cache_read_tokens"),
        web_search_requests=extra_fields.get("web_search_requests"),
        web_fetch_requests=extra_fields.get("web_fetch_requests"),
        scraper_vendor=scraper_vendor,
        scraper_endpoint=scraper_endpoint,
        raw_response_envelope=None if response is None else json.dumps(response),
    )


def sources_rows_for_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Thin wrapper around response_fields.extract_sources_rows, kept
    separate from parse_run_record's return value since these are
    collector.storage.api.insert_sources kwargs, not insert_run kwargs."""
    return extract_sources_rows(record["engine"], record.get("response"))


@dataclass
class LoadStats:
    inserted: int = 0
    skipped_existing: int = 0
    files: int = 0


def load_jsonl_dir(
    runs_dir: str | Path | None = None,
    *,
    db_path: str | Path | None = None,
    instrument_commit: str,
) -> LoadStats:
    """Insert every data/01_raw/runs/*.jsonl line into the runs table. Safe to re-run:
    a run_id already present in the DB is counted in skipped_existing and
    left alone (INSTRUMENT.md section 10: nothing is edited in place)
    rather than raising, since more *.jsonl files land as the collection
    window progresses and this loader is expected to run again each time.

    Within this load, a retried run_id (multiple lines, same id -- a failed
    call retried into a fresh attempt) is deduplicated by keeping only the
    LAST line for that run_id, matching collector.storage.jsonl's own
    documented contract ("any reader takes the LAST line for a given run_id
    as authoritative")."""
    directory = Path(runs_dir) if runs_dir is not None else DEFAULT_RUNS_DIR
    stats = LoadStats()
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.jsonl")):
        stats.files += 1
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                records[record["run_id"]] = record  # last occurrence wins

    conn = get_connection(db_path)
    try:
        for record in records.values():
            kwargs = parse_run_record(record, instrument_commit=instrument_commit)
            try:
                api.insert_run(**kwargs, conn=conn)
                stats.inserted += 1
                rows = sources_rows_for_record(record)
                if rows:
                    api.insert_sources(record["run_id"], rows, conn=conn)
            except sqlite3.IntegrityError:
                stats.skipped_existing += 1
            except sqlite3.OperationalError as exc:
                # Most likely cause in practice: a live Python process (a
                # Jupyter kernel, typically) imported collector.storage.api
                # before a schema.sql/insert_run column change and is now
                # calling the *old* insert_run against the *new* table (or
                # vice versa) -- editing the .py file on disk doesn't touch
                # an already-imported module's cached code. Re-raise with
                # that pointer instead of a bare "no such column" so it's
                # diagnosable without reading this function's source.
                raise sqlite3.OperationalError(
                    f"{exc} -- if this names a column that should exist (or shouldn't), "
                    "restart this notebook's kernel: it likely has a stale import of "
                    "collector.storage.api or schema.sql from before a recent change."
                ) from exc
        conn.commit()
    finally:
        conn.close()
    return stats


def reload_runs_db(
    runs_dir: str | Path | None = None,
    *,
    db_path: str | Path | None = None,
    instrument_commit: str,
) -> LoadStats:
    """Drop + recreate the `runs` and `sources` tables, then load_jsonl_dir(),
    in one call.

    Drop-and-recreate the table, not delete-the-*file*-and-recreate: the
    latter orphans the file handle of anything that already has the db open
    (a running `datasette`/`view_db.py` process, most concretely) -- that
    process keeps serving whatever it saw at the moment it opened the file,
    forever, because a file recreated at the same path is a different inode
    it never touches. Dropping just the table keeps the file's inode stable.
    Shared by notebooks/aggregate.ipynb and notebooks/database_ui.ipynb so
    both reload the exact same way.

    Drop (not a plain `DELETE FROM runs`) specifically because
    `init_db()`'s `CREATE TABLE IF NOT EXISTS` is a no-op against an
    already-existing table -- a `DELETE` would empty the *rows* but leave an
    existing `.sqlite3` file stuck on whatever column set it was created
    with, silently ignoring any later schema.sql edit. Dropping first means
    every reload always reflects the current schema, not just current data.

    Emptying first (rather than load_jsonl_dir's own skip-existing dedup)
    also matters because run_id has no calendar date in it (just day_index)
    -- two unrelated runs sharing the same --day-index produce colliding
    run_ids, and an incremental load would silently keep whichever one
    loaded first instead of the current content of data/01_raw/runs/*.jsonl.
    """
    conn = get_connection(db_path)
    try:
        # sources first: it has a FOREIGN KEY REFERENCES runs(run_id).
        conn.execute("DROP TABLE IF EXISTS sources")
        conn.execute("DROP TABLE IF EXISTS runs")
        conn.commit()
    finally:
        conn.close()
    init_db(db_path)
    return load_jsonl_dir(runs_dir, db_path=db_path, instrument_commit=instrument_commit)
