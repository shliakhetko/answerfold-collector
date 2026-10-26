"""In-process micro API for the collector's SQLite store.

Plain functions, no ORM, no HTTP layer (deliberate choice over a local
service -- this is imported directly by runner/engines/snapshot code). Every
function accepts an optional `conn`; when omitted, a short-lived connection is
opened and committed/closed internally. Callers doing many writes in one
process should open one connection via db.get_connection() and pass it through.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Iterable, Mapping

from collector.storage.db import get_connection


def _dump(value: Any) -> str | None:
    return None if value is None else json.dumps(value)


def _load(value: str | None) -> Any:
    return None if value is None else json.loads(value)


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    return None if row is None else dict(row)


def _with_connection(conn: sqlite3.Connection | None) -> tuple[sqlite3.Connection, bool]:
    if conn is not None:
        return conn, False
    return get_connection(), True


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------

def insert_run(
    *,
    run_id: str,
    instrument_commit: str,
    timestamp: str,
    engine: str,
    intent_id: int,
    paraphrase_id: int,
    prompt_text: str,
    rep_index: int,
    day_index: int,
    model_requested: str | None = None,
    model_returned: str | None = None,
    raw_response_text: str | None = None,
    search_fired: bool | None = None,
    fan_out_queries: str | None = None,
    latency_ms: int | None = None,
    finish_reason: str | None = None,
    error: str | None = None,
    tokens_input: int | None = None,
    tokens_output: int | None = None,
    tokens_total: int | None = None,
    reasoning_tokens: int | None = None,
    tool_tokens: int | None = None,
    cached_tokens: int | None = None,
    cache_write_tokens: int | None = None,
    cached_content_tokens: int | None = None,
    cache_creation_tokens: int | None = None,
    cache_read_tokens: int | None = None,
    web_search_requests: int | None = None,
    web_fetch_requests: int | None = None,
    scraper_vendor: str | None = None,
    scraper_endpoint: str | None = None,
    raw_response_envelope: str | None = None,
    is_manual: bool = False,
    screenshot_path: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> str:
    """Insert one run row. Returns run_id. Raises sqlite3.IntegrityError on duplicate id."""
    c, owns = _with_connection(conn)
    try:
        c.execute(
            """
            INSERT INTO runs (
                run_id, instrument_commit, timestamp, engine,
                model_requested, model_returned, intent_id, paraphrase_id,
                prompt_text, rep_index, day_index, raw_response_text,
                search_fired, fan_out_queries, latency_ms, finish_reason,
                error, tokens_input, tokens_output, tokens_total,
                reasoning_tokens, tool_tokens, cached_tokens, cache_write_tokens,
                cached_content_tokens, cache_creation_tokens, cache_read_tokens,
                web_search_requests, web_fetch_requests,
                scraper_vendor, scraper_endpoint,
                raw_response_envelope, is_manual, screenshot_path
            ) VALUES (
                :run_id, :instrument_commit, :timestamp, :engine,
                :model_requested, :model_returned, :intent_id, :paraphrase_id,
                :prompt_text, :rep_index, :day_index, :raw_response_text,
                :search_fired, :fan_out_queries, :latency_ms, :finish_reason,
                :error, :tokens_input, :tokens_output, :tokens_total,
                :reasoning_tokens, :tool_tokens, :cached_tokens, :cache_write_tokens,
                :cached_content_tokens, :cache_creation_tokens, :cache_read_tokens,
                :web_search_requests, :web_fetch_requests,
                :scraper_vendor, :scraper_endpoint,
                :raw_response_envelope, :is_manual, :screenshot_path
            )
            """,
            {
                "run_id": run_id,
                "instrument_commit": instrument_commit,
                "timestamp": timestamp,
                "engine": engine,
                "model_requested": model_requested,
                "model_returned": model_returned,
                "intent_id": intent_id,
                "paraphrase_id": paraphrase_id,
                "prompt_text": prompt_text,
                "rep_index": rep_index,
                "day_index": day_index,
                "raw_response_text": raw_response_text,
                "search_fired": None if search_fired is None else int(search_fired),
                "fan_out_queries": fan_out_queries,
                "latency_ms": latency_ms,
                "finish_reason": finish_reason,
                "error": error,
                "tokens_input": tokens_input,
                "tokens_output": tokens_output,
                "tokens_total": tokens_total,
                "reasoning_tokens": reasoning_tokens,
                "tool_tokens": tool_tokens,
                "cached_tokens": cached_tokens,
                "cache_write_tokens": cache_write_tokens,
                "cached_content_tokens": cached_content_tokens,
                "cache_creation_tokens": cache_creation_tokens,
                "cache_read_tokens": cache_read_tokens,
                "web_search_requests": web_search_requests,
                "web_fetch_requests": web_fetch_requests,
                "scraper_vendor": scraper_vendor,
                "scraper_endpoint": scraper_endpoint,
                "raw_response_envelope": raw_response_envelope,
                "is_manual": int(is_manual),
                "screenshot_path": screenshot_path,
            },
        )
        if owns:
            c.commit()
        return run_id
    finally:
        if owns:
            c.close()


def get_run(run_id: str, *, conn: sqlite3.Connection | None = None) -> dict | None:
    c, owns = _with_connection(conn)
    try:
        row = c.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return _row_to_dict(row)
    finally:
        if owns:
            c.close()


def list_runs(
    *,
    engine: str | None = None,
    intent_id: int | None = None,
    paraphrase_id: int | None = None,
    day_index: int | None = None,
    rep_index: int | None = None,
    is_manual: bool | None = None,
    has_error: bool | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[dict]:
    clauses: list[str] = []
    params: list[Any] = []
    for column, value in (
        ("engine", engine), ("intent_id", intent_id), ("paraphrase_id", paraphrase_id),
        ("day_index", day_index), ("rep_index", rep_index),
    ):
        if value is not None:
            clauses.append(f"{column} = ?")
            params.append(value)
    if is_manual is not None:
        clauses.append("is_manual = ?")
        params.append(int(is_manual))
    if has_error is not None:
        clauses.append("error IS NOT NULL" if has_error else "error IS NULL")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    c, owns = _with_connection(conn)
    try:
        rows = c.execute(f"SELECT * FROM runs {where} ORDER BY timestamp", params).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        if owns:
            c.close()


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------

def insert_sources(
    run_id: str,
    sources: Iterable[Mapping[str, Any]],
    *,
    conn: sqlite3.Connection | None = None,
) -> int:
    """Insert many source rows for one run in a single transaction.

    Each item: {"position": int, "url_raw": str, "domain": str | None,
    "title": str | None, "extra": dict | None}. Returns rows inserted.
    """
    rows = list(sources)
    c, owns = _with_connection(conn)
    try:
        c.executemany(
            """
            INSERT INTO sources (run_id, position, url_raw, domain, title, extra)
            VALUES (:run_id, :position, :url_raw, :domain, :title, :extra)
            """,
            [
                {
                    "run_id": run_id,
                    "position": s["position"],
                    "url_raw": s["url_raw"],
                    "domain": s.get("domain"),
                    "title": s.get("title"),
                    "extra": _dump(s.get("extra")),
                }
                for s in rows
            ],
        )
        if owns:
            c.commit()
        return len(rows)
    finally:
        if owns:
            c.close()


def get_sources(run_id: str, *, conn: sqlite3.Connection | None = None) -> list[dict]:
    c, owns = _with_connection(conn)
    try:
        rows = c.execute(
            "SELECT * FROM sources WHERE run_id = ? ORDER BY position", (run_id,)
        ).fetchall()
        results = [_row_to_dict(r) for r in rows]
        for r in results:
            r["extra"] = _load(r["extra"])
        return results
    finally:
        if owns:
            c.close()


# --------------------------------------------------------------------------
# Once-per-window / daily snapshots
# --------------------------------------------------------------------------

def record_domain_redirect(
    *,
    canonical_domain: str,
    checked_at: str,
    resolved_url: str | None = None,
    status_code: int | None = None,
    day_index: int | None = None,
    error: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> int:
    """Append one daily redirect-resolution row. Always an insert, never an
    update -- a redirect state that existed cannot be reconstructed later
    (INSTRUMENT.md 8.4/8.5)."""
    c, owns = _with_connection(conn)
    try:
        cur = c.execute(
            """
            INSERT INTO domain_redirect_snapshot
                (canonical_domain, day_index, checked_at, resolved_url, status_code, error)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (canonical_domain, day_index, checked_at, resolved_url, status_code, error),
        )
        if owns:
            c.commit()
        return cur.lastrowid
    finally:
        if owns:
            c.close()


def list_domain_redirects(
    *,
    canonical_domain: str | None = None,
    day_index: int | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[dict]:
    clauses, params = [], []
    if canonical_domain is not None:
        clauses.append("canonical_domain = ?")
        params.append(canonical_domain)
    if day_index is not None:
        clauses.append("day_index = ?")
        params.append(day_index)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    c, owns = _with_connection(conn)
    try:
        rows = c.execute(
            f"SELECT * FROM domain_redirect_snapshot {where} ORDER BY checked_at", params
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        if owns:
            c.close()


def record_ranking_snapshot(
    *,
    entity_id: str,
    intent_id: int,
    ranking_system: str,
    metric: str,
    value: str,
    checked_at: str,
    raw: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> int:
    """Append one third-party ranking observation. `ranking_system` names the
    external system (e.g. 'google', 'bing'); `metric` names what was captured
    (e.g. 'top10', 'rank'); `value` is stored as text so it can be a boolean
    literal, an integer rank, or a small JSON blob -- the instrument does not fix
    which ranking systems or metrics exist."""
    c, owns = _with_connection(conn)
    try:
        cur = c.execute(
            """
            INSERT INTO ranking_snapshot
                (entity_id, intent_id, ranking_system, metric, value, checked_at, raw)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (entity_id, intent_id, ranking_system, metric, value, checked_at, raw),
        )
        if owns:
            c.commit()
        return cur.lastrowid
    finally:
        if owns:
            c.close()


def list_ranking_snapshots(
    *,
    entity_id: str | None = None,
    intent_id: int | None = None,
    ranking_system: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[dict]:
    clauses, params = [], []
    for column, value in (("entity_id", entity_id), ("intent_id", intent_id), ("ranking_system", ranking_system)):
        if value is not None:
            clauses.append(f"{column} = ?")
            params.append(value)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    c, owns = _with_connection(conn)
    try:
        rows = c.execute(
            f"SELECT * FROM ranking_snapshot {where} ORDER BY checked_at", params
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        if owns:
            c.close()


def record_robots_check(
    *,
    domain: str,
    checked_at: str,
    blocked: bool | None = None,
    raw_notes: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> int:
    """Append one crawler-access-policy check. blocked=None means unverified,
    a distinct state from True/False (INSTRUMENT.md 8.5)."""
    c, owns = _with_connection(conn)
    try:
        cur = c.execute(
            """
            INSERT INTO robots_ai_crawler_block (domain, blocked, checked_at, raw_notes)
            VALUES (?, ?, ?, ?)
            """,
            (domain, None if blocked is None else int(blocked), checked_at, raw_notes),
        )
        if owns:
            c.commit()
        return cur.lastrowid
    finally:
        if owns:
            c.close()


def list_robots_checks(*, domain: str | None = None, conn: sqlite3.Connection | None = None) -> list[dict]:
    where = "WHERE domain = ?" if domain is not None else ""
    params = (domain,) if domain is not None else ()
    c, owns = _with_connection(conn)
    try:
        rows = c.execute(
            f"SELECT * FROM robots_ai_crawler_block {where} ORDER BY checked_at", params
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        if owns:
            c.close()
