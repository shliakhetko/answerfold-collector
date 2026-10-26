"""SQLite persistence for the coding pipeline (A1-B3 outputs). Mirrors
collector/storage/api.py's style: plain functions, no ORM, an optional
`conn` (a short-lived connection is opened/committed/closed internally when
omitted). Callers doing many writes in one pipeline run should open one
connection via collector.storage.db.get_connection() and pass it through.

Rerunnability contract: each stage's `clear_X` + bulk `insert_X` pair
replaces "rewrite the output file from scratch" -- a rerun always clears
that table before reinserting, never appends or leaves stale rows.

`list_surface_forms`/`list_entities` are deliberate adapter shims: they
reshape normalized SQL rows back into the exact flat-dict contract
b1_per_run_company.py/b2_per_run_rollups.py/b3_aggregates.py already expect
(surface_forms as a semicolon-joined string; meets_frame as
'in'/'out'/'borderline'/'unresolved'/None, in_seed_list as real bool|None)
-- see the "Adapter-shim decision" in the migration plan. Those three
modules are otherwise untouched by this migration.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from typing import Any, Iterable

from collector.storage.db import get_connection


def _with_connection(conn: sqlite3.Connection | None) -> tuple[sqlite3.Connection, bool]:
    if conn is not None:
        return conn, False
    return get_connection(), True


# --------------------------------------------------------------------------
# A1: mentions_raw
# --------------------------------------------------------------------------

def clear_mentions_raw(*, conn: sqlite3.Connection | None = None) -> None:
    c, owns = _with_connection(conn)
    try:
        c.execute("DELETE FROM mentions_raw")
        if owns:
            c.commit()
    finally:
        if owns:
            c.close()


def insert_mentions_raw(rows: Iterable[dict[str, Any]], *, conn: sqlite3.Connection | None = None) -> int:
    """Each row: {run_id, surface_form, evidence_span, is_recommended, votes
    (list[str])} -- exactly a1_answer_coding.MentionRow's shape, `votes` as
    a real list (JSON-encoded here for storage)."""
    rows = list(rows)
    c, owns = _with_connection(conn)
    try:
        c.executemany(
            """
            INSERT INTO mentions_raw (run_id, surface_form, evidence_span, is_recommended, votes)
            VALUES (:run_id, :surface_form, :evidence_span, :is_recommended, :votes)
            """,
            [
                {
                    "run_id": r["run_id"],
                    "surface_form": r["surface_form"],
                    "evidence_span": r["evidence_span"],
                    "is_recommended": int(bool(r["is_recommended"])),
                    "votes": json.dumps(list(r["votes"])),
                }
                for r in rows
            ],
        )
        if owns:
            c.commit()
        return len(rows)
    finally:
        if owns:
            c.close()


def list_mentions_raw(*, conn: sqlite3.Connection | None = None) -> list[dict[str, Any]]:
    c, owns = _with_connection(conn)
    try:
        rows = c.execute(
            "SELECT run_id, surface_form, evidence_span, is_recommended, votes FROM mentions_raw"
        ).fetchall()
        return [
            {
                "run_id": r["run_id"],
                "surface_form": r["surface_form"],
                "evidence_span": r["evidence_span"],
                "is_recommended": bool(r["is_recommended"]),
                "votes": json.loads(r["votes"]),
            }
            for r in rows
        ]
    finally:
        if owns:
            c.close()


# --------------------------------------------------------------------------
# A2: surface_forms (view, read-only)
# --------------------------------------------------------------------------

def list_surface_forms(*, conn: sqlite3.Connection | None = None) -> list[dict[str, Any]]:
    """Queries the surface_forms VIEW (analysis/coding/schema.sql) -- no
    write stage, replaces the old collapse_surface_forms()."""
    c, owns = _with_connection(conn)
    try:
        rows = c.execute("SELECT surface_form, mention_count, run_ids FROM surface_forms ORDER BY surface_form").fetchall()
        return [dict(r) for r in rows]
    finally:
        if owns:
            c.close()


# --------------------------------------------------------------------------
# A3/A4: entities, entity_surface_forms
# --------------------------------------------------------------------------

def clear_entities(*, conn: sqlite3.Connection | None = None) -> None:
    c, owns = _with_connection(conn)
    try:
        c.execute("DELETE FROM entity_surface_forms")
        c.execute("DELETE FROM entities")
        if owns:
            c.commit()
    finally:
        if owns:
            c.close()


def insert_entities(rows: Iterable[dict[str, Any]], *, conn: sqlite3.Connection | None = None) -> int:
    """Each row: {entity_id, canonical_name, homepage_url, surface_forms
    (list[str])} -- a3_entity_resolution.EntityRow's shape."""
    rows = list(rows)
    c, owns = _with_connection(conn)
    try:
        c.executemany(
            "INSERT INTO entities (entity_id, canonical_name, homepage_url) VALUES (:entity_id, :canonical_name, :homepage_url)",
            [{"entity_id": r["entity_id"], "canonical_name": r["canonical_name"], "homepage_url": r.get("homepage_url")} for r in rows],
        )
        c.executemany(
            "INSERT INTO entity_surface_forms (entity_id, surface_form) VALUES (:entity_id, :surface_form)",
            [
                {"entity_id": r["entity_id"], "surface_form": sf}
                for r in rows
                for sf in r["surface_forms"]
            ],
        )
        if owns:
            c.commit()
        return len(rows)
    finally:
        if owns:
            c.close()


def update_entity_classification(
    entity_id: str,
    *,
    meets_frame: str | None,
    in_seed_list: bool | None,
    segment: str | None,
    evidence_url: str | None,
    frame_reason: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> None:
    """A4: enrich an already-inserted (by A3) entity row in place -- not a
    full rewrite, the row already exists. meets_frame is one of
    'in'/'out'/'borderline'/'unresolved' (config/inclusion_frame.yaml's
    4-way outcome), not a bool."""
    c, owns = _with_connection(conn)
    try:
        c.execute(
            """
            UPDATE entities SET meets_frame = :meets_frame, in_seed_list = :in_seed_list,
                                 segment = :segment, evidence_url = :evidence_url,
                                 frame_reason = :frame_reason
            WHERE entity_id = :entity_id
            """,
            {
                "entity_id": entity_id,
                "meets_frame": meets_frame,
                "in_seed_list": None if in_seed_list is None else int(in_seed_list),
                "segment": segment,
                "evidence_url": evidence_url,
                "frame_reason": frame_reason,
            },
        )
        if owns:
            c.commit()
    finally:
        if owns:
            c.close()


def list_entities(*, conn: sqlite3.Connection | None = None) -> list[dict[str, Any]]:
    """Adapter shim -- see module docstring. surface_forms comes back as a
    semicolon-joined string (b1_per_run_company.surface_form_to_entity_id
    does `.split(";")`). meets_frame is 'in'/'out'/'borderline'/'unresolved'
    or None (A4 hasn't run); in_seed_list is a real bool|None."""
    c, owns = _with_connection(conn)
    try:
        rows = c.execute(
            """
            SELECT e.entity_id, e.canonical_name, e.homepage_url, e.meets_frame,
                   e.in_seed_list, e.segment, e.evidence_url, e.frame_reason,
                   GROUP_CONCAT(esf.surface_form, ';') AS surface_forms
            FROM entities e
            LEFT JOIN entity_surface_forms esf ON esf.entity_id = e.entity_id
            GROUP BY e.entity_id
            """
        ).fetchall()
        return [
            {
                "entity_id": r["entity_id"],
                "canonical_name": r["canonical_name"],
                "homepage_url": r["homepage_url"],
                "meets_frame": r["meets_frame"],
                "in_seed_list": None if r["in_seed_list"] is None else bool(r["in_seed_list"]),
                "segment": r["segment"],
                "evidence_url": r["evidence_url"],
                "frame_reason": r["frame_reason"],
                "surface_forms": r["surface_forms"] or "",
            }
            for r in rows
        ]
    finally:
        if owns:
            c.close()


# --------------------------------------------------------------------------
# A4-prime / a3b: entity_verification (analysis/coding/a3b_entity_verification.py)
# --------------------------------------------------------------------------

def clear_entity_verification(*, conn: sqlite3.Connection | None = None) -> None:
    c, owns = _with_connection(conn)
    try:
        c.execute("DELETE FROM entity_verification")
        if owns:
            c.commit()
    finally:
        if owns:
            c.close()


def insert_entity_verification(rows: Iterable[dict[str, Any]], *, conn: sqlite3.Connection | None = None) -> int:
    """Each row: {entity_id, is_real (bool|None), entity_type,
    verified_homepage_url, disambiguation_note, confidence}."""
    rows = list(rows)
    c, owns = _with_connection(conn)
    try:
        c.executemany(
            """
            INSERT OR REPLACE INTO entity_verification
                (entity_id, is_real, entity_type, verified_homepage_url, disambiguation_note, confidence)
            VALUES (:entity_id, :is_real, :entity_type, :verified_homepage_url, :disambiguation_note, :confidence)
            """,
            [
                {
                    "entity_id": r["entity_id"],
                    "is_real": None if r.get("is_real") is None else int(bool(r["is_real"])),
                    "entity_type": r.get("entity_type"),
                    "verified_homepage_url": r.get("verified_homepage_url"),
                    "disambiguation_note": r.get("disambiguation_note"),
                    "confidence": None if r.get("confidence") is None else float(r["confidence"]),
                }
                for r in rows
            ],
        )
        if owns:
            c.commit()
        return len(rows)
    finally:
        if owns:
            c.close()


def list_entity_verification(*, conn: sqlite3.Connection | None = None) -> list[dict[str, Any]]:
    c, owns = _with_connection(conn)
    try:
        rows = c.execute(
            """
            SELECT entity_id, is_real, entity_type, verified_homepage_url, disambiguation_note, confidence
            FROM entity_verification
            """
        ).fetchall()
        return [
            {
                "entity_id": r["entity_id"],
                "is_real": None if r["is_real"] is None else bool(r["is_real"]),
                "entity_type": r["entity_type"],
                "verified_homepage_url": r["verified_homepage_url"],
                "disambiguation_note": r["disambiguation_note"],
                "confidence": r["confidence"],
            }
            for r in rows
        ]
    finally:
        if owns:
            c.close()


# --------------------------------------------------------------------------
# A5: url_registry
# --------------------------------------------------------------------------

def clear_url_registry(*, conn: sqlite3.Connection | None = None) -> None:
    c, owns = _with_connection(conn)
    try:
        c.execute("DELETE FROM url_registry")
        if owns:
            c.commit()
    finally:
        if owns:
            c.close()


def insert_url_registry(rows: Iterable[dict[str, Any]], *, conn: sqlite3.Connection | None = None) -> int:
    rows = list(rows)
    c, owns = _with_connection(conn)
    try:
        c.executemany(
            """
            INSERT INTO url_registry (url_normalized, url_class, publisher_is_in_category, classifier_confidence)
            VALUES (:url_normalized, :url_class, :publisher_is_in_category, :classifier_confidence)
            """,
            [
                {
                    "url_normalized": r["url_normalized"],
                    "url_class": r.get("url_class") or None,
                    "publisher_is_in_category": None if r.get("publisher_is_in_category") in (None, "") else int(bool(r["publisher_is_in_category"])),
                    "classifier_confidence": None if r.get("classifier_confidence") in (None, "") else float(r["classifier_confidence"]),
                }
                for r in rows
            ],
        )
        if owns:
            c.commit()
        return len(rows)
    finally:
        if owns:
            c.close()


def list_url_registry(*, conn: sqlite3.Connection | None = None) -> list[dict[str, Any]]:
    c, owns = _with_connection(conn)
    try:
        rows = c.execute(
            "SELECT url_normalized, url_class, publisher_is_in_category, classifier_confidence, "
            "settled_class, settled_publisher_is_in_category FROM url_registry"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        if owns:
            c.close()


def update_url_registry_settled(rows: Iterable[dict[str, Any]], *, conn: sqlite3.Connection | None = None) -> int:
    """2026-09-12 close-out overlay: set settled_class/
    settled_publisher_is_in_category on already-populated url_registry rows.
    Never called by A5 itself -- see scripts/coding/close_out_a5_url_classification.py.
    Each row must carry url_normalized, settled_class, settled_publisher_is_in_category."""
    rows = list(rows)
    c, owns = _with_connection(conn)
    try:
        c.executemany(
            """
            UPDATE url_registry
            SET settled_class = :settled_class,
                settled_publisher_is_in_category = :settled_publisher_is_in_category
            WHERE url_normalized = :url_normalized
            """,
            [
                {
                    "url_normalized": r["url_normalized"],
                    "settled_class": r.get("settled_class") or None,
                    "settled_publisher_is_in_category": None
                    if r.get("settled_publisher_is_in_category") in (None, "")
                    else int(bool(r["settled_publisher_is_in_category"])),
                }
                for r in rows
            ],
        )
        if owns:
            c.commit()
        return len(rows)
    finally:
        if owns:
            c.close()


# --------------------------------------------------------------------------
# cited_url_resolution (Task 5 overlay -- see analysis/coding/cited_url_resolution.py)
# --------------------------------------------------------------------------

def clear_cited_url_resolution(*, conn: sqlite3.Connection | None = None) -> None:
    c, owns = _with_connection(conn)
    try:
        c.execute("DELETE FROM cited_url_resolution")
        if owns:
            c.commit()
    finally:
        if owns:
            c.close()


def insert_cited_url_resolution(rows: Iterable[dict[str, Any]], *, conn: sqlite3.Connection | None = None) -> int:
    """Each row: {cited_url, final_url, final_domain, status_bucket} --
    cited_url_resolution.ResolvedCitation's persisted shape."""
    rows = list(rows)
    c, owns = _with_connection(conn)
    try:
        c.executemany(
            """
            INSERT OR REPLACE INTO cited_url_resolution (cited_url, final_url, final_domain, status_bucket)
            VALUES (:cited_url, :final_url, :final_domain, :status_bucket)
            """,
            [
                {
                    "cited_url": r["cited_url"],
                    "final_url": r.get("final_url"),
                    "final_domain": r.get("final_domain"),
                    "status_bucket": r.get("status_bucket"),
                }
                for r in rows
            ],
        )
        if owns:
            c.commit()
        return len(rows)
    finally:
        if owns:
            c.close()


def list_cited_url_resolution(*, conn: sqlite3.Connection | None = None) -> list[dict[str, Any]]:
    c, owns = _with_connection(conn)
    try:
        rows = c.execute(
            "SELECT cited_url, final_url, final_domain, status_bucket FROM cited_url_resolution"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        if owns:
            c.close()


# --------------------------------------------------------------------------
# alt_domains
# --------------------------------------------------------------------------

def clear_alt_domains(*, conn: sqlite3.Connection | None = None) -> None:
    c, owns = _with_connection(conn)
    try:
        c.execute("DELETE FROM alt_domains")
        if owns:
            c.commit()
    finally:
        if owns:
            c.close()


def insert_alt_domains(rows: Iterable[dict[str, Any]], *, conn: sqlite3.Connection | None = None) -> int:
    rows = list(rows)
    c, owns = _with_connection(conn)
    try:
        c.executemany(
            "INSERT INTO alt_domains (entity_id, alt_domain, source_note) VALUES (:entity_id, :alt_domain, :source_note)",
            [{"entity_id": r["entity_id"], "alt_domain": r["alt_domain"], "source_note": r.get("source_note")} for r in rows],
        )
        if owns:
            c.commit()
        return len(rows)
    finally:
        if owns:
            c.close()


def list_alt_domains(*, conn: sqlite3.Connection | None = None) -> list[dict[str, Any]]:
    c, owns = _with_connection(conn)
    try:
        rows = c.execute("SELECT entity_id, alt_domain, source_note FROM alt_domains").fetchall()
        return [dict(r) for r in rows]
    finally:
        if owns:
            c.close()


# --------------------------------------------------------------------------
# B1: per_run_company
# --------------------------------------------------------------------------

def clear_per_run_company(*, conn: sqlite3.Connection | None = None) -> None:
    c, owns = _with_connection(conn)
    try:
        c.execute("DELETE FROM per_run_company")
        if owns:
            c.commit()
    finally:
        if owns:
            c.close()


def insert_per_run_company(rows: Iterable[Any], *, conn: sqlite3.Connection | None = None) -> int:
    """rows: b1_per_run_company.PerRunCompanyRow instances (or equivalent dicts)."""
    dict_rows = [r if isinstance(r, dict) else asdict(r) for r in rows]
    c, owns = _with_connection(conn)
    try:
        c.executemany(
            """
            INSERT INTO per_run_company (
                run_id, entity_id, mentioned, recommended, own_domain_cited,
                linked, mention_without_own_citation, first_mention_position
            ) VALUES (
                :run_id, :entity_id, :mentioned, :recommended, :own_domain_cited,
                :linked, :mention_without_own_citation, :first_mention_position
            )
            """,
            [
                {
                    "run_id": r["run_id"],
                    "entity_id": r["entity_id"],
                    "mentioned": int(r["mentioned"]),
                    "recommended": int(r["recommended"]),
                    "own_domain_cited": int(r["own_domain_cited"]),
                    "linked": int(r["linked"]),
                    "mention_without_own_citation": int(r["mention_without_own_citation"]),
                    "first_mention_position": r["first_mention_position"],
                }
                for r in dict_rows
            ],
        )
        if owns:
            c.commit()
        return len(dict_rows)
    finally:
        if owns:
            c.close()


# --------------------------------------------------------------------------
# B2: per_run_rollups
# --------------------------------------------------------------------------

def clear_per_run_rollups(*, conn: sqlite3.Connection | None = None) -> None:
    c, owns = _with_connection(conn)
    try:
        c.execute("DELETE FROM per_run_rollups")
        if owns:
            c.commit()
    finally:
        if owns:
            c.close()


def insert_per_run_rollups(rows: Iterable[Any], *, conn: sqlite3.Connection | None = None) -> int:
    """rows: b2_per_run_rollups.PerRunRollupRow instances (or equivalent dicts)."""
    dict_rows = [r if isinstance(r, dict) else asdict(r) for r in rows]
    c, owns = _with_connection(conn)
    try:
        c.executemany(
            """
            INSERT INTO per_run_rollups (run_id, n_companies_mentioned, n_sources, vendor_source_share, n_out_of_frame_brands)
            VALUES (:run_id, :n_companies_mentioned, :n_sources, :vendor_source_share, :n_out_of_frame_brands)
            """,
            dict_rows,
        )
        if owns:
            c.commit()
        return len(dict_rows)
    finally:
        if owns:
            c.close()


# --------------------------------------------------------------------------
# B3: eligibility_qualifying, cell_rates
# --------------------------------------------------------------------------

def clear_eligibility_qualifying(*, conn: sqlite3.Connection | None = None) -> None:
    c, owns = _with_connection(conn)
    try:
        c.execute("DELETE FROM eligibility_qualifying")
        if owns:
            c.commit()
    finally:
        if owns:
            c.close()


def insert_eligibility_qualifying(qualifying_by_engine: dict[str, Iterable[str]], *, conn: sqlite3.Connection | None = None) -> int:
    """qualifying_by_engine: b3_aggregates.EligibilityReport.qualifying_by_engine
    shape -- {engine: {entity_id, ...}}."""
    rows = [{"engine": engine, "entity_id": entity_id} for engine, entity_ids in qualifying_by_engine.items() for entity_id in entity_ids]
    c, owns = _with_connection(conn)
    try:
        c.executemany(
            "INSERT INTO eligibility_qualifying (engine, entity_id) VALUES (:engine, :entity_id)", rows
        )
        if owns:
            c.commit()
        return len(rows)
    finally:
        if owns:
            c.close()


def clear_cell_rates(*, conn: sqlite3.Connection | None = None) -> None:
    c, owns = _with_connection(conn)
    try:
        c.execute("DELETE FROM cell_rates")
        if owns:
            c.commit()
    finally:
        if owns:
            c.close()


def insert_cell_rates(rows: Iterable[Any], *, conn: sqlite3.Connection | None = None) -> int:
    """rows: b3_aggregates.CellRates instances (or equivalent dicts)."""
    dict_rows = [r if isinstance(r, dict) else asdict(r) for r in rows]
    c, owns = _with_connection(conn)
    try:
        c.executemany(
            """
            INSERT INTO cell_rates (entity_id, intent_id, engine, n_runs, mention_rate, recommendation_rate)
            VALUES (:entity_id, :intent_id, :engine, :n_runs, :mention_rate, :recommendation_rate)
            """,
            dict_rows,
        )
        if owns:
            c.commit()
        return len(dict_rows)
    finally:
        if owns:
            c.close()


# --------------------------------------------------------------------------
# B3b: serp_binary
# --------------------------------------------------------------------------

def clear_serp_binary(*, conn: sqlite3.Connection | None = None) -> None:
    c, owns = _with_connection(conn)
    try:
        c.execute("DELETE FROM serp_binary")
        if owns:
            c.commit()
    finally:
        if owns:
            c.close()


def insert_serp_binary(rows: Iterable[dict[str, Any]], *, conn: sqlite3.Connection | None = None) -> int:
    """rows: b3b_serp_derivation.derive_serp_binary's output shape."""
    rows = list(rows)
    c, owns = _with_connection(conn)
    try:
        c.executemany(
            "INSERT INTO serp_binary (entity_id, intent_id, in_google_top10) VALUES (:entity_id, :intent_id, :in_google_top10)",
            [{"entity_id": r["entity_id"], "intent_id": r["intent_id"], "in_google_top10": int(bool(r["in_google_top10"]))} for r in rows],
        )
        if owns:
            c.commit()
        return len(rows)
    finally:
        if owns:
            c.close()


# --------------------------------------------------------------------------
# Alias-regex validation track: alias_hits
# --------------------------------------------------------------------------

def clear_alias_hits(*, conn: sqlite3.Connection | None = None) -> None:
    c, owns = _with_connection(conn)
    try:
        c.execute("DELETE FROM alias_hits")
        if owns:
            c.commit()
    finally:
        if owns:
            c.close()


def insert_alias_hits(rows: Iterable[dict[str, Any]], *, conn: sqlite3.Connection | None = None) -> int:
    """rows: alias_regex_validation.run_alias_regex's output shape --
    {run_id, canonical_name}."""
    rows = list(rows)
    c, owns = _with_connection(conn)
    try:
        c.executemany(
            "INSERT INTO alias_hits (run_id, canonical_name) VALUES (:run_id, :canonical_name)", rows
        )
        if owns:
            c.commit()
        return len(rows)
    finally:
        if owns:
            c.close()
