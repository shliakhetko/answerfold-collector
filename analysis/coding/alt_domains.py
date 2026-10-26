"""alt_domains (Post-Collection Coding Pipeline Spec): a frozen Phase A
output the spec names but defines no stage that produces it (A1-A5 don't
mention it). Treated here as a human-maintained seed
(config/alt_domains_seed.csv) copied through into the alt_domains table
(analysis/coding/storage.py), appendable from domain_redirect_snapshot
(collector/storage/schema.sql) once that table is populated for real post-hoc
redirect resolution -- a gap-fill, not a spec requirement.
"""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any

from analysis.coding import storage

from collector import paths
DEFAULT_SEED_PATH = paths.ALT_DOMAINS_SEED_CSV


def build_alt_domains(
    seed_path: Path = DEFAULT_SEED_PATH,
    *,
    domain_redirect_rows: list[dict[str, Any]] | None = None,
    conn: sqlite3.Connection,
) -> int:
    """Copies the human-maintained seed through, then appends any resolved
    redirects supplied via domain_redirect_rows (each expected to carry
    entity_id/alt_domain -- domain_redirect_snapshot itself has no
    entity_id column, so a caller must join it to the entities table
    before calling this; not done here to keep this function a pure
    read-seed/write-table step). Returns the number of rows written."""
    with seed_path.open(encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(filter(lambda line: not line.startswith("#"), f))]

    for row in domain_redirect_rows or []:
        rows.append(
            {
                "entity_id": row["entity_id"],
                "alt_domain": row["alt_domain"],
                "source_note": row.get("source_note", "domain_redirect_snapshot"),
            }
        )

    storage.clear_alt_domains(conn=conn)
    if rows:
        storage.insert_alt_domains(rows, conn=conn)
    return len(rows)
