"""Connection handling and schema initialization for the coding pipeline's
SQLite tables (A1-B3 outputs). Mirrors collector/storage/db.py's pattern:
a single schema definition, no migration sequence, CREATE ... IF NOT EXISTS
throughout so init_coding_schema() is idempotent and safe to call at the
start of every process.

Applies analysis/coding/schema.sql to the SAME database collector/storage/schema.sql
initializes (default data/01_raw/collector.sqlite3) -- new tables live alongside
runs/sources so a query can join raw collection data against derived
results. Reuses collector.storage.db.get_connection for the actual
connection (same pragmas: WAL, foreign_keys on) rather than duplicating it
-- analysis/ may depend on collector/, never the reverse (INSTRUMENT.md
section 12), and this file is never read by collector/.
"""
from __future__ import annotations

from pathlib import Path

from collector.storage.db import get_connection

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def init_coding_schema(db_path: str | Path | None = None, schema_path: str | Path | None = None) -> None:
    """Create every coding-pipeline table/view/index if it does not already
    exist. Never drops or alters existing data -- safe to call
    unconditionally, same contract as collector.storage.db.init_db."""
    schema_file = Path(schema_path) if schema_path is not None else SCHEMA_PATH
    ddl = schema_file.read_text()
    conn = get_connection(db_path)
    try:
        conn.executescript(ddl)
        # entities.frame_reason was added after some databases already had
        # an `entities` table (CREATE TABLE IF NOT EXISTS above is then a
        # no-op) -- add it in place rather than introducing a migration
        # sequence for one column.
        _ensure_column(conn, "entities", "frame_reason", "TEXT")
        # settled_class/settled_publisher_is_in_category were added after
        # the 2026-09-05 url_registry table already existed (2026-09-12
        # close-out human-review overlay) -- same situation as
        # entities.frame_reason above.
        _ensure_column(conn, "url_registry", "settled_class", "TEXT")
        _ensure_column(conn, "url_registry", "settled_publisher_is_in_category", "INTEGER")
        conn.commit()
    finally:
        conn.close()


def _ensure_column(conn, table: str, column: str, sql_type: str) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
