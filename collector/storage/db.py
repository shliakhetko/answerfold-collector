"""Connection handling and schema initialization for the collector's SQLite store.

The schema is a single definition applied at initialization (INSTRUMENT.md
section 13): there is no migration sequence. collector/storage/schema.sql uses
CREATE TABLE/INDEX IF NOT EXISTS so init_db() is idempotent and safe to call at
the start of every process (runner, snapshot job, verification script).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from collector import paths
DEFAULT_DB_PATH = paths.COLLECTOR_DB
SCHEMA_PATH = paths.SCHEMA_SQL


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a connection to the collector database with pragmas set.

    WAL journal mode: the collector writes continuously during a run window
    while other short-lived processes (verification scripts, ad-hoc inspection,
    eventually the analysis pass reading mid- or post-window) may open the same
    file concurrently. WAL lets readers proceed without blocking the writer and
    vice versa, which the default rollback journal does not allow.

    synchronous=NORMAL is the standard pairing with WAL: safe against
    application crashes (the failure mode this instrument cares about --
    section 7.1, "it does not raise") and only risks the last
    not-yet-checkpointed transaction on a full OS/power failure. Acceptable for
    a local research instrument that is not the system of record until
    exported (section 13).
    """
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def init_db(db_path: str | Path | None = None, schema_path: str | Path | None = None) -> None:
    """Create every table/index defined in schema.sql if it does not already exist.

    Never drops or alters existing data -- safe to call unconditionally.
    """
    schema_file = Path(schema_path) if schema_path is not None else SCHEMA_PATH
    ddl = schema_file.read_text()
    conn = get_connection(db_path)
    try:
        conn.executescript(ddl)
    finally:
        conn.close()
