import sqlite3

from analysis.coding.db import init_coding_schema
from collector.storage.db import init_db


def test_init_coding_schema_creates_tables_alongside_runs(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    init_db(db_path)  # runs/sources first, same order a real pipeline uses
    init_coding_schema(db_path)

    conn = sqlite3.connect(db_path)
    names = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')").fetchall()
    }
    conn.close()

    assert {"runs", "sources"} <= names  # collector's tables untouched
    assert {
        "mentions_raw", "surface_forms", "entities", "entity_surface_forms",
        "url_registry", "alt_domains", "per_run_company", "per_run_rollups",
        "eligibility_qualifying", "cell_rates", "serp_binary", "alias_hits",
    } <= names


def test_init_coding_schema_is_idempotent(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    init_db(db_path)
    init_coding_schema(db_path)
    init_coding_schema(db_path)  # must not raise
