from analysis.coding import storage
from analysis.coding.alt_domains import build_alt_domains
from analysis.coding.db import init_coding_schema
from collector.storage.db import get_connection, init_db


def _fresh_conn(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    init_db(db_path)
    init_coding_schema(db_path)
    return get_connection(db_path)


def test_copies_seed_rows_through(tmp_path):
    seed_path = tmp_path / "seed.csv"
    seed_path.write_text("# comment\nentity_id,alt_domain,source_note\ngoodcall,goodcall.io,manual\n")
    conn = _fresh_conn(tmp_path)

    n = build_alt_domains(seed_path, conn=conn)

    assert n == 1
    assert storage.list_alt_domains(conn=conn) == [{"entity_id": "goodcall", "alt_domain": "goodcall.io", "source_note": "manual"}]


def test_appends_resolved_redirects(tmp_path):
    seed_path = tmp_path / "seed.csv"
    seed_path.write_text("entity_id,alt_domain,source_note\ngoodcall,goodcall.io,manual\n")
    conn = _fresh_conn(tmp_path)

    n = build_alt_domains(seed_path, domain_redirect_rows=[{"entity_id": "goodcall", "alt_domain": "goodcall.app"}], conn=conn)

    assert n == 2
    rows = storage.list_alt_domains(conn=conn)
    assert rows[1] == {"entity_id": "goodcall", "alt_domain": "goodcall.app", "source_note": "domain_redirect_snapshot"}


def test_empty_seed_writes_nothing(tmp_path):
    seed_path = tmp_path / "seed.csv"
    seed_path.write_text("entity_id,alt_domain,source_note\n")
    conn = _fresh_conn(tmp_path)

    n = build_alt_domains(seed_path, conn=conn)

    assert n == 0
    assert storage.list_alt_domains(conn=conn) == []


def test_rerun_clears_previous_rows_not_append(tmp_path):
    seed_path = tmp_path / "seed.csv"
    seed_path.write_text("entity_id,alt_domain,source_note\ngoodcall,goodcall.io,manual\n")
    conn = _fresh_conn(tmp_path)

    build_alt_domains(seed_path, conn=conn)
    build_alt_domains(seed_path, conn=conn)

    assert len(storage.list_alt_domains(conn=conn)) == 1  # not duplicated on rerun
