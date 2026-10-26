from collector.storage.db import get_connection
from analysis.coding.db import init_coding_schema
from collector.storage.db import init_db
from analysis.coding import storage as coding_storage

from analysis.aux import roster


def _seed_db(tmp_path):
    db = tmp_path / "coding.sqlite3"
    init_db(db)
    init_coding_schema(db)
    conn = get_connection(db)
    coding_storage.insert_entities(
        [
            {"entity_id": "goodcall", "canonical_name": "Goodcall", "homepage_url": "https://www.goodcall.com", "surface_forms": ["Goodcall"]},
            {"entity_id": "rosie", "canonical_name": "Rosie", "homepage_url": None, "surface_forms": ["Rosie"]},
            {"entity_id": "verifonly", "canonical_name": "VerifOnly", "homepage_url": None, "surface_forms": ["VerifOnly"]},
            {"entity_id": "outof", "canonical_name": "OutOfFrame", "homepage_url": "https://of.com", "surface_forms": ["OutOfFrame"]},
        ],
        conn=conn,
    )
    coding_storage.update_entity_classification("goodcall", meets_frame="in", in_seed_list=True, segment="done-for-you", evidence_url=None, conn=conn)
    coding_storage.update_entity_classification("rosie", meets_frame="in", in_seed_list=False, segment="no-code", evidence_url=None, conn=conn)
    coding_storage.update_entity_classification("verifonly", meets_frame="in", in_seed_list=False, segment="done-for-you", evidence_url=None, conn=conn)
    coding_storage.update_entity_classification("outof", meets_frame="out", in_seed_list=False, segment=None, evidence_url=None, conn=conn)
    coding_storage.insert_alt_domains([{"entity_id": "goodcall", "alt_domain": "goodcall.io", "source_note": "seed"}], conn=conn)
    coding_storage.insert_entity_verification(
        [
            {"entity_id": "verifonly", "is_real": 1, "entity_type": "vendor",
             "verified_homepage_url": "https://verifonly.example", "disambiguation_note": None, "confidence": 0.9},
        ],
        conn=conn,
    )
    conn.commit()
    conn.close()
    return db


def test_load_from_coding_db_in_frame_only(tmp_path):
    db = _seed_db(tmp_path)
    r = roster.load_from_coding_db(db, in_frame_only=True)
    ids = {e.entity_id for e in r}
    assert ids == {"goodcall", "rosie", "verifonly"}  # outof excluded


def test_homepage_falls_back_to_verified_url(tmp_path):
    db = _seed_db(tmp_path)
    by_id = {e.entity_id: e for e in roster.load_from_coding_db(db)}
    # no entities.homepage_url, but entity_verification has one
    assert by_id["verifonly"].primary_domain == "verifonly.example"
    assert by_id["verifonly"].needs_manual is False
    # no homepage anywhere -> still flagged
    assert by_id["rosie"].primary_domain is None
    assert by_id["rosie"].needs_manual is True


def test_primary_and_alt_domains(tmp_path):
    db = _seed_db(tmp_path)
    by_id = {e.entity_id: e for e in roster.load_from_coding_db(db)}
    assert by_id["goodcall"].primary_domain == "goodcall.com"
    assert "goodcall.io" in by_id["goodcall"].alt_domains
    assert by_id["goodcall"].domains == ["goodcall.com", "goodcall.io"]


def test_unresolved_entity_flagged_needs_manual(tmp_path):
    db = _seed_db(tmp_path)
    by_id = {e.entity_id: e for e in roster.load_from_coding_db(db)}
    assert by_id["rosie"].primary_domain is None
    assert by_id["rosie"].needs_manual is True


def test_csv_round_trip_preserves_fields(tmp_path):
    rows = [
        roster.RosterEntity("goodcall", "Goodcall", "goodcall.com", ["goodcall.io"], "both", False, ""),
        roster.RosterEntity("rosie", "Rosie", None, [], "discovered", True, "resolve"),
    ]
    path = tmp_path / "roster.csv"
    roster.write_csv(rows, path)
    back = {e.entity_id: e for e in roster.load_from_csv(path)}
    assert back["goodcall"].alt_domains == ["goodcall.io"]
    assert back["rosie"].needs_manual is True
    assert path.read_text().startswith("# PROVISIONAL")
