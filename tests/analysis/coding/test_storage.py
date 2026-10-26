import pytest

from analysis.coding import storage
from analysis.coding.db import init_coding_schema
from collector.storage import api
from collector.storage.db import get_connection, init_db


@pytest.fixture
def conn(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    init_db(db_path)
    init_coding_schema(db_path)
    c = get_connection(db_path)
    api.insert_run(
        run_id="run-1", instrument_commit="deadbeef", timestamp="2026-08-24T10:00:00+00:00",
        engine="anthropic_claude", intent_id=1, paraphrase_id=1, prompt_text="best AI note taker",
        rep_index=1, day_index=1, conn=c,
    )
    api.insert_run(
        run_id="run-2", instrument_commit="deadbeef", timestamp="2026-08-24T10:00:00+00:00",
        engine="anthropic_claude", intent_id=1, paraphrase_id=1, prompt_text="best AI note taker",
        rep_index=2, day_index=1, conn=c,
    )
    c.commit()
    yield c
    c.close()


# --------------------------------------------------------------------------
# mentions_raw / surface_forms
# --------------------------------------------------------------------------

def test_mentions_raw_roundtrip_and_clear(conn):
    storage.insert_mentions_raw(
        [{"run_id": "run-1", "surface_form": "Otter.ai", "evidence_span": "Otter.ai is great.", "is_recommended": True, "votes": ["anthropic", "openai"]}],
        conn=conn,
    )
    rows = storage.list_mentions_raw(conn=conn)
    assert rows == [
        {"run_id": "run-1", "surface_form": "Otter.ai", "evidence_span": "Otter.ai is great.", "is_recommended": True, "votes": ["anthropic", "openai"]}
    ]

    storage.clear_mentions_raw(conn=conn)
    assert storage.list_mentions_raw(conn=conn) == []


def test_surface_forms_view_groups_by_exact_surface_form(conn):
    storage.insert_mentions_raw(
        [
            {"run_id": "run-1", "surface_form": "Otter.ai", "evidence_span": "a", "is_recommended": True, "votes": ["a"]},
            {"run_id": "run-2", "surface_form": "Otter.ai", "evidence_span": "b", "is_recommended": True, "votes": ["a"]},
            {"run_id": "run-1", "surface_form": "Fathom", "evidence_span": "c", "is_recommended": False, "votes": ["a"]},
        ],
        conn=conn,
    )
    rows = {r["surface_form"]: r for r in storage.list_surface_forms(conn=conn)}
    assert rows["Otter.ai"]["mention_count"] == 2
    assert set(rows["Otter.ai"]["run_ids"].split(";")) == {"run-1", "run-2"}
    assert rows["Fathom"]["mention_count"] == 1


# --------------------------------------------------------------------------
# entities / entity_surface_forms
# --------------------------------------------------------------------------

def test_entities_roundtrip_with_surface_forms_semicolon_joined(conn):
    storage.insert_entities(
        [{"entity_id": "otter", "canonical_name": "Otter.ai", "homepage_url": "https://otter.ai/", "surface_forms": ["Otter.ai", "Otter AI"]}],
        conn=conn,
    )
    rows = storage.list_entities(conn=conn)
    assert len(rows) == 1
    assert rows[0]["entity_id"] == "otter"
    assert set(rows[0]["surface_forms"].split(";")) == {"Otter.ai", "Otter AI"}
    assert rows[0]["meets_frame"] is None  # A4 hasn't run yet


def test_update_entity_classification_enriches_existing_row(conn):
    storage.insert_entities(
        [{"entity_id": "otter", "canonical_name": "Otter.ai", "homepage_url": "https://otter.ai/", "surface_forms": ["Otter.ai"]}],
        conn=conn,
    )
    storage.update_entity_classification(
        "otter", meets_frame="in", in_seed_list=False, segment="done-for-you", evidence_url="https://otter.ai/", conn=conn
    )
    rows = storage.list_entities(conn=conn)
    assert rows[0]["meets_frame"] == "in"
    assert rows[0]["in_seed_list"] is False
    assert rows[0]["segment"] == "done-for-you"


def test_clear_entities_clears_junction_table_too(conn):
    storage.insert_entities(
        [{"entity_id": "otter", "canonical_name": "Otter.ai", "homepage_url": None, "surface_forms": ["Otter.ai"]}],
        conn=conn,
    )
    storage.clear_entities(conn=conn)
    assert storage.list_entities(conn=conn) == []
    # re-insert must not hit a leftover PK/junction-row conflict
    storage.insert_entities(
        [{"entity_id": "otter", "canonical_name": "Otter.ai", "homepage_url": None, "surface_forms": ["Otter.ai"]}],
        conn=conn,
    )
    assert len(storage.list_entities(conn=conn)) == 1


# --------------------------------------------------------------------------
# url_registry
# --------------------------------------------------------------------------

def test_url_registry_roundtrip_and_clear(conn):
    storage.insert_url_registry(
        [{"url_normalized": "https://otter.ai", "url_class": "vendor page", "publisher_is_in_category": True, "classifier_confidence": 0.9}],
        conn=conn,
    )
    rows = storage.list_url_registry(conn=conn)
    assert rows == [
        {
            "url_normalized": "https://otter.ai",
            "url_class": "vendor page",
            "publisher_is_in_category": 1,
            "classifier_confidence": 0.9,
            "settled_class": None,
            "settled_publisher_is_in_category": None,
        }
    ]

    storage.clear_url_registry(conn=conn)
    assert storage.list_url_registry(conn=conn) == []


def test_update_url_registry_settled_overlays_existing_rows(conn):
    storage.insert_url_registry(
        [{"url_normalized": "https://otter.ai", "url_class": "vendor page", "publisher_is_in_category": True, "classifier_confidence": 0.45}],
        conn=conn,
    )
    storage.update_url_registry_settled(
        [{"url_normalized": "https://otter.ai", "settled_class": "forum", "settled_publisher_is_in_category": False}],
        conn=conn,
    )
    rows = storage.list_url_registry(conn=conn)
    assert rows[0]["settled_class"] == "forum"
    assert rows[0]["settled_publisher_is_in_category"] == 0
    # original model columns untouched
    assert rows[0]["url_class"] == "vendor page"


# --------------------------------------------------------------------------
# alt_domains
# --------------------------------------------------------------------------

def test_alt_domains_roundtrip_and_clear(conn):
    storage.insert_alt_domains([{"entity_id": "otter", "alt_domain": "otter.io", "source_note": "seed"}], conn=conn)
    rows = storage.list_alt_domains(conn=conn)
    assert rows == [{"entity_id": "otter", "alt_domain": "otter.io", "source_note": "seed"}]

    storage.clear_alt_domains(conn=conn)
    assert storage.list_alt_domains(conn=conn) == []


# --------------------------------------------------------------------------
# B1/B2/B3 tables -- write-only from storage's perspective (B-stage math
# reads its inputs as plain in-memory rows, never re-reads these back)
# --------------------------------------------------------------------------

def test_per_run_company_insert_and_clear(conn):
    from analysis.coding.b1_per_run_company import PerRunCompanyRow

    row = PerRunCompanyRow(
        run_id="run-1", entity_id="otter", mentioned=True, recommended=True,
        own_domain_cited=False, linked=False, mention_without_own_citation=True, first_mention_position=5,
    )
    storage.insert_entities([{"entity_id": "otter", "canonical_name": "Otter.ai", "homepage_url": None, "surface_forms": ["Otter.ai"]}], conn=conn)
    n = storage.insert_per_run_company([row], conn=conn)
    assert n == 1
    stored = conn.execute("SELECT * FROM per_run_company").fetchall()
    assert len(stored) == 1

    storage.clear_per_run_company(conn=conn)
    assert conn.execute("SELECT COUNT(*) FROM per_run_company").fetchone()[0] == 0


def test_per_run_rollups_insert_and_clear(conn):
    from analysis.coding.b2_per_run_rollups import PerRunRollupRow

    row = PerRunRollupRow(run_id="run-1", n_companies_mentioned=1, n_sources=2, vendor_source_share=0.5, n_out_of_frame_brands=0)
    storage.insert_per_run_rollups([row], conn=conn)
    assert conn.execute("SELECT COUNT(*) FROM per_run_rollups").fetchone()[0] == 1

    storage.clear_per_run_rollups(conn=conn)
    assert conn.execute("SELECT COUNT(*) FROM per_run_rollups").fetchone()[0] == 0


def test_eligibility_qualifying_flattens_by_engine_dict(conn):
    storage.insert_eligibility_qualifying({"anthropic_claude": {"otter", "fathom"}, "openai_chatgpt": {"otter"}}, conn=conn)
    rows = conn.execute("SELECT engine, entity_id FROM eligibility_qualifying ORDER BY engine, entity_id").fetchall()
    assert [tuple(r) for r in rows] == [
        ("anthropic_claude", "fathom"), ("anthropic_claude", "otter"), ("openai_chatgpt", "otter"),
    ]

    storage.clear_eligibility_qualifying(conn=conn)
    assert conn.execute("SELECT COUNT(*) FROM eligibility_qualifying").fetchone()[0] == 0


def test_cell_rates_insert_and_clear(conn):
    from analysis.coding.b3_aggregates import CellRates

    row = CellRates(entity_id="otter", intent_id=1, engine="anthropic_claude", n_runs=8, mention_rate=1.0, recommendation_rate=0.875)
    storage.insert_cell_rates([row], conn=conn)
    assert conn.execute("SELECT COUNT(*) FROM cell_rates").fetchone()[0] == 1

    storage.clear_cell_rates(conn=conn)
    assert conn.execute("SELECT COUNT(*) FROM cell_rates").fetchone()[0] == 0


def test_serp_binary_insert_and_clear(conn):
    storage.insert_serp_binary([{"entity_id": "otter", "intent_id": 1, "in_google_top10": True}], conn=conn)
    assert conn.execute("SELECT COUNT(*) FROM serp_binary").fetchone()[0] == 1

    storage.clear_serp_binary(conn=conn)
    assert conn.execute("SELECT COUNT(*) FROM serp_binary").fetchone()[0] == 0


def test_alias_hits_insert_and_clear(conn):
    storage.insert_alias_hits([{"run_id": "run-1", "canonical_name": "Otter.ai"}], conn=conn)
    assert conn.execute("SELECT COUNT(*) FROM alias_hits").fetchone()[0] == 1

    storage.clear_alias_hits(conn=conn)
    assert conn.execute("SELECT COUNT(*) FROM alias_hits").fetchone()[0] == 0
