from analysis.coding.batch_session import BatchSession, default_session_path


def _session(**over):
    base = dict(
        runs_dir="runs", db_path="data/c.sqlite3", cache_dir="data/cache",
        unbranded_intents=[1, 2], inclusion_frame_path="f.yaml", seed_list_path="s.csv",
        alt_domains_seed_path="a.csv", cited_url_resolution_path="r.jsonl",
        skip_entity_verification=False, report_path=None,
    )
    base.update(over)
    return BatchSession(**base)


def test_default_session_path_sits_next_to_db():
    assert default_session_path("data/03_coded/coding.sqlite3").name == "coding.batch-session.json"


def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "s.json"
    s = _session(primary_batch_ids={"openai": "b1", "gemini": "b2"})
    s.advance("submitted", "two batches", "2026-08-31T00:00:00+00:00")
    s.save(p)

    back = BatchSession.load(p)
    assert back.primary_batch_ids == {"openai": "b1", "gemini": "b2"}
    assert back.phase == "submitted"
    assert back.unbranded_intents == [1, 2]
    assert back.skip_eligibility_floor is False
    assert back.limit is None
    assert back.history == ["2026-08-31T00:00:00+00:00  submitted  two batches"]


def test_advance_appends_history_and_updates_phase():
    s = _session()
    s.advance("primaries_done", "fetched", "T1")
    s.advance("arbiter_submitted", "abc", "T2")
    assert s.phase == "arbiter_submitted"
    assert s.updated_at == "T2"
    assert [h.split("  ")[1] for h in s.history] == ["primaries_done", "arbiter_submitted"]
