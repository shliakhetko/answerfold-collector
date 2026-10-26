"""submit_coding_batches / collect_coding_batches state machine, with the
vendor batch calls and the heavy finalize stubbed out."""
import json

import pytest

from analysis.coding.batch_session import BatchSession
from collector.storage.db import get_connection, init_db
import scripts.coding.collect_coding_batches as collect_mod
import scripts.coding.submit_coding_batches as submit_mod


class _FakeReport:
    entity_verification = {"ran": False}


def _make_runs_dir(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    rec = {
        "run_id": "run-1", "ts_utc": "2026-08-24T10:00:00+00:00", "engine": "anthropic_claude",
        "intent_id": 1, "paraphrase_id": 1, "rep_index": 1, "day_index": 1,
        "request": {"model": "claude-sonnet-5", "messages": [{"role": "user", "content": "q"}]},
        "response": {"model": "claude-sonnet-5", "stop_reason": "end_turn",
                     "content": [{"type": "text", "text": "Otter is good.", "citations": []}]},
    }
    (runs_dir / "2026-08-24.jsonl").write_text(json.dumps(rec) + "\n")
    return runs_dir


def _common_args(tmp_path, runs_dir):
    frame = tmp_path / "frame.yaml"; frame.write_text("stub")
    seed = tmp_path / "seed.csv"; seed.write_text("canonical_name,homepage_domain\n")
    alt = tmp_path / "alt.csv"; alt.write_text("entity_id,alt_domain,source_note\n")
    return [
        "--runs-dir", str(runs_dir), "--db-path", str(tmp_path / "c.sqlite3"),
        "--unbranded-intents", "1", "--inclusion-frame-path", str(frame),
        "--seed-list-path", str(seed), "--alt-domains-seed-path", str(alt),
        "--cache-dir", str(tmp_path / "cache"), "--no-url-resolution",
        "--session-path", str(tmp_path / "sess.json"),
    ]


def test_submit_writes_session_with_batch_ids(tmp_path, monkeypatch, capsys):
    runs_dir = _make_runs_dir(tmp_path)
    monkeypatch.setattr(submit_mod, "submit_primary_batches", lambda *a, **k: {"openai": "b-o", "gemini": "b-g"})

    submit_mod.main(_common_args(tmp_path, runs_dir))

    session = BatchSession.load(tmp_path / "sess.json")
    assert session.phase == "submitted"
    assert session.primary_batch_ids == {"openai": "b-o", "gemini": "b-g"}
    assert "shut the machine down" in capsys.readouterr().out


def test_submit_with_nothing_to_send_goes_straight_to_primaries_done(tmp_path, monkeypatch):
    runs_dir = _make_runs_dir(tmp_path)
    monkeypatch.setattr(submit_mod, "submit_primary_batches", lambda *a, **k: {})
    submit_mod.main(_common_args(tmp_path, runs_dir))
    assert BatchSession.load(tmp_path / "sess.json").phase == "primaries_done"


def _write_session(tmp_path, **over):
    base = dict(
        runs_dir=str(tmp_path / "runs"), db_path=str(tmp_path / "c.sqlite3"),
        cache_dir=str(tmp_path / "cache"), unbranded_intents=[1],
        inclusion_frame_path=str(tmp_path / "frame.yaml"), seed_list_path=str(tmp_path / "seed.csv"),
        alt_domains_seed_path=str(tmp_path / "alt.csv"), cited_url_resolution_path=None,
        skip_entity_verification=False, report_path=None,
        phase="submitted", primary_batch_ids={"openai": "b-o", "gemini": "b-g"},
    )
    base.update(over)
    s = BatchSession(**base)
    s.save(tmp_path / "sess.json")
    # a DB with the one run, so api.list_runs works
    from analysis.aggregate.load_runs import load_jsonl_dir
    from analysis.coding.db import init_coding_schema
    _make_runs_dir(tmp_path)
    init_db(tmp_path / "c.sqlite3"); init_coding_schema(tmp_path / "c.sqlite3")
    load_jsonl_dir(tmp_path / "runs", db_path=tmp_path / "c.sqlite3", instrument_commit="x")
    return tmp_path / "sess.json"


def test_collect_exits_2_while_primary_batch_still_running(tmp_path, monkeypatch):
    sess = _write_session(tmp_path)
    monkeypatch.setattr(collect_mod, "primary_batch_statuses", lambda *a, **k: {"anthropic": "ended", "gemini": None})

    with pytest.raises(SystemExit) as exc:
        collect_mod.main(["--session-path", str(sess)])
    assert exc.value.code == collect_mod.STILL_WAITING
    assert BatchSession.load(sess).phase == "submitted"  # unchanged


def test_collect_aborts_when_a_primary_batch_failed(tmp_path, monkeypatch):
    sess = _write_session(tmp_path)
    monkeypatch.setattr(
        collect_mod, "primary_batch_statuses",
        lambda *a, **k: {"anthropic": "failed", "gemini": "JOB_STATE_SUCCEEDED"},
    )
    monkeypatch.setattr(collect_mod, "fetch_primary_batches",
                        lambda *a, **k: pytest.fail("must not fetch a failed batch"))

    with pytest.raises(SystemExit) as exc:
        collect_mod.main(["--session-path", str(sess)])
    assert exc.value.code == 1
    assert BatchSession.load(sess).phase == "submitted"  # unchanged


def test_collect_advances_through_all_phases_when_everything_ready(tmp_path, monkeypatch):
    sess = _write_session(tmp_path)
    calls = []
    monkeypatch.setattr(collect_mod, "primary_batch_statuses",
                        lambda *a, **k: {"anthropic": "ended", "gemini": "JOB_STATE_SUCCEEDED"})
    monkeypatch.setattr(collect_mod, "fetch_primary_batches", lambda *a, **k: calls.append("fetch_primaries"))
    monkeypatch.setattr(collect_mod, "disputed_by_run_from_cache",
                        lambda *a, **k: {"run-1": [{"surface_form": "Otter", "evidence_span": "Otter is good."}]})
    monkeypatch.setattr(collect_mod, "run_arbiter_sync_into_cache",
                        lambda *a, **k: calls.append("arbiter_sync") or 1)

    def _fake_run_pipeline(**k):
        calls.append(("run_pipeline", k["a1_offline"]))
        return _FakeReport()

    monkeypatch.setattr(collect_mod, "run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr(collect_mod, "_print_summary", lambda r: None)

    collect_mod.main(["--session-path", str(sess)])

    assert BatchSession.load(sess).phase == "complete"
    assert "fetch_primaries" in calls and "arbiter_sync" in calls
    assert ("run_pipeline", True) in calls  # a1_offline=True
