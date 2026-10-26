"""The detached submit/collect helpers in a1_answer_coding.py."""
import pytest

from analysis.coding.a1_answer_coding import (
    PRIMARY_EXTRACTORS,
    BatchExtractor,
    _arbiter_cache_key,
    disputed_by_run_from_cache,
    fetch_arbiter_batch_into_cache,
    fetch_primary_batches,
    primary_batch_statuses,
    run_a1,
    run_arbiter_sync_into_cache,
    submit_arbiter_batch_from_cache,
    submit_primary_batches,
)
from analysis.coding.cache import LLMCache
from analysis.coding.db import init_coding_schema
from analysis.coding.extractors.arbiter import ArbiterResult, ArbiterVote
from analysis.coding.extractors.base import BatchItem, ExtractionResult, MentionCandidate
from collector.storage.db import get_connection, init_db

NAME_A, NAME_B = PRIMARY_EXTRACTORS


def _runs():
    return [
        {"run_id": "r1", "raw_response_text": "Otter and Fireflies are great."},
        {"run_id": "r2", "raw_response_text": "Fireflies is great."},
        {"run_id": "r3", "raw_response_text": None},  # errored run, ignored
    ]


def _batch_extractor(mentions_by_run, *, submitted=None, status="ended"):
    def submit(items):
        if submitted is not None:
            submitted.append([i.custom_id for i in items])
        return "b-fake"

    def poll(batch_id, *, max_wait_seconds):
        if status is None:
            raise TimeoutError("still running")
        return status

    def fetch(batch_id, items):
        return {i.custom_id: ExtractionResult(mentions=list(mentions_by_run.get(i.custom_id, []))) for i in items}

    return BatchExtractor(submit=submit, poll=poll, fetch=fetch)


def test_submit_primary_batches_one_per_primary_skips_cached(tmp_path):
    cache = LLMCache(cache_dir=tmp_path / "ext")
    submitted_a, submitted_b = [], []
    bx = {
        NAME_A: _batch_extractor({}, submitted=submitted_a),
        NAME_B: _batch_extractor({}, submitted=submitted_b),
    }
    ids = submit_primary_batches(_runs(), cache=cache, batch_extractors=bx)
    assert set(ids) == {NAME_A, NAME_B}
    assert sorted(submitted_a[0]) == ["r1", "r2"]  # r3 (no text) excluded

    # after caching everything, a re-submit has nothing to do
    fetch_primary_batches(_runs(), ids, cache=cache, batch_extractors=bx)
    assert submit_primary_batches(_runs(), cache=cache, batch_extractors=bx) == {}


def test_primary_batch_statuses_reports_none_while_running():
    bx = {NAME_A: _batch_extractor({}, status="ended"), NAME_B: _batch_extractor({}, status=None)}
    statuses = primary_batch_statuses({NAME_A: "x", NAME_B: "y"}, batch_extractors=bx)
    assert statuses == {NAME_A: "ended", NAME_B: None}


def test_fetch_primary_batches_writes_results_and_disputed_is_computed(tmp_path):
    cache = LLMCache(cache_dir=tmp_path / "ext")
    # A sees Otter in r1; B sees nothing -> "Otter" is a single-vote (disputed) mention.
    bx = {
        NAME_A: _batch_extractor({"r1": [MentionCandidate("Otter", "Otter and Fireflies are great.", True)]}),
        NAME_B: _batch_extractor({}),
    }
    ids = submit_primary_batches(_runs(), cache=cache, batch_extractors=bx)
    fetch_primary_batches(_runs(), ids, cache=cache, batch_extractors=bx)

    disputed = disputed_by_run_from_cache(_runs(), cache=cache)
    assert list(disputed) == ["r1"]
    assert disputed["r1"][0]["surface_form"] == "Otter"


def test_arbiter_batch_submit_and_fetch_roundtrip(tmp_path):
    ext_cache = LLMCache(cache_dir=tmp_path / "ext")
    arb_cache = LLMCache(cache_dir=tmp_path / "arb")
    bx = {
        NAME_A: _batch_extractor({"r1": [MentionCandidate("Otter", "Otter and Fireflies are great.", True)]}),
        NAME_B: _batch_extractor({}),
    }
    ids = submit_primary_batches(_runs(), cache=ext_cache, batch_extractors=bx)
    fetch_primary_batches(_runs(), ids, cache=ext_cache, batch_extractors=bx)
    disputed = disputed_by_run_from_cache(_runs(), cache=ext_cache)

    submits = []

    def submit_fn(items):
        submits.append([i.custom_id for i in items])
        return "arb-fake"

    def fetch_fn(batch_id):
        return {"r1": ArbiterResult(votes=[ArbiterVote(surface_form="Otter", confirmed=True, is_recommended=True)])}

    batch_id = submit_arbiter_batch_from_cache(_runs(), disputed, arbiter_cache=arb_cache, submit_fn=submit_fn)
    assert batch_id == "arb-fake"
    assert submits == [["r1"]]

    fetch_arbiter_batch_into_cache(_runs(), disputed, batch_id, arbiter_cache=arb_cache, fetch_fn=fetch_fn)
    # second submit finds it cached -> nothing to do
    assert submit_arbiter_batch_from_cache(_runs(), disputed, arbiter_cache=arb_cache, submit_fn=submit_fn) is None


def test_run_arbiter_sync_into_cache_calls_only_uncached_disputed_runs(tmp_path):
    ext_cache = LLMCache(cache_dir=tmp_path / "ext")
    arb_cache = LLMCache(cache_dir=tmp_path / "arb")
    # r1: "Otter" seen by only NAME_A -> disputed; r2: "Fireflies" by both -> agreed.
    bx = {
        NAME_A: _batch_extractor({
            "r1": [MentionCandidate("Otter", "Otter and Fireflies are great.", True)],
            "r2": [MentionCandidate("Fireflies", "Fireflies is great.", True)],
        }),
        NAME_B: _batch_extractor({"r2": [MentionCandidate("Fireflies", "Fireflies is great.", True)]}),
    }
    ids = submit_primary_batches(_runs(), cache=ext_cache, batch_extractors=bx)
    fetch_primary_batches(_runs(), ids, cache=ext_cache, batch_extractors=bx)
    disputed = disputed_by_run_from_cache(_runs(), cache=ext_cache)
    assert list(disputed) == ["r1"]

    seen = []

    def fake_arbiter(*, raw_response_text, candidates):
        seen.append(raw_response_text)
        return ArbiterResult(votes=[ArbiterVote(surface_form="Otter", confirmed=True, is_recommended=False)])

    made = run_arbiter_sync_into_cache(_runs(), disputed, arbiter_cache=arb_cache, call_fn=fake_arbiter, max_workers=2)
    assert made == 1 and len(seen) == 1
    assert arb_cache.get(_arbiter_cache_key("Otter and Fireflies are great.", disputed["r1"])) is not None

    # re-run: everything cached -> no calls
    made_again = run_arbiter_sync_into_cache(
        _runs(), disputed, arbiter_cache=arb_cache, call_fn=fake_arbiter, max_workers=2
    )
    assert made_again == 0 and len(seen) == 1


def test_run_a1_offline_raises_when_cache_incomplete(tmp_path):
    db = tmp_path / "d.sqlite3"
    init_db(db)
    init_coding_schema(db)
    conn = get_connection(db)
    with pytest.raises(RuntimeError, match="offline"):
        run_a1(
            _runs(), cache=LLMCache(cache_dir=tmp_path / "ext"),
            arbiter_cache=LLMCache(cache_dir=tmp_path / "arb"),
            use_batch_api=True, offline=True, conn=conn,
        )
    conn.close()
