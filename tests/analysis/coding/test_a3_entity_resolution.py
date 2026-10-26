import json

from analysis.coding import storage
import pytest

from analysis.coding.a3_entity_resolution import (
    EntityResolutionResult,
    EntityRow,
    _chunk_candidates,
    _merge_entities,
    build_surface_form_candidates,
    call_entity_resolution,
    resolve_entities,
)
from analysis.coding.cache import LLMCache
from analysis.coding.db import init_coding_schema
from collector.config import AnthropicConfig
from collector.storage import api
from collector.storage.db import get_connection, init_db
from tests.fakes import FakeAnthropicClient, FakeMessages, fake_entity_resolution_response


def _fresh_conn(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    init_db(db_path)
    init_coding_schema(db_path)
    return get_connection(db_path)


def _seed_run(conn, run_id):
    api.insert_run(
        run_id=run_id, instrument_commit="deadbeef", timestamp="2026-08-24T10:00:00+00:00",
        engine="anthropic_claude", intent_id=1, paraphrase_id=1, prompt_text="x",
        rep_index=1, day_index=1, conn=conn,
    )
    conn.commit()


def _seed_mentions_raw(conn, rows):
    """rows: list of (run_id, surface_form) -- seeds a real runs row plus a
    minimal mentions_raw row for each, so the surface_forms view (which A3
    reads) has real content to group."""
    for run_id, surface_form in rows:
        _seed_run(conn, run_id)
    storage.insert_mentions_raw(
        [
            {"run_id": run_id, "surface_form": surface_form, "evidence_span": f"{surface_form} is good.", "is_recommended": True, "votes": ["anthropic", "openai"]}
            for run_id, surface_form in rows
        ],
        conn=conn,
    )

ROSIE_ENTITIES_JSON = json.dumps(
    [
        {
            "entity_id": "askrosie",
            "canonical_name": "AskRosie",
            "surface_forms": ["Rosie"],  # placeholder; real grouping keys on which row, tested via candidates
            "homepage_url": "https://askrosie.ai/",
        }
    ]
)


# --------------------------------------------------------------------------
# build_surface_form_candidates
# --------------------------------------------------------------------------

def test_build_candidates_collects_distinct_domains_per_surface_form():
    surface_forms_rows = [{"surface_form": "Rosie", "mention_count": "2", "run_ids": "run-1;run-2"}]
    sources_by_run = {
        "run-1": [{"domain": "askrosie.ai", "url_raw": "https://askrosie.ai/"}],
        "run-2": [{"domain": "askrosie.ai", "url_raw": "https://askrosie.ai/pricing"}],  # dup domain, dropped
    }
    candidates = build_surface_form_candidates(surface_forms_rows, sources_by_run)
    assert candidates == [{"surface_form": "Rosie", "mention_count": 2, "candidate_domains": ["askrosie.ai"]}]


def test_build_candidates_handles_missing_sources():
    surface_forms_rows = [{"surface_form": "Goodcall", "mention_count": "1", "run_ids": "run-1"}]
    candidates = build_surface_form_candidates(surface_forms_rows, sources_by_run={})
    assert candidates == [{"surface_form": "Goodcall", "mention_count": 1, "candidate_domains": []}]


def test_build_candidates_caps_domain_list():
    rows = [{"surface_form": "X", "mention_count": "9", "run_ids": "r"}]
    sbr = {"r": [{"domain": f"d{i}.com"} for i in range(30)]}
    cands = build_surface_form_candidates(rows, sbr, max_domains=8)
    assert cands[0]["candidate_domains"] == [f"d{i}.com" for i in range(8)]


# --------------------------------------------------------------------------
# chunking + cross-chunk merge (real-window scale: ~2,200 surface forms
# blow one 200k-token prompt, so A3 chunks and merges)
# --------------------------------------------------------------------------

def test_chunk_candidates_groups_name_variants_together():
    cands = [
        {"surface_form": "Zeta"}, {"surface_form": "Rosie AI"}, {"surface_form": "rosie"},
        {"surface_form": "RosieAI"}, {"surface_form": "Alpha"},
    ]
    chunks = _chunk_candidates(cands, chunk_size=3)
    flat = [c["surface_form"] for ch in chunks for c in ch]
    assert flat == ["Alpha", "rosie", "Rosie AI", "RosieAI", "Zeta"]  # all rosie* adjacent


def test_merge_entities_unions_across_chunks_on_name_or_domain():
    # "Claire AI" and "Smith.ai" were resolved in different chunks but share
    # a homepage domain -> merge, and the domain-matching name is kept.
    a = EntityRow("claire-ai", "Claire AI", ("Claire AI",), "https://smith.ai")
    b = EntityRow("smith-ai", "Smith.ai", ("Smith.ai", "Smith"), "https://smith.ai")
    c = EntityRow("retell", "Retell AI", ("Retell",), None)
    merged = _merge_entities([a, b, c])
    assert len(merged) == 2
    smith = next(e for e in merged if e.homepage_url == "https://smith.ai")
    assert smith.canonical_name == "Smith.ai"  # not "Claire AI"
    assert set(smith.surface_forms) == {"Claire AI", "Smith.ai", "Smith"}
    assert {e.canonical_name for e in merged} == {"Smith.ai", "Retell AI"}


def test_resolve_entities_raises_and_does_not_cache_when_a_chunk_errors(tmp_path):
    conn = _fresh_conn(tmp_path)
    _seed_mentions_raw(conn, [(f"run-{i}", sf) for i, sf in enumerate(["Aaa", "Bbb", "Ccc", "Ddd"])])
    cache = LLMCache(cache_dir=tmp_path / "cache")

    calls = {"n": 0}

    def flaky_call_fn(*, surface_form_candidates):
        calls["n"] += 1
        if any(c["surface_form"] == "Ccc" for c in surface_form_candidates):
            return EntityResolutionResult(error="APITimeoutError: boom")
        ents = [{"entity_id": c["surface_form"].lower(), "canonical_name": c["surface_form"],
                 "surface_forms": [c["surface_form"]], "homepage_url": None} for c in surface_form_candidates]
        return call_entity_resolution(
            surface_form_candidates=surface_form_candidates,
            config=AnthropicConfig(api_key="sk-ant-test"),
            client=FakeAnthropicClient(FakeMessages(response=fake_entity_resolution_response(json.dumps(ents)))),
        )

    import analysis.coding.a3_entity_resolution as a3
    orig = a3.RESOLUTION_CHUNK_SIZE
    a3.RESOLUTION_CHUNK_SIZE = 1
    try:
        with pytest.raises(TimeoutError):
            resolve_entities(sources_by_run={}, cache=cache, call_fn=flaky_call_fn, conn=conn)
        # the errored chunk is not cached -> a resume retries it
        n_after_first = calls["n"]
        with pytest.raises(TimeoutError):
            resolve_entities(sources_by_run={}, cache=cache, call_fn=flaky_call_fn, conn=conn)
        assert calls["n"] > n_after_first  # re-called the uncached failed chunk
    finally:
        a3.RESOLUTION_CHUNK_SIZE = orig


def test_resolve_entities_chunks_and_merges(tmp_path):
    conn = _fresh_conn(tmp_path)
    _seed_mentions_raw(conn, [(f"run-{i}", sf) for i, sf in enumerate(["Goodcall", "GoodCall", "Retell", "Vapi"])])

    def fake_call_fn(*, surface_form_candidates):
        # each chunk echoes one entity per surface form; "Goodcall"/"GoodCall"
        # split across chunks but share a homepage domain -> must merge.
        ents = []
        for c in surface_form_candidates:
            sf = c["surface_form"]
            url = "https://goodcall.com/" if sf.lower() == "goodcall" else None
            ents.append({"entity_id": sf.lower(), "canonical_name": sf, "surface_forms": [sf], "homepage_url": url})
        return call_entity_resolution(
            surface_form_candidates=surface_form_candidates,
            config=AnthropicConfig(api_key="sk-ant-test"),
            client=FakeAnthropicClient(FakeMessages(response=fake_entity_resolution_response(json.dumps(ents)))),
        )

    import analysis.coding.a3_entity_resolution as a3
    orig = a3.RESOLUTION_CHUNK_SIZE
    a3.RESOLUTION_CHUNK_SIZE = 2
    try:
        result = resolve_entities(sources_by_run={}, cache=LLMCache(cache_dir=tmp_path / "cache"),
                                  call_fn=fake_call_fn, conn=conn)
    finally:
        a3.RESOLUTION_CHUNK_SIZE = orig

    names = {e.canonical_name for e in result.entities}
    assert "Retell" in names and "Vapi" in names
    gc = [e for e in result.entities if "goodcall.com" in (e.homepage_url or "")]
    assert len(gc) == 1
    assert set(gc[0].surface_forms) == {"Goodcall", "GoodCall"}  # merged across chunks


# --------------------------------------------------------------------------
# call_entity_resolution
# --------------------------------------------------------------------------

def test_call_entity_resolution_happy_path():
    client = FakeAnthropicClient(FakeMessages(response=fake_entity_resolution_response(ROSIE_ENTITIES_JSON)))
    cfg = AnthropicConfig(api_key="sk-ant-test")

    result = call_entity_resolution(surface_form_candidates=[], config=cfg, client=client)

    assert result.error is None
    assert result.entities == [
        EntityRow("askrosie", "AskRosie", ("Rosie",), "https://askrosie.ai/")
    ]
    assert result.tokens_input == 200


def test_call_entity_resolution_transport_failure_produces_error_not_exception():
    client = FakeAnthropicClient(FakeMessages(exc=TimeoutError("simulated timeout")))
    cfg = AnthropicConfig(api_key="sk-ant-test")

    result = call_entity_resolution(surface_form_candidates=[], config=cfg, client=client)

    assert result.error == "TimeoutError: simulated timeout"
    assert result.entities == []


# --------------------------------------------------------------------------
# resolve_entities -- the askrosie.ai/heyrosie.com collision is the single
# most important assertion in this test suite (spec's stated known trap).
# --------------------------------------------------------------------------

def test_resolve_entities_writes_registry_from_fake_response(tmp_path):
    conn = _fresh_conn(tmp_path)
    _seed_mentions_raw(conn, [("run-1", "Goodcall")])

    def fake_call_fn(*, surface_form_candidates):
        return call_entity_resolution(
            surface_form_candidates=surface_form_candidates,
            config=AnthropicConfig(api_key="sk-ant-test"),
            client=FakeAnthropicClient(FakeMessages(response=fake_entity_resolution_response(json.dumps([
                {"entity_id": "goodcall", "canonical_name": "Goodcall", "surface_forms": ["Goodcall"], "homepage_url": "https://goodcall.com/"}
            ])))),
        )

    cache = LLMCache(cache_dir=tmp_path / "cache")

    resolve_entities(sources_by_run={}, cache=cache, call_fn=fake_call_fn, conn=conn)

    rows = storage.list_entities(conn=conn)
    assert rows == [{
        "entity_id": "goodcall", "canonical_name": "Goodcall", "homepage_url": "https://goodcall.com/",
        "meets_frame": None, "in_seed_list": None, "segment": None, "evidence_url": None, "frame_reason": None,
        "surface_forms": "Goodcall",
    }]


def test_resolve_entities_askrosie_heyrosie_collision_resolves_to_distinct_entities(tmp_path):
    # Both rows say the bare word "Rosie" in the text -- only the cited
    # source domain (askrosie.ai vs heyrosie.com) tells them apart. A real
    # LLM call would use build_surface_form_candidates' domain evidence to
    # make this call; here we fake the model's (correct) response directly
    # to assert the pipeline plumbs distinct entities through to storage.
    conn = _fresh_conn(tmp_path)
    _seed_mentions_raw(conn, [("google_ai_mode-run-a", "Rosie"), ("google_gemini-run-b", "Rosie")])

    sources_by_run = {
        "google_ai_mode-run-a": [{"domain": "askrosie.ai", "url_raw": "https://askrosie.ai/"}],
        "google_gemini-run-b": [{"domain": "heyrosie.com", "url_raw": "https://heyrosie.com/"}],
    }

    two_entities_json = json.dumps(
        [
            {"entity_id": "askrosie", "canonical_name": "AskRosie", "surface_forms": ["Rosie"], "homepage_url": "https://askrosie.ai/"},
            {"entity_id": "heyrosie", "canonical_name": "HeyRosie", "surface_forms": ["Rosie"], "homepage_url": "https://heyrosie.com/"},
        ]
    )

    def fake_call_fn(*, surface_form_candidates):
        # Sanity: the candidate context actually carries both domains, which
        # is what would let a real model disambiguate.
        assert surface_form_candidates == [
            {"surface_form": "Rosie", "mention_count": 2, "candidate_domains": ["askrosie.ai", "heyrosie.com"]}
        ]
        return call_entity_resolution(
            surface_form_candidates=surface_form_candidates,
            config=AnthropicConfig(api_key="sk-ant-test"),
            client=FakeAnthropicClient(FakeMessages(response=fake_entity_resolution_response(two_entities_json))),
        )

    cache = LLMCache(cache_dir=tmp_path / "cache")

    result = resolve_entities(sources_by_run=sources_by_run, cache=cache, call_fn=fake_call_fn, conn=conn)

    assert len(result.entities) == 2
    entity_ids = {e.entity_id for e in result.entities}
    assert entity_ids == {"askrosie", "heyrosie"}
    homepages = {e.entity_id: e.homepage_url for e in result.entities}
    assert homepages == {"askrosie": "https://askrosie.ai/", "heyrosie": "https://heyrosie.com/"}

    # And it actually landed in storage as two distinct entities, not just
    # in the returned in-memory result.
    stored_ids = {row["entity_id"] for row in storage.list_entities(conn=conn)}
    assert stored_ids == {"askrosie", "heyrosie"}


def test_resolve_entities_uses_cache_on_second_call(tmp_path):
    conn = _fresh_conn(tmp_path)
    _seed_mentions_raw(conn, [("run-1", "Goodcall")])

    call_count = {"n": 0}

    def counting_call_fn(*, surface_form_candidates):
        call_count["n"] += 1
        return call_entity_resolution(
            surface_form_candidates=surface_form_candidates,
            config=AnthropicConfig(api_key="sk-ant-test"),
            client=FakeAnthropicClient(FakeMessages(response=fake_entity_resolution_response(json.dumps([
                {"entity_id": "goodcall", "canonical_name": "Goodcall", "surface_forms": ["Goodcall"], "homepage_url": "https://goodcall.com/"}
            ])))),
        )

    cache = LLMCache(cache_dir=tmp_path / "cache")
    resolve_entities(sources_by_run={}, cache=cache, call_fn=counting_call_fn, conn=conn)
    resolve_entities(sources_by_run={}, cache=cache, call_fn=counting_call_fn, conn=conn)

    assert call_count["n"] == 1
