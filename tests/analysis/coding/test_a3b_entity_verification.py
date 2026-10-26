import json

import pytest

from analysis.coding import storage
from analysis.coding.a3b_entity_verification import (
    VerificationContext,
    VerificationResult,
    VerificationRow,
    build_verification_context,
    call_entity_verification,
    fetch_verification_batch_results,
    is_verified_vendor,
    poll_verification_batch,
    submit_verification_batch,
    verify_entities,
)
from analysis.coding.cache import LLMCache
from analysis.coding.db import init_coding_schema
from collector.config import AnthropicConfig
from collector.storage.db import get_connection, init_db
from tests.fakes import (
    FakeAnthropicBatchClient,
    FakeAnthropicBatches,
    FakeAnthropicClient,
    FakeMessages,
    fake_anthropic_batch_result_entry,
    fake_entity_verification_response,
)

VENDOR_JSON = json.dumps({
    "is_real": True, "entity_type": "vendor",
    "homepage_url": "https://sembly.ai", "disambiguation_note": None, "confidence": 0.9,
})
DIRECTORY_JSON = json.dumps({
    "is_real": True, "entity_type": "directory",
    "homepage_url": "https://www.g2.com", "disambiguation_note": None, "confidence": 0.95,
})


def _fresh_conn(tmp_path):
    db_path = tmp_path / "t.sqlite3"
    init_db(db_path)
    init_coding_schema(db_path)
    return get_connection(db_path)


# --------------------------------------------------------------------------
# call_entity_verification
# --------------------------------------------------------------------------

def test_call_parses_json_after_search_narration():
    client = FakeAnthropicClient(FakeMessages(
        response=fake_entity_verification_response(VENDOR_JSON, preamble="Let me search. Sembly is a meeting AI.\n\n")
    ))
    result = call_entity_verification(
        canonical_name="Sembly", surface_forms=["Sembly", "Sembly AI"], evidence_spans=["Sembly transcribes meetings"],
        candidate_domains=["sembly.ai"], config=AnthropicConfig(api_key="sk-ant-test"), client=client,
    )
    assert result.error is None
    assert result.verification == VerificationRow(True, "vendor", "https://sembly.ai", None, 0.9)
    # web_search tool was attached
    assert client.messages.last_kwargs["tools"][0]["name"] == "web_search"


def test_call_transport_failure_is_error_not_exception():
    client = FakeAnthropicClient(FakeMessages(exc=TimeoutError("boom")))
    result = call_entity_verification(
        canonical_name="X", surface_forms=["X"], evidence_spans=[], candidate_domains=[],
        config=AnthropicConfig(api_key="sk-ant-test"), client=client,
    )
    assert result.error == "TimeoutError: boom"
    assert result.verification is None


def test_call_unknown_entity_type_coerced():
    client = FakeAnthropicClient(FakeMessages(response=fake_entity_verification_response(
        json.dumps({"is_real": True, "entity_type": "megacorp", "homepage_url": None,
                    "disambiguation_note": None, "confidence": 0.2})
    )))
    result = call_entity_verification(
        canonical_name="X", surface_forms=["X"], evidence_spans=[], candidate_domains=[],
        config=AnthropicConfig(api_key="sk-ant-test"), client=client,
    )
    assert result.verification.entity_type == "unknown"


# --------------------------------------------------------------------------
# build_verification_context
# --------------------------------------------------------------------------

def test_context_groups_spans_and_domains_by_entity():
    entities = [{"entity_id": "otter", "canonical_name": "Otter", "surface_forms": "Otter;Otter.ai"}]
    mentions = [
        {"run_id": "r1", "surface_form": "Otter", "evidence_span": "Otter is great", "is_recommended": True, "votes": []},
        {"run_id": "r2", "surface_form": "Otter.ai", "evidence_span": "Otter.ai transcribes", "is_recommended": False, "votes": []},
        {"run_id": "r2", "surface_form": "Otter.ai", "evidence_span": "Otter.ai transcribes", "is_recommended": False, "votes": []},
    ]
    sources = {"r1": [{"domain": "otter.ai"}], "r2": [{"domain": "otter.ai"}, {"domain": "techcrunch.com"}]}

    ctx = build_verification_context(entities, mentions, sources)["otter"]
    assert ctx.surface_forms == ["Otter", "Otter.ai"]
    assert ctx.evidence_spans == ["Otter is great", "Otter.ai transcribes"]  # deduped
    assert ctx.candidate_domains == ["otter.ai", "techcrunch.com"]


# --------------------------------------------------------------------------
# verify_entities
# --------------------------------------------------------------------------

def test_verify_entities_writes_one_row_per_entity_and_caches(tmp_path):
    conn = _fresh_conn(tmp_path)
    storage.insert_entities([
        {"entity_id": "sembly", "canonical_name": "Sembly", "homepage_url": None, "surface_forms": ["Sembly"]},
        {"entity_id": "g2", "canonical_name": "G2", "homepage_url": None, "surface_forms": ["G2"]},
    ], conn=conn)

    calls = {"n": 0}

    def fake_call_fn(*, canonical_name, surface_forms, evidence_spans, candidate_domains):
        calls["n"] += 1
        payload = VENDOR_JSON if canonical_name == "Sembly" else DIRECTORY_JSON
        return call_entity_verification(
            canonical_name=canonical_name, surface_forms=surface_forms, evidence_spans=evidence_spans,
            candidate_domains=candidate_domains, config=AnthropicConfig(api_key="sk-ant-test"),
            client=FakeAnthropicClient(FakeMessages(response=fake_entity_verification_response(payload))),
        )

    ctx = {"sembly": VerificationContext(["Sembly"]), "g2": VerificationContext(["G2"])}
    cache = LLMCache(cache_dir=tmp_path / "cache")
    verify_entities(context_by_entity=ctx, cache=cache, call_fn=fake_call_fn, conn=conn)
    verify_entities(context_by_entity=ctx, cache=cache, call_fn=fake_call_fn, conn=conn)  # cached

    assert calls["n"] == 2  # two entities, second pass all cache hits
    rows = {r["entity_id"]: r for r in storage.list_entity_verification(conn=conn)}
    assert rows["sembly"]["entity_type"] == "vendor"
    assert rows["sembly"]["verified_homepage_url"] == "https://sembly.ai"
    assert rows["g2"]["entity_type"] == "directory"


# --------------------------------------------------------------------------
# Batch API path
# --------------------------------------------------------------------------

def _batch_fns(results_by_id, *, status="ended"):
    submitted: list[list[str]] = []

    def submit(items):
        submitted.append([i.custom_id for i in items])
        return "vb-1"

    def poll(batch_id, *, max_wait_seconds):
        if status is None:
            raise TimeoutError("still running")
        return status

    def fetch(batch_id):
        return results_by_id

    return (submit, poll, fetch), submitted


def test_submit_verification_batch_uses_shared_request_body():
    batches = FakeAnthropicBatches(batch_id="vb-9")
    client = FakeAnthropicBatchClient(batches)
    from analysis.coding.a3b_entity_verification import VerificationBatchItem

    bid = submit_verification_batch(
        [VerificationBatchItem("sembly", "Sembly", ["Sembly"], ["notes"], ["sembly.ai"])],
        config=AnthropicConfig(api_key="sk-ant-test"), client=client,
    )
    assert bid == "vb-9"
    req = batches.create_kwargs["requests"][0]
    assert req["custom_id"] == "sembly"
    assert req["params"]["model"] == "claude-sonnet-5"
    assert req["params"]["tools"][0]["name"] == "web_search"


def test_fetch_verification_batch_results_parses_success_and_error():
    batches = FakeAnthropicBatches(results=[
        fake_anthropic_batch_result_entry(custom_id="sembly", text_json=VENDOR_JSON),
    ])
    client = FakeAnthropicBatchClient(batches)
    out = fetch_verification_batch_results("vb-1", config=AnthropicConfig(api_key="sk-ant-test"), client=client)
    assert out["sembly"].verification.entity_type == "vendor"


def test_verify_entities_batch_prefills_cache_and_skips_sync_calls(tmp_path):
    conn = _fresh_conn(tmp_path)
    storage.insert_entities([
        {"entity_id": "sembly", "canonical_name": "Sembly", "homepage_url": None, "surface_forms": ["Sembly"]},
        {"entity_id": "g2", "canonical_name": "G2", "homepage_url": None, "surface_forms": ["G2"]},
    ], conn=conn)

    results_by_id = {
        "sembly": VerificationResult(verification=VerificationRow(True, "vendor", "https://sembly.ai", None, 0.9)),
        "g2": VerificationResult(verification=VerificationRow(True, "directory", "https://g2.com", None, 0.9)),
    }
    fns, submitted = _batch_fns(results_by_id)

    def no_sync(**_kw):
        raise AssertionError("sync call_fn must not run when the batch covered every entity")

    ctx = {"sembly": VerificationContext(["Sembly"]), "g2": VerificationContext(["G2"])}
    cache = LLMCache(cache_dir=tmp_path / "cache")
    verify_entities(context_by_entity=ctx, cache=cache, call_fn=no_sync, use_batch=True, batch_fns=fns, conn=conn)

    assert sorted(submitted[0]) == ["g2", "sembly"]
    rows = {r["entity_id"]: r for r in storage.list_entity_verification(conn=conn)}
    assert rows["sembly"]["entity_type"] == "vendor"
    assert rows["g2"]["entity_type"] == "directory"
    assert not (cache.cache_dir / "_pending_verification_batch.json").exists()  # cleared on success


def test_verify_entities_batch_timeout_parks_batch_id_for_resume(tmp_path):
    conn = _fresh_conn(tmp_path)
    storage.insert_entities(
        [{"entity_id": "sembly", "canonical_name": "Sembly", "homepage_url": None, "surface_forms": ["Sembly"]}],
        conn=conn,
    )
    fns, _ = _batch_fns({}, status=None)  # poll raises TimeoutError
    cache = LLMCache(cache_dir=tmp_path / "cache")

    with pytest.raises(TimeoutError):
        verify_entities(
            context_by_entity={"sembly": VerificationContext(["Sembly"])},
            cache=cache, call_fn=lambda **k: None, use_batch=True, batch_fns=fns, conn=conn,
        )

    parked = json.loads((cache.cache_dir / "_pending_verification_batch.json").read_text())
    assert parked["batch_id"] == "vb-1"

    # resume: batch now done -> uses the parked id, does not re-submit
    fns2, submitted2 = _batch_fns(
        {"sembly": VerificationResult(verification=VerificationRow(True, "vendor", "https://sembly.ai", None, 0.9))}
    )
    verify_entities(
        context_by_entity={"sembly": VerificationContext(["Sembly"])},
        cache=cache, call_fn=lambda **k: None, use_batch=True, batch_fns=fns2, conn=conn,
    )
    assert submitted2 == []  # no new submission
    assert not (cache.cache_dir / "_pending_verification_batch.json").exists()


# --------------------------------------------------------------------------
# is_verified_vendor
# --------------------------------------------------------------------------

def test_is_verified_vendor_gate():
    assert is_verified_vendor(None) is True  # missing = fail open
    assert is_verified_vendor({"is_real": True, "entity_type": "vendor"}) is True
    assert is_verified_vendor({"is_real": True, "entity_type": "directory"}) is False
    assert is_verified_vendor({"is_real": False, "entity_type": "vendor"}) is False
    assert is_verified_vendor({"is_real": None, "entity_type": None}) is False
