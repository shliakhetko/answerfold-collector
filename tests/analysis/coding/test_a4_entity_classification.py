import json

import pytest

from analysis.coding import storage
from analysis.coding.a4_entity_classification import (
    ClassificationResult,
    ClassificationRow,
    EntityClassificationBatchItem,
    _classification_cache_key,
    call_entity_classification,
    classify_entities,
    fetch_entity_classification_batch_results,
    is_in_seed_list,
    load_seed_list,
    submit_entity_classification_batch,
)
from analysis.coding.cache import LLMCache
from analysis.coding.db import init_coding_schema
from collector.config import AnthropicConfig
from collector.storage.db import get_connection, init_db
from tests.fakes import (
    FakeAnthropicBatchClient,
    FakeAnthropicBatches,
    FakeAnthropicClient,
    FakeHomepageFetcher,
    FakeMessages,
    fake_anthropic_batch_error_entry,
    fake_anthropic_batch_result_entry,
    fake_classification_response,
)

GOODCALL_JSON = json.dumps(
    {"meets_frame": "in", "frame_reason": None, "segment": "done-for-you",
     "evidence_url": "https://goodcall.com/product"}
)
# Human-staffed answering services are a resolved hard line: IN, segment
# human_staffed -- not an out-of-frame case (see config/inclusion_frame.yaml).
METRO_JSON = json.dumps(
    {"meets_frame": "in", "frame_reason": None, "segment": "human_staffed",
     "evidence_url": "https://metroanswering.example.com/about"}
)
# A local dental practice named as a usage example -- a genuine OUT case
# (end_user: buys these products, doesn't sell them).
DENTAL_JSON = json.dumps(
    {"meets_frame": "out", "frame_reason": "end_user", "segment": None,
     "evidence_url": "https://brightsmiledental.example.com/"}
)


def _fresh_conn(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    init_db(db_path)
    init_coding_schema(db_path)
    return get_connection(db_path)


def _seed_entities(conn, rows):
    """rows: (entity_id, canonical_name, surface_form, homepage_url) tuples,
    same shape the old entity_registry.csv fixture used."""
    storage.insert_entities(
        [{"entity_id": r[0], "canonical_name": r[1], "surface_forms": [r[2]], "homepage_url": r[3]} for r in rows],
        conn=conn,
    )


# --------------------------------------------------------------------------
# call_entity_classification
# --------------------------------------------------------------------------

def test_call_entity_classification_happy_path():
    client = FakeAnthropicClient(FakeMessages(response=fake_classification_response(GOODCALL_JSON)))
    cfg = AnthropicConfig(api_key="sk-ant-test")

    result = call_entity_classification(
        canonical_name="Goodcall", homepage_url="https://goodcall.com/", homepage_text="AI receptionist for local businesses.",
        inclusion_frame_text="STUB frame", config=cfg, client=client,
    )

    assert result.error is None
    assert result.classification == ClassificationRow("in", None, "done-for-you", "https://goodcall.com/product")


def test_call_entity_classification_tolerates_trailing_content_after_json():
    # ~22% of A4 responses on the real run had a valid JSON object followed
    # by trailing content (a stray note, a repeated line) -- json.loads on
    # the whole string fails with "Extra data"; raw_decode must not.
    trailing = GOODCALL_JSON + "\n\nNote: classified from the homepage only.\n"
    client = FakeAnthropicClient(FakeMessages(response=fake_classification_response(trailing)))
    result = call_entity_classification(
        canonical_name="Goodcall", homepage_url="https://goodcall.com/", homepage_text="...",
        inclusion_frame_text="STUB frame", config=AnthropicConfig(api_key="sk-ant-test"), client=client,
    )
    assert result.error is None
    assert result.classification == ClassificationRow("in", None, "done-for-you", "https://goodcall.com/product")


def test_call_entity_classification_transport_failure_produces_error_not_exception():
    client = FakeAnthropicClient(FakeMessages(exc=TimeoutError("simulated timeout")))
    cfg = AnthropicConfig(api_key="sk-ant-test")

    result = call_entity_classification(
        canonical_name="Goodcall", homepage_url="https://goodcall.com/", homepage_text="...",
        inclusion_frame_text="STUB frame", config=cfg, client=client,
    )

    assert result.error == "TimeoutError: simulated timeout"
    assert result.classification is None


# --------------------------------------------------------------------------
# seed list
# --------------------------------------------------------------------------

def test_load_seed_list_skips_comment_lines(tmp_path):
    seed_path = tmp_path / "seed.csv"
    seed_path.write_text("# a comment\ncanonical_name,homepage_domain\nGoodcall,goodcall.com\n")
    assert load_seed_list(seed_path) == {"goodcall", "goodcall.com"}


def test_is_in_seed_list_matches_by_name_or_domain():
    seed_list = {"goodcall", "goodcall.com"}
    assert is_in_seed_list(canonical_name="Goodcall", homepage_url=None, seed_list=seed_list) is True
    assert is_in_seed_list(canonical_name="Other Co", homepage_url="https://goodcall.com/pricing", seed_list=seed_list) is True
    assert is_in_seed_list(canonical_name="Unrelated", homepage_url="https://unrelated.example.com/", seed_list=seed_list) is False


# --------------------------------------------------------------------------
# classify_entities -- the resolved hard lines: human-staffed answering
# services are IN (segment human_staffed); a local business named only as a
# usage example is OUT (end_user). Both come from the homepage, never from
# answer text.
# --------------------------------------------------------------------------

def test_classify_entities_resolves_human_staffed_in_and_end_user_out(tmp_path):
    conn = _fresh_conn(tmp_path)
    _seed_entities(conn, [
        ("goodcall", "Goodcall", "Goodcall", "https://goodcall.com/"),
        ("metro-answering", "Metro Answering Service", "Metro Answering Service", "https://metroanswering.example.com/"),
        ("dental", "Bright Smile Dental", "Bright Smile Dental", "https://brightsmiledental.example.com/"),
    ])
    inclusion_frame_path = tmp_path / "inclusion_frame.yaml"
    inclusion_frame_path.write_text("STUB frame")
    seed_list_path = tmp_path / "seed.csv"
    seed_list_path.write_text("canonical_name,homepage_domain\nGoodcall,goodcall.com\n")

    responses_by_name = {
        "Goodcall": GOODCALL_JSON, "Metro Answering Service": METRO_JSON, "Bright Smile Dental": DENTAL_JSON,
    }

    def fake_call_fn(*, canonical_name, homepage_url, homepage_text, inclusion_frame_text):
        return call_entity_classification(
            canonical_name=canonical_name, homepage_url=homepage_url, homepage_text=homepage_text,
            inclusion_frame_text=inclusion_frame_text,
            config=AnthropicConfig(api_key="sk-ant-test"),
            client=FakeAnthropicClient(FakeMessages(response=fake_classification_response(responses_by_name[canonical_name]))),
        )

    fetcher = FakeHomepageFetcher({
        "https://goodcall.com/": "AI receptionist for local businesses.",
        "https://metroanswering.example.com/": "Live human receptionists answering your business calls, no AI.",
        "https://brightsmiledental.example.com/": "Family dentistry. Call to book your appointment.",
    })
    cache = LLMCache(cache_dir=tmp_path / "cache")

    classify_entities(
        inclusion_frame_path=inclusion_frame_path, seed_list_path=seed_list_path,
        fetcher=fetcher, cache=cache, call_fn=fake_call_fn, conn=conn,
    )

    rows = {r["entity_id"]: r for r in storage.list_entities(conn=conn)}

    assert rows["goodcall"]["meets_frame"] == "in"
    assert rows["goodcall"]["segment"] == "done-for-you"
    assert rows["goodcall"]["in_seed_list"] is True

    assert rows["metro-answering"]["meets_frame"] == "in"
    assert rows["metro-answering"]["segment"] == "human_staffed"
    assert rows["metro-answering"]["in_seed_list"] is False

    assert rows["dental"]["meets_frame"] == "out"
    assert rows["dental"]["frame_reason"] == "end_user"
    assert rows["dental"]["segment"] is None


def test_classify_entities_verification_gate(tmp_path):
    conn = _fresh_conn(tmp_path)
    _seed_entities(conn, [
        ("sembly", "Sembly", "Sembly", None),           # real vendor, no homepage from A3
        ("g2", "G2", "G2", "https://www.g2.com/"),       # directory -> excluded
    ])
    inclusion_frame_path = tmp_path / "inclusion_frame.yaml"
    inclusion_frame_path.write_text("STUB frame")
    seed_list_path = tmp_path / "seed.csv"
    seed_list_path.write_text("canonical_name,homepage_domain\n")

    seen = []

    def fake_call_fn(*, canonical_name, homepage_url, homepage_text, inclusion_frame_text):
        seen.append((canonical_name, homepage_url))
        return call_entity_classification(
            canonical_name=canonical_name, homepage_url=homepage_url, homepage_text=homepage_text,
            inclusion_frame_text=inclusion_frame_text,
            config=AnthropicConfig(api_key="sk-ant-test"),
            client=FakeAnthropicClient(FakeMessages(response=fake_classification_response(GOODCALL_JSON))),
        )

    verification_by_entity = {
        "sembly": {"is_real": True, "entity_type": "vendor", "verified_homepage_url": "https://sembly.ai"},
        "g2": {"is_real": True, "entity_type": "directory", "verified_homepage_url": "https://www.g2.com"},
    }

    classify_entities(
        inclusion_frame_path=inclusion_frame_path, seed_list_path=seed_list_path,
        fetcher=FakeHomepageFetcher({"https://sembly.ai": "Sembly is an AI meeting assistant."}),
        cache=LLMCache(cache_dir=tmp_path / "cache"), call_fn=fake_call_fn,
        verification_by_entity=verification_by_entity, conn=conn,
    )

    # G2 skipped entirely; Sembly classified using the discovered homepage.
    assert seen == [("Sembly", "https://sembly.ai")]
    rows = {r["entity_id"]: r for r in storage.list_entities(conn=conn)}
    assert rows["g2"]["meets_frame"] == "out"
    assert rows["g2"]["frame_reason"] == "not_a_vendor"  # a3b entity_type "directory"
    assert rows["g2"]["segment"] is None
    assert rows["sembly"]["meets_frame"] == "in"


def test_classify_entities_uses_cache_on_second_call(tmp_path):
    conn = _fresh_conn(tmp_path)
    _seed_entities(conn, [("goodcall", "Goodcall", "Goodcall", "https://goodcall.com/")])
    inclusion_frame_path = tmp_path / "inclusion_frame.yaml"
    inclusion_frame_path.write_text("STUB frame")
    seed_list_path = tmp_path / "seed.csv"
    seed_list_path.write_text("canonical_name,homepage_domain\n")

    call_count = {"n": 0}

    def counting_call_fn(*, canonical_name, homepage_url, homepage_text, inclusion_frame_text):
        call_count["n"] += 1
        return call_entity_classification(
            canonical_name=canonical_name, homepage_url=homepage_url, homepage_text=homepage_text,
            inclusion_frame_text=inclusion_frame_text,
            config=AnthropicConfig(api_key="sk-ant-test"),
            client=FakeAnthropicClient(FakeMessages(response=fake_classification_response(GOODCALL_JSON))),
        )

    fetcher = FakeHomepageFetcher({"https://goodcall.com/": "AI receptionist."})
    cache = LLMCache(cache_dir=tmp_path / "cache")

    classify_entities(inclusion_frame_path=inclusion_frame_path, seed_list_path=seed_list_path,
                       fetcher=fetcher, cache=cache, call_fn=counting_call_fn, conn=conn)
    classify_entities(inclusion_frame_path=inclusion_frame_path, seed_list_path=seed_list_path,
                       fetcher=fetcher, cache=cache, call_fn=counting_call_fn, conn=conn)

    assert call_count["n"] == 1


# --------------------------------------------------------------------------
# Batch API path
# --------------------------------------------------------------------------

def _batch_fns(results_by_id, *, status="ended"):
    submitted: list[list[str]] = []

    def submit(items):
        submitted.append([i.custom_id for i in items])
        return "eb-1"

    def poll(batch_id, *, max_wait_seconds):
        if status is None:
            raise TimeoutError("still running")
        return status

    def fetch(batch_id):
        return results_by_id

    return (submit, poll, fetch), submitted


def _frame_and_seed(tmp_path, frame="STUB frame"):
    inclusion_frame_path = tmp_path / "inclusion_frame.yaml"
    inclusion_frame_path.write_text(frame)
    seed_list_path = tmp_path / "seed.csv"
    seed_list_path.write_text("canonical_name,homepage_domain\n")
    return inclusion_frame_path, seed_list_path


def test_submit_entity_classification_batch_uses_shared_request_body():
    batches = FakeAnthropicBatches(batch_id="eb-9")
    client = FakeAnthropicBatchClient(batches)
    bid = submit_entity_classification_batch(
        [EntityClassificationBatchItem("k1", "Sembly", "https://sembly.ai", "AI notes.", "FRAME")],
        config=AnthropicConfig(api_key="sk-ant-test"), client=client,
    )
    assert bid == "eb-9"
    req = batches.create_kwargs["requests"][0]
    assert req["custom_id"] == "k1"
    assert req["params"]["model"] == "claude-haiku-4-5-20251001"
    assert "Sembly" in req["params"]["messages"][0]["content"]


def test_fetch_entity_classification_batch_results_parses_success_and_error():
    batches = FakeAnthropicBatches(results=[
        fake_anthropic_batch_result_entry(custom_id="k_ok", text_json=GOODCALL_JSON),
        fake_anthropic_batch_error_entry(custom_id="k_bad"),
    ])
    client = FakeAnthropicBatchClient(batches)
    out = fetch_entity_classification_batch_results("eb-1", config=AnthropicConfig(api_key="sk-ant-test"), client=client)
    assert out["k_ok"].classification.meets_frame == "in"
    assert out["k_bad"].error == "batch_result_errored"


def test_classify_entities_batch_prefills_cache_and_skips_sync_calls(tmp_path):
    conn = _fresh_conn(tmp_path)
    _seed_entities(conn, [("goodcall", "Goodcall", "Goodcall", "https://goodcall.com/")])
    inclusion_frame_path, seed_list_path = _frame_and_seed(tmp_path)
    frame_text = inclusion_frame_path.read_text()
    fetcher = FakeHomepageFetcher({"https://goodcall.com/": "AI receptionist."})
    key = _classification_cache_key("Goodcall", "https://goodcall.com/", "AI receptionist.", frame_text)

    fns, submitted = _batch_fns({
        key: ClassificationResult(classification=ClassificationRow("in", None, "done-for-you", "https://goodcall.com/x")),
    })

    def no_sync(**_kw):
        raise AssertionError("sync call_fn must not run when the batch covered every entity")

    cache = LLMCache(cache_dir=tmp_path / "cache")
    classify_entities(
        inclusion_frame_path=inclusion_frame_path, seed_list_path=seed_list_path,
        fetcher=fetcher, cache=cache, call_fn=no_sync, use_batch=True, batch_fns=fns, conn=conn,
    )

    assert submitted[0] == [key]
    rows = {r["entity_id"]: r for r in storage.list_entities(conn=conn)}
    assert rows["goodcall"]["meets_frame"] == "in"
    assert not (cache.cache_dir / "_pending_entity_classification_batch.json").exists()


def test_classify_entities_batch_timeout_parks_batch_id_for_resume(tmp_path):
    conn = _fresh_conn(tmp_path)
    _seed_entities(conn, [("goodcall", "Goodcall", "Goodcall", "https://goodcall.com/")])
    inclusion_frame_path, seed_list_path = _frame_and_seed(tmp_path)
    frame_text = inclusion_frame_path.read_text()
    fetcher = FakeHomepageFetcher({"https://goodcall.com/": "AI receptionist."})
    key = _classification_cache_key("Goodcall", "https://goodcall.com/", "AI receptionist.", frame_text)
    cache = LLMCache(cache_dir=tmp_path / "cache")

    fns, _ = _batch_fns({}, status=None)  # poll raises TimeoutError
    with pytest.raises(TimeoutError):
        classify_entities(
            inclusion_frame_path=inclusion_frame_path, seed_list_path=seed_list_path,
            fetcher=fetcher, cache=cache, call_fn=lambda **k: None, use_batch=True, batch_fns=fns, conn=conn,
        )
    parked = json.loads((cache.cache_dir / "_pending_entity_classification_batch.json").read_text())
    assert parked["batch_id"] == "eb-1"

    fns2, submitted2 = _batch_fns({
        key: ClassificationResult(classification=ClassificationRow("in", None, "done-for-you", "https://goodcall.com/x")),
    })
    classify_entities(
        inclusion_frame_path=inclusion_frame_path, seed_list_path=seed_list_path,
        fetcher=fetcher, cache=cache, call_fn=lambda **k: None, use_batch=True, batch_fns=fns2, conn=conn,
    )
    assert submitted2 == []
    assert not (cache.cache_dir / "_pending_entity_classification_batch.json").exists()
