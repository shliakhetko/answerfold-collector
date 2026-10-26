import json

import pytest

from analysis.coding import storage
from analysis.coding.a5_url_classification import (
    URLClassificationBatchItem,
    URLClassificationRow,
    _url_cache_key,
    call_url_classification,
    classify_urls,
    dedupe_sources_by_normalized_url,
    fetch_url_classification_batch_results,
    submit_url_classification_batch,
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
    fake_url_classification_response,
)


def _fresh_conn(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    init_db(db_path)
    init_coding_schema(db_path)
    return get_connection(db_path)

VENDOR_PAGE_JSON = json.dumps(
    {"url_class": "vendor page", "publisher_is_in_category": True, "classifier_confidence": 0.95}
)


# --------------------------------------------------------------------------
# dedupe_sources_by_normalized_url
# --------------------------------------------------------------------------

def test_dedupe_collapses_query_and_www_and_trailing_slash_variants():
    sources = [
        {"url_raw": "https://www.goodcall.com/pricing/?utm_source=chatgpt", "title": "Goodcall Pricing"},
        {"url_raw": "https://goodcall.com/pricing", "title": "Goodcall Pricing (dup)"},
    ]
    deduped = dedupe_sources_by_normalized_url(sources)
    assert list(deduped.keys()) == ["https://goodcall.com/pricing"]
    # First-seen title wins.
    assert deduped["https://goodcall.com/pricing"]["title"] == "Goodcall Pricing"


def test_dedupe_skips_sources_with_no_url():
    assert dedupe_sources_by_normalized_url([{"title": "no url"}]) == {}


# --------------------------------------------------------------------------
# call_url_classification
# --------------------------------------------------------------------------

def test_call_url_classification_happy_path():
    client = FakeAnthropicClient(FakeMessages(response=fake_url_classification_response(VENDOR_PAGE_JSON)))
    cfg = AnthropicConfig(api_key="sk-ant-test")

    result = call_url_classification(url_normalized="https://goodcall.com/", title="Goodcall", config=cfg, client=client)

    assert result.error is None
    assert result.classification == URLClassificationRow("vendor page", True, 0.95)


def test_call_url_classification_transport_failure_produces_error_not_exception():
    client = FakeAnthropicClient(FakeMessages(exc=TimeoutError("simulated timeout")))
    cfg = AnthropicConfig(api_key="sk-ant-test")

    result = call_url_classification(url_normalized="https://goodcall.com/", title=None, config=cfg, client=client)

    assert result.error == "TimeoutError: simulated timeout"
    assert result.classification is None


# --------------------------------------------------------------------------
# classify_urls
# --------------------------------------------------------------------------

def test_classify_urls_writes_one_row_per_unique_normalized_url(tmp_path):
    sources = [
        {"url_raw": "https://www.goodcall.com/?utm_source=chatgpt", "title": "Goodcall"},
        {"url_raw": "https://goodcall.com/", "title": "Goodcall dup"},  # same normalized URL
        {"url_raw": "https://metroanswering.example.com/", "title": "Metro"},
    ]

    def fake_call_fn(*, url_normalized, title):
        json_by_url = {
            "https://goodcall.com": VENDOR_PAGE_JSON,
            "https://metroanswering.example.com": json.dumps(
                {"url_class": "vendor page", "publisher_is_in_category": False, "classifier_confidence": 0.6}
            ),
        }
        return call_url_classification(
            url_normalized=url_normalized, title=title,
            config=AnthropicConfig(api_key="sk-ant-test"),
            client=FakeAnthropicClient(FakeMessages(response=fake_url_classification_response(json_by_url[url_normalized]))),
        )

    cache = LLMCache(cache_dir=tmp_path / "cache")
    conn = _fresh_conn(tmp_path)

    results = classify_urls(sources, cache=cache, call_fn=fake_call_fn, conn=conn)

    assert len(results) == 2  # deduped to 2 unique normalized URLs
    rows = {r["url_normalized"]: r for r in storage.list_url_registry(conn=conn)}
    assert set(rows) == {"https://goodcall.com", "https://metroanswering.example.com"}
    assert rows["https://goodcall.com"]["publisher_is_in_category"] == 1
    assert rows["https://metroanswering.example.com"]["publisher_is_in_category"] == 0


def test_classify_urls_skips_llm_for_resolution_dead_sources(tmp_path):
    sources = [
        {"url_raw": "https://goodcall.com/", "title": "Goodcall", "resolution_dead": False},
        {"url_raw": "https://metroanswering.example.com/gone", "title": "404", "resolution_dead": True},
    ]
    calls = {"n": 0}

    def fake_call_fn(*, url_normalized, title):
        calls["n"] += 1
        return call_url_classification(
            url_normalized=url_normalized, title=title,
            config=AnthropicConfig(api_key="sk-ant-test"),
            client=FakeAnthropicClient(FakeMessages(response=fake_url_classification_response(VENDOR_PAGE_JSON))),
        )

    conn = _fresh_conn(tmp_path)
    classify_urls(sources, cache=LLMCache(cache_dir=tmp_path / "cache"), call_fn=fake_call_fn, conn=conn)

    assert calls["n"] == 1  # only the live URL hit the model
    rows = {r["url_normalized"]: r for r in storage.list_url_registry(conn=conn)}
    assert rows["https://metroanswering.example.com/gone"]["url_class"] == "other"
    assert rows["https://metroanswering.example.com/gone"]["publisher_is_in_category"] == 0
    assert rows["https://metroanswering.example.com/gone"]["classifier_confidence"] == 0.0


def test_classify_urls_uses_cache_on_second_call(tmp_path):
    sources = [{"url_raw": "https://goodcall.com/", "title": "Goodcall"}]
    call_count = {"n": 0}

    def counting_call_fn(*, url_normalized, title):
        call_count["n"] += 1
        return call_url_classification(
            url_normalized=url_normalized, title=title,
            config=AnthropicConfig(api_key="sk-ant-test"),
            client=FakeAnthropicClient(FakeMessages(response=fake_url_classification_response(VENDOR_PAGE_JSON))),
        )

    cache = LLMCache(cache_dir=tmp_path / "cache")
    conn = _fresh_conn(tmp_path)
    classify_urls(sources, cache=cache, call_fn=counting_call_fn, conn=conn)
    classify_urls(sources, cache=cache, call_fn=counting_call_fn, conn=conn)

    assert call_count["n"] == 1


# --------------------------------------------------------------------------
# Batch API path
# --------------------------------------------------------------------------

def _batch_fns(results_by_id, *, status="ended"):
    submitted: list[list[str]] = []

    def submit(items):
        submitted.append([i.custom_id for i in items])
        return "ub-1"

    def poll(batch_id, *, max_wait_seconds):
        if status is None:
            raise TimeoutError("still running")
        return status

    def fetch(batch_id):
        return results_by_id

    return (submit, poll, fetch), submitted


def test_submit_url_classification_batch_uses_shared_request_body():
    batches = FakeAnthropicBatches(batch_id="ub-9")
    client = FakeAnthropicBatchClient(batches)
    bid = submit_url_classification_batch(
        [URLClassificationBatchItem("k1", "https://goodcall.com", "Goodcall")],
        config=AnthropicConfig(api_key="sk-ant-test"), client=client,
    )
    assert bid == "ub-9"
    req = batches.create_kwargs["requests"][0]
    assert req["custom_id"] == "k1"
    assert req["params"]["model"] == "claude-haiku-4-5-20251001"
    assert "https://goodcall.com" in req["params"]["messages"][0]["content"]


def test_fetch_url_classification_batch_results_parses_success_and_error():
    from tests.fakes import fake_anthropic_batch_error_entry

    batches = FakeAnthropicBatches(results=[
        fake_anthropic_batch_result_entry(custom_id="k_ok", text_json=VENDOR_PAGE_JSON),
        fake_anthropic_batch_error_entry(custom_id="k_bad"),
    ])
    client = FakeAnthropicBatchClient(batches)
    out = fetch_url_classification_batch_results("ub-1", config=AnthropicConfig(api_key="sk-ant-test"), client=client)
    assert out["k_ok"].classification.url_class == "vendor page"
    assert out["k_bad"].error == "batch_result_errored"


def test_classify_urls_batch_prefills_cache_and_skips_sync_calls(tmp_path):
    sources = [
        {"url_raw": "https://goodcall.com/", "title": "Goodcall"},
        {"url_raw": "https://metroanswering.example.com/", "title": "Metro"},
    ]
    from analysis.coding.a5_url_classification import URLClassificationResult

    key_gc = _url_cache_key("https://goodcall.com", "Goodcall")
    key_metro = _url_cache_key("https://metroanswering.example.com", "Metro")
    fns, submitted = _batch_fns({
        key_gc: URLClassificationResult(classification=URLClassificationRow("vendor page", True, 0.9)),
        key_metro: URLClassificationResult(classification=URLClassificationRow("review site", False, 0.7)),
    })

    def no_sync(**_kw):
        raise AssertionError("sync call_fn must not run when the batch covered every URL")

    cache = LLMCache(cache_dir=tmp_path / "cache")
    conn = _fresh_conn(tmp_path)
    classify_urls(sources, cache=cache, call_fn=no_sync, use_batch=True, batch_fns=fns, conn=conn)

    assert sorted(submitted[0]) == sorted([key_gc, key_metro])
    rows = {r["url_normalized"]: r for r in storage.list_url_registry(conn=conn)}
    assert rows["https://goodcall.com"]["url_class"] == "vendor page"
    assert rows["https://metroanswering.example.com"]["url_class"] == "review site"
    assert not (cache.cache_dir / "_pending_url_classification_batch.json").exists()


def test_classify_urls_batch_timeout_parks_batch_id_for_resume(tmp_path):
    from analysis.coding.a5_url_classification import URLClassificationResult

    sources = [{"url_raw": "https://goodcall.com/", "title": "Goodcall"}]
    key_gc = _url_cache_key("https://goodcall.com", "Goodcall")
    cache = LLMCache(cache_dir=tmp_path / "cache")
    conn = _fresh_conn(tmp_path)

    fns, _ = _batch_fns({}, status=None)  # poll raises TimeoutError
    with pytest.raises(TimeoutError):
        classify_urls(sources, cache=cache, call_fn=lambda **k: None, use_batch=True, batch_fns=fns, conn=conn)
    parked = json.loads((cache.cache_dir / "_pending_url_classification_batch.json").read_text())
    assert parked["batch_id"] == "ub-1"

    # resume: batch now done -> uses the parked id, does not re-submit
    fns2, submitted2 = _batch_fns(
        {key_gc: URLClassificationResult(classification=URLClassificationRow("vendor page", True, 0.9))}
    )
    classify_urls(sources, cache=cache, call_fn=lambda **k: None, use_batch=True, batch_fns=fns2, conn=conn)
    assert submitted2 == []
    assert not (cache.cache_dir / "_pending_url_classification_batch.json").exists()
