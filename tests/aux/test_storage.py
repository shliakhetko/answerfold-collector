from analysis.aux.storage import AuxStore, domain_key, url_key


def test_append_and_read_records(tmp_path):
    store = AuxStore(tmp_path)
    idx = store.path("cited_urls", "resolution.jsonl")
    store.append_record(idx, {"url": "https://a.com", "status_bucket": "ok"})
    store.append_record(idx, {"url": "https://b.com", "status_bucket": "http_404"})
    rows = store.read_records(idx)
    assert [r["url"] for r in rows] == ["https://a.com", "https://b.com"]


def test_done_keys_powers_idempotency(tmp_path):
    store = AuxStore(tmp_path)
    idx = store.path("x", "index.jsonl")
    store.append_record(idx, {"url": "https://a.com"})
    store.append_record(idx, {"url": "https://b.com"})
    done = store.done_keys(idx, lambda r: r["url"])
    assert done == {"https://a.com", "https://b.com"}
    remaining = [u for u in ["https://a.com", "https://c.com"] if u not in done]
    assert remaining == ["https://c.com"]


def test_done_keys_empty_when_no_file(tmp_path):
    store = AuxStore(tmp_path)
    assert store.done_keys(store.path("nope.jsonl"), lambda r: r["url"]) == set()


def test_write_helpers_create_parents(tmp_path):
    store = AuxStore(tmp_path)
    p = store.write_bytes("robots/example.com/robots.txt", b"User-agent: *\n")
    assert p.read_bytes() == b"User-agent: *\n"
    j = store.write_json("google_serp/i1-p1.json", {"a": 1})
    assert j.exists()


def test_domain_key_normalizes():
    assert domain_key("https://www.Example.com/") == "example.com"
    assert domain_key("Example.com") == "example.com"


def test_url_key_is_stable_and_hex():
    k = url_key("https://a.com/x")
    assert k == url_key("https://a.com/x")
    assert len(k) == 64
