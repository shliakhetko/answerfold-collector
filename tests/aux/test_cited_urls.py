import json

from analysis.aux.cited_urls import enumerate_cited_urls, resolve, status_bucket
from analysis.aux.http import PoliteSession
from tests.fakes import FakeRequestsResponse, FakeRequestsSession


def _write_run(path, engine, response):
    path.write_text(json.dumps({"run_id": f"{engine}-i1-p1-d1-r1", "ts_utc": "2026-08-20T14:00:00+00:00", "engine": engine, "day_index": 1, "response": response}) + "\n")


def test_enumerate_dedupes_across_engines(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_run(
        runs / "2026-08-20.jsonl",
        "anthropic_claude",
        {"content": [{"type": "text", "citations": [{"type": "web_search_result_location", "url": "https://a.com", "title": "A"}]}]},
    )
    _write_run(
        runs / "2026-08-21.jsonl",
        "google_ai_mode",
        {"references": [{"link": "https://a.com"}, {"link": "https://b.com"}]},
    )
    urls = enumerate_cited_urls(runs)
    assert [u["url"] for u in urls] == ["https://a.com", "https://b.com"]
    assert urls[0]["first_engine"] == "anthropic_claude"


def test_status_bucket_distinguishes_failure_modes():
    def rr(code):
        return FakeRequestsResponse(url="https://x.com/", status_code=code, body=b"<html></html>")

    ps = PoliteSession(session=FakeRequestsSession(), min_host_interval_seconds=0, capture_tls=False)
    ok = ps.get("https://x.com/")  # fake default 404
    assert status_bucket(ok, ok) == "http_404"

    from analysis.aux.http import FetchResult

    good = FetchResult(url="u", ok=True, status_code=200, body_bytes=b"<html>hello</html>")
    assert status_bucket(good, good) == "ok"
    f403 = FetchResult(url="u", ok=True, status_code=403)
    assert status_bucket(f403, f403) == "http_403"
    dead = FetchResult(url="u", ok=False, error="ConnectionError: nope")
    assert status_bucket(dead, dead) == "dead"
    to = FetchResult(url="u", ok=False, error="timeout: slow")
    assert status_bucket(to, to) == "timeout"
    wall = FetchResult(url="u", ok=True, status_code=200, body_bytes=b"please subscribe to continue reading")
    assert status_bucket(wall, wall) == "paywalled"


def test_resolve_stores_body_only_for_200_html(tmp_path):
    routes = {
        "https://ok.com": FakeRequestsResponse(url="https://ok.com/", status_code=200, headers={"Content-Type": "text/html"}, body=b"<title> Page </title>"),
    }
    ps = PoliteSession(session=FakeRequestsSession(routes=routes), min_host_interval_seconds=0, capture_tls=False)
    fields, body = resolve(ps, "https://ok.com")
    assert fields["status_bucket"] == "ok"
    assert fields["page_title"] == "Page"
    assert body == b"<title> Page </title>"
    assert fields["body_blob"].endswith(".html")
