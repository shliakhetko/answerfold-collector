import requests

from analysis.aux.http import PoliteSession
from tests.fakes import FakeRequestsResponse, FakeRequestsSession


def _session(routes=None, default=None):
    fake = FakeRequestsSession(routes=routes, default=default)
    return PoliteSession(session=fake, min_host_interval_seconds=0, capture_tls=False), fake


def test_get_captures_status_headers_body():
    ps, _ = _session(
        default=FakeRequestsResponse(
            url="https://x.com/",
            status_code=200,
            headers={"Content-Type": "text/html", "X-Robots-Tag": "noindex"},
            body=b"<title>Hi</title>",
        )
    )
    r = ps.get("https://x.com/")
    assert r.ok and r.status_code == 200
    assert r.headers["X-Robots-Tag"] == "noindex"
    assert r.body_bytes == b"<title>Hi</title>"
    assert r.content_type == "text/html"


def test_body_is_capped():
    ps, _ = _session(default=FakeRequestsResponse(url="https://x.com/", body=b"a" * 5000))
    ps.max_body_bytes = 100
    r = ps.get("https://x.com/")
    assert len(r.body_bytes) == 100
    assert r.body_truncated is True


def test_redirect_chain_recorded():
    hop = FakeRequestsResponse(url="https://x.com/old", status_code=301, headers={"Location": "https://x.com/new"})
    final = FakeRequestsResponse(url="https://x.com/new", status_code=200, body=b"ok", history=[hop])
    ps, _ = _session(default=final)
    r = ps.get("https://x.com/old")
    assert r.final_url == "https://x.com/new"
    assert r.redirect_chain == [{"url": "https://x.com/old", "status_code": 301, "location": "https://x.com/new"}]


def test_timeout_never_raises():
    ps, _ = _session(default=requests.exceptions.Timeout("slow"))
    r = ps.get("https://x.com/")
    assert r.ok is False
    assert r.error.startswith("timeout:")


def test_connection_error_never_raises():
    ps, _ = _session(default=requests.exceptions.ConnectionError("no route"))
    r = ps.head("https://x.com/")
    assert r.ok is False
    assert "ConnectionError" in r.error


def test_throttle_orders_same_host(monkeypatch):
    sleeps = []
    monkeypatch.setattr("analysis.aux.http.time.sleep", lambda s: sleeps.append(s))
    ps, _ = _session(default=FakeRequestsResponse(url="https://x.com/", body=b"ok"))
    ps.min_host_interval_seconds = 5
    ps.get("https://x.com/a")
    ps.get("https://x.com/b")
    assert sleeps and sleeps[0] > 0
