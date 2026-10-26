from urllib.error import HTTPError

from analysis.coding.homepage_fetch import RealHomepageFetcher, html_to_text
from tests.fakes import FakeHTTPResponse, fake_opener


def test_html_to_text_strips_tags_and_collapses_whitespace():
    html = "<html><body>  <h1>Goodcall</h1>\n<p>AI   receptionist</p></body></html>"
    assert html_to_text(html) == "Goodcall AI receptionist"


def test_html_to_text_drops_script_and_style_contents():
    html = "<style>body{color:red}</style><p>Real text</p><script>evil()</script>"
    assert html_to_text(html) == "Real text"


def test_fetch_returns_plain_text_on_success():
    body = b"<html><body><h1>Goodcall</h1><p>AI receptionist for local businesses.</p></body></html>"
    fetcher = RealHomepageFetcher(opener=fake_opener(FakeHTTPResponse(body=body)))
    assert fetcher.fetch("https://goodcall.com/") == "Goodcall AI receptionist for local businesses."


def test_fetch_returns_empty_string_on_timeout():
    fetcher = RealHomepageFetcher(opener=fake_opener(exc=TimeoutError("timed out")))
    assert fetcher.fetch("https://slow.example.com/") == ""


def test_fetch_returns_empty_string_on_http_error():
    err = HTTPError("https://x.example.com/", 404, "Not Found", hdrs=None, fp=None)
    fetcher = RealHomepageFetcher(opener=fake_opener(exc=err))
    assert fetcher.fetch("https://x.example.com/") == ""


def test_fetch_returns_empty_string_for_non_html_content_type():
    fetcher = RealHomepageFetcher(
        opener=fake_opener(FakeHTTPResponse(body=b"%PDF-1.4 ...", content_type="application/pdf"))
    )
    assert fetcher.fetch("https://x.example.com/file.pdf") == ""

def test_fetch_truncates_to_max_chars():
    html = ("<p>" + "a" * 100 + "</p>").encode()
    fetcher = RealHomepageFetcher(max_chars=10, opener=fake_opener(FakeHTTPResponse(body=html)))
    result = fetcher.fetch("https://x.example.com/")
    assert result == "a" * 10


def test_fetch_never_raises_on_unexpected_exception():
    fetcher = RealHomepageFetcher(opener=fake_opener(exc=RuntimeError("something weird")))
    assert fetcher.fetch("https://x.example.com/") == ""
