from analysis.aux.directories import check_directory, slugify
from analysis.aux.http import PoliteSession
from analysis.aux.robots_snapshot import snapshot_domain
from analysis.aux.sitemap import _parse_sitemap_xml, inventory_domain
from tests.fakes import FakeRequestsResponse, FakeRequestsSession


def _ps(routes=None, default=None):
    return PoliteSession(
        session=FakeRequestsSession(routes=routes, default=default),
        min_host_interval_seconds=0,
        capture_tls=False,
    )


# -- directories -----------------------------------------------------

def test_slugify():
    assert slugify("Smith.ai") == "smith-ai"
    assert slugify("tl;dv") == "tl-dv"


def test_directory_found_stores_body():
    routes = {"g2.com/products/goodcall/reviews": FakeRequestsResponse(url="https://www.g2.com/products/goodcall/reviews", status_code=200, body=b"<h1>Goodcall reviews</h1>")}
    ps = _ps(routes=routes, default=FakeRequestsResponse(url="x", status_code=404))
    fields, body = check_directory(ps, "g2", canonical_name="Goodcall", primary_domain="goodcall.com")
    assert fields["outcome"] == "found"
    assert body is not None
    assert fields["profile_url"].endswith("/reviews")


def test_directory_absent_is_reproducible():
    ps = _ps(default=FakeRequestsResponse(url="x", status_code=404))
    fields, body = check_directory(ps, "crunchbase", canonical_name="Nope Inc", primary_domain=None)
    assert fields["outcome"] == "absent"
    assert body is None
    assert fields["search_method"]  # the templates tried are recorded


def test_directory_blocked_on_challenge_page():
    routes = {"trustpilot.com": FakeRequestsResponse(url="https://www.trustpilot.com/review/x.com", status_code=200, body=b"Just a moment... cf-browser-verification")}
    ps = _ps(routes=routes)
    fields, _ = check_directory(ps, "trustpilot", canonical_name="X", primary_domain="x.com")
    assert fields["outcome"] == "blocked"


# -- sitemap --------------------------------------------------------

SITEMAP_INDEX = b"""<?xml version="1.0"?><sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
<sitemap><loc>https://x.com/sitemap-pages.xml</loc></sitemap></sitemapindex>"""
URLSET = b"""<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
<url><loc>https://x.com/a</loc></url><url><loc>https://x.com/b</loc></url></urlset>"""


def test_parse_sitemap_xml_kinds():
    assert _parse_sitemap_xml(SITEMAP_INDEX) == ("index", ["https://x.com/sitemap-pages.xml"])
    assert _parse_sitemap_xml(URLSET)[0] == "urlset"


def test_inventory_follows_index_one_level():
    routes = {
        "https://x.com/sitemap.xml": FakeRequestsResponse(url="https://x.com/sitemap.xml", status_code=200, headers={"Content-Type": "application/xml"}, body=SITEMAP_INDEX),
        "https://x.com/sitemap_index.xml": FakeRequestsResponse(url="x", status_code=404),
        "https://x.com/sitemap-pages.xml": FakeRequestsResponse(url="https://x.com/sitemap-pages.xml", status_code=200, body=URLSET),
    }
    ps = _ps(routes=routes, default=FakeRequestsResponse(url="x", status_code=404))
    fields, blobs = inventory_domain(ps, "x.com", crawl_fallback=False)
    assert fields["sitemap_exists"] is True
    assert set(fields["urls"]) == {"https://x.com/a", "https://x.com/b"}
    assert any("raw_sitemaps" in k for k in blobs)


def test_inventory_crawl_fallback_when_no_sitemap():
    home = FakeRequestsResponse(url="https://x.com", status_code=200, body=b'<a href="/about">about</a><a href="https://other.com/x">x</a>')
    routes = {"https://x.com/sitemap": FakeRequestsResponse(url="x", status_code=404)}
    ps = _ps(routes=routes, default=home)
    fields, _ = inventory_domain(ps, "x.com", crawl_fallback=True, page_limit=5)
    assert fields["source"] == "crawl"
    assert "https://x.com/about" in fields["urls"]
    assert all("other.com" not in u for u in fields["urls"])  # same-site only


# -- robots ---------------------------------------------------------

def test_snapshot_domain_records_status_for_missing_files():
    routes = {
        "https://x.com/robots.txt": FakeRequestsResponse(url="https://x.com/robots.txt", status_code=200, headers={"Content-Type": "text/plain"}, body=b"User-agent: *\nDisallow:"),
        "https://x.com/llms.txt": FakeRequestsResponse(url="x", status_code=404),
    }
    ps = _ps(routes=routes, default=FakeRequestsResponse(url="x", status_code=404, body=b"<html><meta name=robots content='noindex'></html>"))
    fields, blobs = snapshot_domain(ps, "x.com")
    assert fields["well_known"]["robots.txt"]["status_code"] == 200
    assert fields["well_known"]["llms.txt"]["status_code"] == 404
    assert "robots/x.com/robots.txt" in blobs
    # robots.txt content is stored raw, not parsed into booleans
    assert blobs["robots/x.com/robots.txt"] == b"User-agent: *\nDisallow:"
