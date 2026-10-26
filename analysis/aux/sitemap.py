"""Task 6 -- company site inventory (one shallow pass per roster domain).

Fetch /sitemap.xml and /sitemap_index.xml, follow a sitemap index one level,
and store every sitemap XML verbatim. Record the URL list only -- do not
fetch the pages. When no sitemap exists, fall back to a shallow crawl from
the homepage, capped at depth 2 and a page limit, again recording only the
URL list.

No page-type classification (comparison / alternatives / pricing / vertical
pages) -- the URL list is the artifact; Phase A classifies.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import deque
from typing import Any
from urllib.parse import urljoin, urlsplit

from analysis.aux.http import PoliteSession
from analysis.aux.storage import domain_key

DEFAULT_CRAWL_PAGE_LIMIT = 200
DEFAULT_CRAWL_DEPTH = 2
MAX_SITEMAP_URLS = 50_000

_HREF_RE = re.compile(rb"""<a\b[^>]*\bhref\s*=\s*['"]([^'"#]+)['"]""", re.IGNORECASE)
_LOC_LOCALNAME = "loc"


def _base(domain: str) -> str:
    d = domain.strip().strip("/").lower()
    if d.startswith("http://") or d.startswith("https://"):
        return d.rstrip("/")
    return f"https://{d}"


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_sitemap_xml(data: bytes) -> tuple[str, list[str]]:
    """Returns (kind, locs) where kind is 'index' | 'urlset' | 'unknown'."""
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return "unknown", []
    kind = _localname(root.tag).lower()
    locs: list[str] = []
    for elem in root.iter():
        if _localname(elem.tag).lower() == _LOC_LOCALNAME and elem.text:
            locs.append(elem.text.strip())
    if "sitemapindex" in kind:
        return "index", locs
    if "urlset" in kind:
        return "urlset", locs
    return "unknown", locs


def _same_site(url: str, host: str) -> bool:
    try:
        h = (urlsplit(url).hostname or "").lower()
    except Exception:
        return False
    h = h[4:] if h.startswith("www.") else h
    host = host[4:] if host.startswith("www.") else host
    return h == host


def collect_from_sitemaps(session: PoliteSession, base: str) -> dict[str, Any]:
    host = (urlsplit(base).hostname or "").lower()
    candidates = [f"{base}/sitemap.xml", f"{base}/sitemap_index.xml"]
    fetched: list[dict[str, Any]] = []
    xml_blobs: dict[str, bytes] = {}
    all_urls: list[str] = []
    child_sitemaps: list[str] = []

    for idx, url in enumerate(candidates):
        res = session.get(url)
        entry = {
            "url": url,
            "ok": res.ok,
            "error": res.error,
            "status_code": res.status_code,
            "final_url": res.final_url,
            "content_type": res.content_type,
        }
        if res.body_bytes and (res.status_code or 0) // 100 == 2:
            xml_blobs[f"root_{idx}.xml"] = res.body_bytes
            kind, locs = _parse_sitemap_xml(res.body_bytes)
            entry["kind"] = kind
            entry["loc_count"] = len(locs)
            if kind == "index":
                child_sitemaps.extend(locs)
            else:
                all_urls.extend(locs)
        fetched.append(entry)

    # follow the sitemap index exactly one level
    for j, child_url in enumerate(dict.fromkeys(child_sitemaps)):
        res = session.get(child_url)
        entry = {
            "url": child_url,
            "level": 1,
            "ok": res.ok,
            "error": res.error,
            "status_code": res.status_code,
        }
        if res.body_bytes and (res.status_code or 0) // 100 == 2:
            xml_blobs[f"child_{j}.xml"] = res.body_bytes
            _, locs = _parse_sitemap_xml(res.body_bytes)
            entry["loc_count"] = len(locs)
            all_urls.extend(locs)
        fetched.append(entry)

    deduped = list(dict.fromkeys(u for u in all_urls if u))[:MAX_SITEMAP_URLS]
    return {
        "sitemap_exists": bool(xml_blobs) and any(f.get("loc_count") for f in fetched),
        "fetched": fetched,
        "xml_blobs": xml_blobs,
        "urls": deduped,
        "url_count": len(deduped),
        "truncated": len(all_urls) > MAX_SITEMAP_URLS,
    }


def shallow_crawl(
    session: PoliteSession,
    base: str,
    *,
    page_limit: int = DEFAULT_CRAWL_PAGE_LIMIT,
    max_depth: int = DEFAULT_CRAWL_DEPTH,
) -> dict[str, Any]:
    host = (urlsplit(base).hostname or "").lower()
    seen: set[str] = set()
    order: list[str] = []
    queue: deque[tuple[str, int]] = deque([(base, 0)])
    fetched_pages = 0

    while queue and fetched_pages < page_limit:
        url, depth = queue.popleft()
        if url in seen:
            continue
        seen.add(url)
        order.append(url)
        if depth >= max_depth:
            continue
        res = session.get(url)
        fetched_pages += 1
        if not res.body_bytes or (res.status_code or 0) // 100 != 2:
            continue
        for m in _HREF_RE.finditer(res.body_bytes[:400_000]):
            try:
                href = m.group(1).decode("utf-8", errors="replace")
            except Exception:
                continue
            absolute = urljoin(res.final_url or url, href)
            if absolute.startswith("http") and _same_site(absolute, host) and absolute not in seen:
                queue.append((absolute, depth + 1))

    return {
        "sitemap_exists": False,
        "pages_fetched": fetched_pages,
        "urls": order[:MAX_SITEMAP_URLS],
        "url_count": len(order),
    }


def inventory_domain(
    session: PoliteSession,
    domain: str,
    *,
    crawl_fallback: bool = True,
    page_limit: int = DEFAULT_CRAWL_PAGE_LIMIT,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    key = domain_key(domain)
    base = _base(domain)
    sm = collect_from_sitemaps(session, base)
    blobs = {f"site_inventory/raw_sitemaps/{key}/{name}": data for name, data in sm.pop("xml_blobs").items()}

    source = "sitemap"
    urls = sm["urls"]
    crawl_info: dict[str, Any] | None = None
    if not sm["sitemap_exists"] and crawl_fallback:
        crawl_info = shallow_crawl(session, base, page_limit=page_limit)
        urls = crawl_info["urls"]
        source = "crawl"

    fields = {
        "domain": key,
        "domain_input": domain,
        "base_url": base,
        "sitemap_exists": sm["sitemap_exists"],
        "source": source,
        "url_count": len(urls),
        "sitemap_detail": sm,
        "crawl_detail": crawl_info,
        "urls": urls,
    }
    return fields, blobs
