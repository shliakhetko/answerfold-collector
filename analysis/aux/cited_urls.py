"""Task 5 -- cited-URL resolution snapshot.

Link rot is the one thing this batch cannot reconstruct later. For every
unique URL that appears in any run's source list, resolve it now: HEAD then
GET, record the final status, the full redirect chain, the final registrable
domain, content type and page title, and store the page body (capped) for
anything that resolves 200 + text/HTML.

Gemini citations are `vertexaisearch.cloud.google.com/...redirect` wrappers;
following redirects yields the real publisher, captured from the last hop.

Status is bucketed but never collapsed: dead / http_404 / http_403 / timeout
/ paywalled stay distinct. No domain-type classification -- that is Phase A
(E1/E3/E4).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any, Iterator

from analysis.aggregate.response_fields import extract_sources_rows
from analysis.aux.htmlmeta import meta_robots_from_body, title_from_body
from analysis.aux.http import FetchResult, PoliteSession
from analysis.aux.storage import url_key

from collector import paths
DEFAULT_RUNS_DIR = paths.RUNS_DIR

# Structural paywall / access-wall signals -- recorded, never used to infer a
# domain class. HTTP 402 is the explicit one; the rest are common hard walls.
_PAYWALL_MARKERS = (
    b"paywall",
    b"subscribe to continue",
    b"subscription required",
    b"this content is for subscribers",
    b"metered",
)


def enumerate_cited_urls(runs_dir: Path | str = DEFAULT_RUNS_DIR) -> list[dict[str, Any]]:
    """Every distinct cited URL across all window runs, with the context of
    where it first appeared (engine, run_id, day). Order is stable
    (first-seen)."""
    seen: dict[str, dict[str, Any]] = {}
    for path in sorted(glob.glob(str(Path(runs_dir) / "*.jsonl"))):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            response = record.get("response")
            if response is None:
                continue
            engine = record.get("engine")
            for src in extract_sources_rows(engine, response):
                url = src.get("url_raw")
                if not url or url in seen:
                    continue
                seen[url] = {
                    "url": url,
                    "first_engine": engine,
                    "first_run_id": record.get("run_id"),
                    "first_day_index": record.get("day_index"),
                    "first_position": src.get("position"),
                    "first_title": src.get("title"),
                }
    return list(seen.values())


def status_bucket(head: FetchResult, get: FetchResult) -> str:
    """One of: ok, http_404, http_403, http_410, dead, timeout, paywalled,
    server_error, redirect_loop, other. Distinct on purpose."""
    primary = get if get.ok else head
    if not primary.ok:
        err = (get.error or head.error or "").lower()
        if "timeout" in err:
            return "timeout"
        if "too_many_redirects" in err or "redirect" in err:
            return "redirect_loop"
        return "dead"
    code = primary.status_code or 0
    if code == 404:
        return "http_404"
    if code == 403:
        return "http_403"
    if code == 410:
        return "http_410"
    if code == 402:
        return "paywalled"
    if 500 <= code < 600:
        return "server_error"
    if 200 <= code < 300:
        body = get.body_bytes or b""
        lowered = body[:200_000].lower()
        if any(marker in lowered for marker in _PAYWALL_MARKERS):
            return "paywalled"
        return "ok"
    return "other"


def resolve(session: PoliteSession, url: str) -> tuple[dict[str, Any], bytes | None]:
    """Returns (resolution fields, body bytes or None). The caller wraps the
    fields in the aux envelope and stores the body as a blob."""
    head = session.head(url)
    get = session.get(url)

    primary = get if get.ok else head
    content_type = primary.content_type
    is_texty = bool(content_type and ("html" in content_type.lower() or "text" in content_type.lower()))
    body = get.body_bytes if (get.ok and (get.status_code or 0) // 100 == 2 and is_texty) else None

    fields: dict[str, Any] = {
        "url": url,
        "url_key": url_key(url),
        "status_bucket": status_bucket(head, get),
        "head": head.meta(),
        "get": get.meta(),
        "final_url": primary.final_url,
        "final_status_code": primary.status_code,
        "final_registrable_domain": primary.final_registrable_host,
        "content_type": content_type,
        "page_title": title_from_body(body),
        "meta_robots": meta_robots_from_body(body),
        "x_robots_tag": get.headers.get("X-Robots-Tag") if get.ok else None,
        "redirect_chain": primary.redirect_chain,
        "body_stored": body is not None,
        "body_blob": f"cited_urls/bodies/{url_key(url)}.html" if body is not None else None,
    }
    return fields, body


def iter_batches(items: list[Any], size: int) -> Iterator[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
