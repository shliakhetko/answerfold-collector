"""Task 4 -- crawler eligibility snapshot.

For every roster domain, fetch and store verbatim: /robots.txt, /llms.txt,
/llms-full.txt, /ai.txt (recording the HTTP status even on 404), plus the
homepage HTML with its meta-robots tag, X-Robots-Tag header, full redirect
chain, final URL, TLS issuer/expiry, and response headers.

robots.txt is NOT parsed into allow/block here -- stored raw. The
AI_CRAWLER_AGENTS list (analysis/aux/crawler_agents.py) is written into
data/02_aux/collector_config.json so the later parse is reproducible.
"""
from __future__ import annotations

from typing import Any

from analysis.aux.htmlmeta import meta_robots_from_body, title_from_body
from analysis.aux.http import PoliteSession
from analysis.aux.storage import domain_key

WELL_KNOWN_FILES = ("robots.txt", "llms.txt", "llms-full.txt", "ai.txt")


def _candidate_base_urls(domain: str) -> list[str]:
    d = domain.strip().strip("/").lower()
    if d.startswith("http://") or d.startswith("https://"):
        return [d.rstrip("/")]
    return [f"https://{d}"]


def snapshot_domain(session: PoliteSession, domain: str) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Returns (record fields, {relpath under data/02_aux/robots/<key>/: bytes}).
    Never raises."""
    key = domain_key(domain)
    base = _candidate_base_urls(domain)[0]
    blobs: dict[str, bytes] = {}

    well_known: dict[str, Any] = {}
    for name in WELL_KNOWN_FILES:
        res = session.get(f"{base}/{name}")
        well_known[name] = {
            "requested_url": f"{base}/{name}",
            "ok": res.ok,
            "error": res.error,
            "status_code": res.status_code,
            "final_url": res.final_url,
            "content_type": res.content_type,
            "redirect_chain": res.redirect_chain,
            "bytes_len": None if res.body_bytes is None else len(res.body_bytes),
            "body_truncated": res.body_truncated,
            "stored": res.body_bytes is not None and bool(res.body_bytes),
        }
        if res.body_bytes:
            blobs[f"robots/{key}/{name}"] = res.body_bytes

    home = session.get(base)
    if home.body_bytes:
        blobs[f"robots/{key}/homepage.html"] = home.body_bytes
    homepage = {
        "requested_url": base,
        "ok": home.ok,
        "error": home.error,
        "status_code": home.status_code,
        "final_url": home.final_url,
        "redirect_chain": home.redirect_chain,
        "headers": home.headers,
        "content_type": home.content_type,
        "x_robots_tag": home.headers.get("X-Robots-Tag"),
        "meta_robots": meta_robots_from_body(home.body_bytes),
        "title": title_from_body(home.body_bytes),
        "tls": home.tls,
        "body_truncated": home.body_truncated,
        "stored": "robots/{}/homepage.html".format(key) in blobs,
    }

    fields = {
        "domain": domain_key(domain),
        "domain_input": domain,
        "base_url": base,
        "well_known": well_known,
        "homepage": homepage,
        "blob_dir": f"robots/{key}/",
    }
    return fields, blobs
