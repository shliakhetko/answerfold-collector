"""Tiny regex helpers to pull a handful of tags out of raw HTML bytes.

Deliberately not a parser and not a classifier: Tasks 4 and 5 store the full
raw body; these extract only the few fields that are cheap to surface now and
awkward to re-derive (the <title>, the page-level meta-robots directive).
Phase A does any real HTML analysis against the stored body.
"""
from __future__ import annotations

import re

_TITLE_RE = re.compile(rb"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_META_ROBOTS_RE = re.compile(rb"""<meta\s+[^>]*name\s*=\s*['"]robots['"][^>]*>""", re.IGNORECASE)
_CONTENT_ATTR_RE = re.compile(rb"""content\s*=\s*['"]([^'"]*)['"]""", re.IGNORECASE)


def title_from_body(body: bytes | None) -> str | None:
    if not body:
        return None
    m = _TITLE_RE.search(body)
    if not m:
        return None
    try:
        text = m.group(1).decode("utf-8", errors="replace")
    except Exception:
        return None
    text = re.sub(r"\s+", " ", text).strip()
    return text[:500] or None


def meta_robots_from_body(body: bytes | None) -> str | None:
    if not body:
        return None
    tag = _META_ROBOTS_RE.search(body)
    if not tag:
        return None
    content = _CONTENT_ATTR_RE.search(tag.group(0))
    if not content:
        return None
    return content.group(1).decode("utf-8", errors="replace").strip() or None
