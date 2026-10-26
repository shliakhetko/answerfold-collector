"""Task 7 -- directory presence snapshot.

For every roster company, check G2, Capterra, TrustRadius, Trustpilot,
Product Hunt and Crunchbase: does a profile exist, and capture the raw page
HTML. Profile URLs are constructed directly from a slug of the company name
(and the domain, for Trustpilot) and then HTTP-verified -- no SERP-assisted
discovery (per the user's "35 calls, minimal" decision).

Outcomes are recorded distinctly and reproducibly:
  found    -- 2xx at a constructed URL; raw HTML stored
  absent   -- 404/410 at every constructed URL; `search_method` lists them
  blocked  -- 403 / challenge page / connection blocked; move on, no workaround
  error    -- transport failure

No review-count or rating extraction -- the page is the artifact.
"""
from __future__ import annotations

import re
from typing import Any

from analysis.aux.http import PoliteSession
from analysis.aux.storage import domain_key

_SLUG_RE = re.compile(r"[^a-z0-9]+")

# {directory: [url templates]} -- {slug} = name slug, {domain} = primary
# registrable domain. Multiple templates are tried in order; the first 2xx
# wins. Where a directory's profile URL needs an internal numeric id that
# cannot be constructed (Capterra), only a best-effort path is tried and a
# miss is recorded as `absent` with the templates tried, so the absence is
# reproducible.
DIRECTORY_TEMPLATES: dict[str, list[str]] = {
    "g2": [
        "https://www.g2.com/products/{slug}/reviews",
        "https://www.g2.com/products/{slug}",
    ],
    "capterra": [
        "https://www.capterra.com/p/{slug}/",
        "https://www.capterra.com/search/?query={name_q}",
    ],
    "trustradius": [
        "https://www.trustradius.com/products/{slug}/reviews",
    ],
    "trustpilot": [
        "https://www.trustpilot.com/review/{domain}",
    ],
    "producthunt": [
        "https://www.producthunt.com/products/{slug}",
    ],
    "crunchbase": [
        "https://www.crunchbase.com/organization/{slug}",
    ],
}

DIRECTORIES = tuple(DIRECTORY_TEMPLATES.keys())

_BLOCK_MARKERS = (
    b"just a moment",
    b"cf-browser-verification",
    b"attention required",
    b"px-captcha",
    b"access denied",
    b"request unsuccessful",
)


def slugify(name: str) -> str:
    return _SLUG_RE.sub("-", name.strip().lower()).strip("-")


def _fill(template: str, *, slug: str, name: str, domain: str | None) -> str:
    from urllib.parse import quote

    return template.format(slug=slug, name_q=quote(name), domain=domain or slug)


def _classify(res: Any) -> str:
    if not res.ok:
        err = (res.error or "").lower()
        if "timeout" in err:
            return "error"
        return "error"
    code = res.status_code or 0
    if 200 <= code < 300:
        body = (res.body_bytes or b"")[:20_000].lower()
        if any(m in body for m in _BLOCK_MARKERS):
            return "blocked"
        return "found"
    if code in (401, 403, 429):
        return "blocked"
    if code in (404, 410):
        return "absent"
    if 500 <= code < 600:
        return "blocked"
    return "absent"


def check_directory(
    session: PoliteSession,
    directory: str,
    *,
    canonical_name: str,
    primary_domain: str | None,
) -> tuple[dict[str, Any], bytes | None]:
    slug = slugify(canonical_name)
    templates = DIRECTORY_TEMPLATES[directory]
    tried: list[dict[str, Any]] = []
    body: bytes | None = None
    outcome = "absent"
    profile_url: str | None = None

    for template in templates:
        url = _fill(template, slug=slug, name=canonical_name, domain=domain_key(primary_domain) if primary_domain else None)
        res = session.get(url)
        this = _classify(res)
        tried.append(
            {
                "url": url,
                "status_code": res.status_code,
                "final_url": res.final_url,
                "error": res.error,
                "outcome": this,
            }
        )
        if this == "found":
            outcome = "found"
            profile_url = res.final_url or url
            body = res.body_bytes
            break
        if this == "blocked":
            outcome = "blocked"
            body = res.body_bytes
            profile_url = res.final_url or url
            break
        if this == "error" and outcome == "absent":
            outcome = "error"

    fields = {
        "directory": directory,
        "outcome": outcome,
        "profile_url": profile_url,
        "search_method": [t["url"] for t in tried],
        "attempts": tried,
        "body_stored": body is not None and outcome in {"found", "blocked"},
    }
    return fields, (body if outcome in {"found", "blocked"} else None)
