"""The one place a URL or bare domain is reduced to a registrable domain
(eTLD+1). Moved here from analysis/phase_b/domain_ownership.py (2026-09-23)
so analysis/coding/cited_url_resolution.py -- a lower layer than phase_b,
which already depends on coding and never the reverse -- can reuse the
exact same reduction for Gemini's title-fallback fix. domain_ownership.py
re-exports these names rather than defining its own copy, so "the matching
rule lives here and nowhere else" (its own long-standing docstring claim)
stays true at the new location.
"""
from __future__ import annotations

import importlib.metadata
from urllib.parse import urlparse

import publicsuffix2


def psl_info() -> dict[str, str]:
    """Recorded as an input alongside the other frozen inputs -- the PSL
    snapshot is bundled with the installed package, fixed at install
    time, never fetched over the network."""
    version = "unknown"
    try:
        with open(publicsuffix2.ABOUT_PSL_FILE, encoding="utf-8") as f:
            for line in f:
                if line.startswith("version:"):
                    version = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        "package": "publicsuffix2",
        "package_version": importlib.metadata.version("publicsuffix2"),
        "psl_snapshot_version": version,
    }


def _normalize_host(url_or_domain: str | None) -> str | None:
    """Steps 1-4 of the matching rule, shared by both the final
    registrable-domain reduction below and the subdomain-vs-exact-host
    classification in domain_ownership.build_source_ownership: lowercase,
    drop scheme, drop a leading www. host prefix, drop port, drop path and
    query. Returns the bare host, not yet reduced to a registrable
    domain."""
    if not url_or_domain or not url_or_domain.strip():
        return None
    raw = url_or_domain.strip().lower()

    # urlparse only populates .netloc when the string looks like
    # //host/... or scheme://host/...; a bare "acme.com" or "www.acme.com"
    # (no scheme) lands entirely in .path otherwise. Force it into netloc
    # so scheme-less bare-domain input (alt_domains.csv's convention) and
    # full-URL input (homepage_url's convention, and every cited
    # source's url_raw) go through the identical code path below.
    parseable = raw if "//" in raw else "//" + raw
    try:
        host = urlparse(parseable).netloc
    except ValueError:
        # urlparse raises (rather than returning a harmless empty netloc)
        # on a few malformed shapes -- e.g. unbalanced "[" "]" read as an
        # attempted IPv6 literal. Surfaced 2026-09-22 by title text (free
        # text, not guaranteed domain-shaped) reaching this function for
        # the first time; every other caller here (url_raw, homepage_url,
        # alt_domain) is naturally well-formed enough to never have hit
        # it. Malformed input -> None is this function's documented
        # contract regardless of which caller triggers it.
        return None

    if "@" in host:  # userinfo, e.g. user:pass@host -- not expected in this
        host = host.rsplit("@", 1)[1]  # dataset, stripped defensively
    if ":" in host:  # drop port
        host = host.split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]

    if not host or " " in host or "\t" in host:
        return None  # no recognizable host -- malformed, not a domain
    return host


def registrable_domain(url_or_domain: str | None) -> str | None:
    """Step 5 of the matching rule on top of _normalize_host: reduce to
    the registrable domain (eTLD+1) via the pinned public suffix list.
    Applied identically regardless of which side (entity-owned domain, or
    cited source URL) calls it. Returns None for empty/unparseable input
    -- never coerced to an empty string or the input echoed back."""
    host = _normalize_host(url_or_domain)
    if host is None:
        return None
    sld = publicsuffix2.get_sld(host)
    if not sld or " " in sld:
        return None
    return sld
