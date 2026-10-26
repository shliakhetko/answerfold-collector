"""url_normalized (Post-Collection Coding Pipeline Spec, A5): "strip UTM and
query, drop trailing slash, lowercase host, remove www." -- computed
deterministically before A5, and reused by B1 for the own_domain_cited join.

Interpreted here as: drop the query string entirely (which necessarily
removes UTM params along with everything else -- the frozen doc this spec
references doesn't spell out which non-UTM params, if any, would be kept,
so dropping the whole query is the only reading that needs no further
guessing) and drop any fragment.
"""
from __future__ import annotations

from urllib.parse import urlparse, urlunparse


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path.rstrip("/")
    return urlunparse((parsed.scheme, host, path, "", "", ""))
