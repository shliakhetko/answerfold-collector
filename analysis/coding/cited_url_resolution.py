"""Overlay Task 5's cited-URL resolution snapshot onto the `sources` view the
coding pipeline consumes.

Gemini citations arrive as `vertexaisearch.cloud.google.com/grounding-api-
redirect/...` wrappers -- ~6.9k of them in this window, every one a distinct
URL that tells A5's classifier nothing and never matches a vendor domain in
B1's `own_domain_cited` join. The post-window auxiliary batch
(`scripts/aux/collect_cited_urls.py`) already followed every citation to its
final URL; this module reads that JSONL and, before A3/A5/B1/B2 see the
sources, swaps each *reachable* citation's `url_raw` / `domain` for its
resolved destination -- in memory only.

The collector `sources` table is never rewritten (the raw citation stays
canonical there). The full mapping is persisted to the coding-layer
`cited_url_resolution` table so the swap is inspectable / joinable.

Citations that resolved to nothing usable (404 / 410 / dead / timeout /
redirect loop) are flagged `resolution_dead`; A5 skips the LLM call for
those and records them as `other`.

**Gemini title fallback (2026-09-23).** A same-day-prior report-only
simulation (analysis/phase_b/domain_ownership.py's `_gemini_title_recovery`,
see reports/ambiguity_ledger.md) proved Gemini's grounding_chunks carry a
`title` field that reduces to the true resolved domain on 100% of already-
resolved Gemini citations (0 mismatches). Applied here, for real, upstream
of every consumer: any Gemini source still carrying the
`vertexaisearch...` wrapper after the normal swap above (whether or not
Task 5's capture ever reached it, and regardless of *why* it's
unreachable -- dead, paywalled, 403, timeout, never captured; the eligibility
condition is "the swap did not happen," not "the target is dead," since the
dead-bucket set alone covers only ~14% of the real unresolved population)
gets `url_raw`/`domain` rewritten from `registrable_domain(title)` when that
reduces to something usable, tagged `resolution_method = "title_fallback"`.
Checked empirically (not assumed) that no other engine has an equivalent
opacity problem: their raw citations are real URLs, never redirect
wrappers, so `registrable_domain(url_raw)` already gives a real publisher
domain on 100% of their citations with or without this fix -- there is
nothing for a title fallback to recover for them. See
`title_domain_match_rate_by_engine` below for the per-engine check that
converts this from an assumption into a stated, checked fact.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analysis.coding.registrable_domain import registrable_domain
from analysis.coding.url_normalize import normalize_url
from collector import paths

DEFAULT_RESOLUTION_PATH = paths.AUX_RESOLUTION_JSONL

# Gemini's raw citations are vertexaisearch.cloud.google.com redirect
# wrappers -- moved here (2026-09-23, from analysis/phase_b/domain_ownership.py)
# since this module is now where the wrapper-marker check that decides
# title-fallback eligibility lives; domain_ownership.py imports these two
# names from here rather than defining its own copy.
GEMINI_ENGINE = "google_gemini"
GEMINI_WRAPPER_MARKER = "vertexaisearch"

# status_bucket values from analysis/aux/cited_urls.py::status_bucket.
# "ok" is the only one whose final_url is a real fetched destination worth
# swapping in. These point nowhere usable now -- A5 should not pay to
# classify them:
DEAD_BUCKETS = frozenset({"http_404", "http_410", "dead", "timeout", "redirect_loop"})
# The rest (http_403, paywalled, server_error, other) keep the original
# cited URL and still go through A5 -- the publisher domain is real.


def _strip_www(host: str | None) -> str | None:
    if not host:
        return None
    host = host.strip().lower()
    return host[4:] if host.startswith("www.") else host


@dataclass(frozen=True)
class ResolvedCitation:
    cited_url: str
    final_url: str | None
    final_domain: str | None  # registrable host, www-stripped, lowercased
    status_bucket: str

    @property
    def is_reachable(self) -> bool:
        return self.status_bucket == "ok" and bool(self.final_url)

    @property
    def is_dead(self) -> bool:
        return self.status_bucket in DEAD_BUCKETS


@dataclass
class ResolutionStats:
    citations_total: int = 0
    citations_matched: int = 0  # had a row in the resolution snapshot
    citations_swapped: int = 0  # url_raw/domain replaced with the resolved destination
    citations_dead: int = 0  # flagged resolution_dead (A5 will skip)
    unique_urls_before: int = 0  # distinct normalize_url(url_raw) across all sources, pre-swap
    unique_urls_after: int = 0  # ... post-swap
    # Gemini title fallback (2026-09-23): citations recovered via
    # registrable_domain(title) after the normal swap above still left the
    # vertexaisearch wrapper in place. citations_title_fallback_attempted
    # counts every such candidate (any engine, though only Gemini is
    # actually eligible -- see GEMINI_ENGINE gating below);
    # citations_title_fallback_recovered is the subset where the title
    # reduced to a usable domain. citations_title_fallback_distinct_domains
    # is the count of *distinct* domains recovered -- always <=
    # citations_title_fallback_recovered, and the gap between them is the
    # known, reported side effect that every title-recovered citation to
    # the same domain collapses onto the same synthetic root URL (no real
    # path is known), so distinct domains, not distinct URLs, is the
    # honest "how much did this actually diversify the citation set"
    # number for this slice.
    citations_title_fallback_attempted: int = 0
    citations_title_fallback_recovered: int = 0
    citations_title_fallback_distinct_domains: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "citations_total": self.citations_total,
            "citations_matched": self.citations_matched,
            "citations_swapped": self.citations_swapped,
            "citations_dead": self.citations_dead,
            "unique_urls_before": self.unique_urls_before,
            "unique_urls_after": self.unique_urls_after,
            "citations_title_fallback_attempted": self.citations_title_fallback_attempted,
            "citations_title_fallback_recovered": self.citations_title_fallback_recovered,
            "citations_title_fallback_distinct_domains": self.citations_title_fallback_distinct_domains,
        }


def load_resolution_map(path: Path | None = DEFAULT_RESOLUTION_PATH) -> dict[str, ResolvedCitation]:
    """cited URL -> ResolvedCitation, from the Task 5 JSONL. Missing/None
    path -> {} (the pipeline then runs exactly as it did before this
    module existed). Last line wins on a duplicate URL (the collector is
    append-only + idempotent, so a rerun's fresh row shadows the old one)."""
    if path is None or not Path(path).exists():
        return {}

    out: dict[str, ResolvedCitation] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            url = record.get("url")
            if not url:
                continue
            final_domain = _strip_www(
                record.get("final_registrable_domain") or record.get("final_registrable_host")
            )
            out[url] = ResolvedCitation(
                cited_url=url,
                final_url=record.get("final_url"),
                final_domain=final_domain,
                status_bucket=record.get("status_bucket") or "other",
            )
    return out


def resolve_sources_by_run(
    sources_by_run: dict[str, list[dict[str, Any]]],
    resolution_map: dict[str, ResolvedCitation],
    *,
    run_engine_by_id: dict[str, str] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], ResolutionStats]:
    """Return a new sources_by_run where every reachable citation's
    `url_raw` / `domain` is its resolved final destination. Original
    citation is preserved on `url_cited`; `resolution_status` /
    `resolution_dead` are added to every row. Pure -- inputs untouched.

    `run_engine_by_id` (run_id -> engine) is optional and keyword-only,
    defaulted to None so every existing caller (scripts/coding/run_coding_pipeline.py,
    scripts/coding/close_out_a5_url_classification.py, this module's own tests)
    keeps working unchanged, with the Gemini title fallback below simply
    never firing when no engine map is supplied. Only
    analysis/phase_b/reading.py passes it today.

    When engine info is available, any Gemini source still carrying the
    vertexaisearch wrapper after the swap above gets a second chance via
    `registrable_domain(title)` -- see this module's docstring for why the
    eligibility check is "the swap didn't happen," not "the target is
    dead."""
    stats = ResolutionStats()
    before: set[str] = set()
    after: set[str] = set()
    recovered_domains: set[str] = set()
    resolved: dict[str, list[dict[str, Any]]] = {}

    for run_id, sources in sources_by_run.items():
        engine = run_engine_by_id.get(run_id) if run_engine_by_id else None
        new_sources: list[dict[str, Any]] = []
        for source in sources:
            stats.citations_total += 1
            cited_url = source.get("url_raw", "") or ""
            if cited_url:
                before.add(normalize_url(cited_url))

            row = dict(source)
            row["url_cited"] = cited_url or None
            row["resolution_method"] = None

            resolution = resolution_map.get(cited_url) if cited_url else None
            if resolution is not None:
                stats.citations_matched += 1
                row["resolution_status"] = resolution.status_bucket
                row["resolution_dead"] = resolution.is_dead
                if resolution.is_dead:
                    stats.citations_dead += 1
                if resolution.is_reachable:
                    row["url_raw"] = resolution.final_url
                    if resolution.final_domain:
                        row["domain"] = resolution.final_domain
                    stats.citations_swapped += 1
            else:
                row["resolution_status"] = None
                row["resolution_dead"] = False

            if engine == GEMINI_ENGINE and GEMINI_WRAPPER_MARKER in (row.get("url_raw") or ""):
                stats.citations_title_fallback_attempted += 1
                title_domain = registrable_domain(source.get("title"))
                if title_domain:
                    row["url_raw"] = f"https://{title_domain}/"
                    row["domain"] = title_domain
                    row["resolution_method"] = "title_fallback"
                    stats.citations_title_fallback_recovered += 1
                    recovered_domains.add(title_domain)

            effective = row.get("url_raw", "") or ""
            if effective:
                after.add(normalize_url(effective))
            new_sources.append(row)
        resolved[run_id] = new_sources

    stats.unique_urls_before = len(before)
    stats.unique_urls_after = len(after)
    stats.citations_title_fallback_distinct_domains = len(recovered_domains)
    return resolved, stats


def title_domain_match_rate_by_engine(
    sources_by_run: dict[str, list[dict[str, Any]]],
    run_engine_by_id: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Per engine: among citations whose `url_raw` is a real, resolved
    publisher URL (i.e. NOT still carrying the Gemini wrapper marker --
    a row with no known ground truth), what share of `title` fields
    reduce, via `registrable_domain`, to that same domain? Converts the
    "does this engine's title field hold a domain, the way Gemini's
    grounding_chunks title does" question from a docstring assumption
    into a checked fact for all four engines, not just Gemini. Call this
    on an already-resolved sources_by_run (post resolve_sources_by_run) --
    it does not itself resolve anything.

    Expected result on real data: near-100% for google_gemini (already
    known, validated against 6,632 citations before this fix existed --
    see analysis/phase_b/domain_ownership.py's _gemini_title_recovery),
    low for the other three engines, whose `title` is a real page title
    (free text), not a domain. A low match rate for a non-Gemini engine
    is not a bug -- it's the expected, now-confirmed-rather-than-assumed
    reason no other engine gets this fallback wired in."""
    checked: dict[str, int] = {}
    matched: dict[str, int] = {}
    for run_id, sources in sources_by_run.items():
        engine = run_engine_by_id.get(run_id)
        if engine is None:
            continue
        for source in sources:
            url_raw = source.get("url_raw") or ""
            if not url_raw or GEMINI_WRAPPER_MARKER in url_raw:
                continue  # no known ground truth to check title against
            if source.get("resolution_method") == "title_fallback":
                continue  # url_raw was itself derived from title -- circular, not independent ground truth
            title = source.get("title")
            if not title:
                continue
            actual_domain = registrable_domain(url_raw)
            if actual_domain is None:
                continue
            checked[engine] = checked.get(engine, 0) + 1
            if registrable_domain(title) == actual_domain:
                matched[engine] = matched.get(engine, 0) + 1

    return {
        engine: {
            "checked": n_checked,
            "matched": matched.get(engine, 0),
            "match_rate": (matched.get(engine, 0) / n_checked) if n_checked else None,
        }
        for engine, n_checked in checked.items()
    }
