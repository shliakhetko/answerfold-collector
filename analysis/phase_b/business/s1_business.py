"""Business figures for S1: source mix, top vendor-published domains and the
self-reference split, per engine.

Exploratory and derived. It reads the same citation panel S1 reads, so the
vendor-published count is S1's (publisher_is_in_category True after the
domain-ownership override) and the display classes of source_classes.py only
relabel the non-vendor rest. Nothing here feeds or changes figures.json.

Self-reference split of the vendor-published citations. "Named in that
answer" is exact: the owned domains of the in-frame companies mentioned in the
same run (citation_panel.mentioned_own_domains_by_run).
* own_page: a page on the domain of a company named in that answer;
* own_listicle: same, but A5 labelled the page a vendor listicle (the
  company's own listicle or blog post);
* other_vendor: vendor-published content of a company that is not named in
  that answer.
The three sum to the engine's vendor-published citation count.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from analysis.phase_b.business import source_classes as sc
from analysis.phase_b.business.extremes import extreme, ordered
from analysis.phase_b.citation_panel import CitationRow, build_citation_panel, mentioned_own_domains_by_run
from analysis.phase_b.reading import FrozenInputs

TOP_DOMAINS_N = 10
OWN_PAGE = "own_page"
OWN_LISTICLE = "own_listicle"
OTHER_VENDOR = "other_vendor"
SELF_REFERENCE_PARTS = (OWN_PAGE, OWN_LISTICLE, OTHER_VENDOR)
VENDOR_LISTICLE_LABEL = "vendor listicle"

NOTE = (
    "Exploratory and derived; separate from figures.json. Source-mix classes other than Vendor-published are a "
    "display relabel of the non-vendor citations (config/source_domain_classes.csv); the vendor-published count "
    "is S1's."
)


def self_reference_part(row: CitationRow, mentioned_own_domains: dict[str, set[str]]) -> str:
    """Which self-reference bucket a vendor-published citation falls into."""
    if sc.row_domain(row) in mentioned_own_domains.get(row.run_id, set()):
        return OWN_LISTICLE if row.url_class == VENDOR_LISTICLE_LABEL else OWN_PAGE
    return OTHER_VENDOR


def _engine_block(
    engine: str,
    rows: list[CitationRow],
    domain_map: dict[str, str],
    mentioned_own_domains: dict[str, set[str]],
) -> dict[str, Any]:
    classified = [r for r in rows if r.url_class is not None]
    n = len(classified)

    mix_counts: Counter = Counter()
    for r in classified:
        mix_counts[sc.display_class(r, domain_map)] += 1
    source_mix = {
        cls: {"count": mix_counts.get(cls, 0), "share": mix_counts.get(cls, 0) / n} for cls in sc.DISPLAY_CLASSES
    }

    domain_total: Counter = Counter()
    domain_self: Counter = Counter()
    parts: Counter = Counter()
    for r in classified:
        if r.publisher_is_in_category is not True:
            continue
        part = self_reference_part(r, mentioned_own_domains)
        parts[part] += 1
        domain = sc.row_domain(r)
        domain_total[domain] += 1
        if part != OTHER_VENDOR:
            domain_self[domain] += 1
    vendor_count = sum(parts.values())

    top = sorted(domain_total.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_DOMAINS_N]
    return {
        "main_result": engine in sc.MAIN_ENGINES,
        "classified_citations": n,
        "source_mix": source_mix,
        "vendor_published": {
            "count": vendor_count,
            "share": vendor_count / n if n else None,
            "distinct_domains": len(domain_total),
            "top_domains": [
                {"domain": d, "count": c, "self_reference_count": domain_self.get(d, 0)} for d, c in top
            ],
            "self_reference": {
                part: {"count": parts.get(part, 0), "share": parts.get(part, 0) / vendor_count if vendor_count else None}
                for part in SELF_REFERENCE_PARTS
            },
        },
    }


# Claim logic, not prose: the S1 text says the Google engines differ from the others, and that OpenAI ChatGPT
# leads on self-reference. The flags below turn false if the data stops saying so.
GOOGLE_ENGINES = ("google_ai_mode", "google_gemini")


def derive(engines: dict[str, Any], leftover: dict[str, Any], study: dict[str, int] | None = None) -> dict[str, Any]:
    """Every number and claim flag the user prose of S1 needs, so the template computes nothing.
    All values come from the engine blocks above; shares are fractions, as stored."""
    names = ordered(engines)
    mix = {e: engines[e]["source_mix"] for e in names}
    vendor = {e: engines[e]["vendor_published"] for e in names}
    own_share = {
        e: vendor[e]["self_reference"][OWN_PAGE]["share"] + vendor[e]["self_reference"][OWN_LISTICLE]["share"]
        for e in names
    }
    top10_share = {
        e: sum(d["count"] for d in vendor[e]["top_domains"]) / vendor[e]["count"] for e in names
    }
    leaders = {
        e: {"domain": vendor[e]["top_domains"][0]["domain"], "count": vendor[e]["top_domains"][0]["count"],
            "self_reference_count": vendor[e]["top_domains"][0]["self_reference_count"]}
        for e in names
    }
    own_leader = max(names, key=own_share.get)
    own_rest = {e: v for e, v in own_share.items() if e != own_leader}
    google = [e for e in names if e in GOOGLE_ENGINES]
    others = [e for e in names if e not in GOOGLE_ENGINES]
    other_classes = [c for c in sc.DISPLAY_CLASSES if c != sc.VENDOR]
    derived: dict[str, Any] = {
        "counts": {"engines": len(names), "classes": len(sc.DISPLAY_CLASSES),
                   "top_domains": max(len(vendor[e]["top_domains"]) for e in names)},
        "metrics": {
            "vendor_share": extreme({e: vendor[e]["share"] for e in names}),
            "video_social_share": extreme({e: mix[e][sc.VIDEO_SOCIAL]["share"] for e in names}),
            "forums_share": extreme({e: mix[e][sc.FORUMS]["share"] for e in names}),
            "leftover_other_share": extreme({e: leftover[e]["other_share"] for e in names}),
            "top10_vendor_domain_share": extreme(top10_share),
            "own_page_plus_listicle_share": extreme(own_share),
        },
        "own_page_plus_listicle_share": own_share,
        "own_page_plus_listicle_rest_max": {"engine": max(own_rest, key=own_rest.get), "value": max(own_rest.values())},
        "own_page_plus_listicle_leader_engine": own_leader,
        "top10_vendor_domain_share": top10_share,
        "distinct_vendor_domains": {e: vendor[e]["distinct_domains"] for e in names},
        "leader_domain": leaders,
        "flags": {
            "vendor_largest_class_in_every_engine": all(
                mix[e][sc.VENDOR]["count"] > max(mix[e][c]["count"] for c in other_classes) for e in names
            ),
            "video_social_and_forums_only_in_google_engines": bool(google) and bool(others) and all(
                mix[e][sc.VIDEO_SOCIAL]["count"] > 0 and mix[e][sc.FORUMS]["count"] > 0 for e in google
            ) and all(
                mix[e][sc.VIDEO_SOCIAL]["count"] == 0 and mix[e][sc.FORUMS]["count"] == 0 for e in others
            ),
            "forums_below_video_social_in_google_engines": all(
                mix[e][sc.FORUMS]["share"] < mix[e][sc.VIDEO_SOCIAL]["share"] for e in google
            ),
            "own_share_leader_is_openai_chatgpt": own_leader == "openai_chatgpt",
            "own_share_leader_ahead_of_every_other_engine": own_share[own_leader] > max(own_rest.values()),
            # Prose sweep (2026-10-12): "most of their vendor citations go to a different vendor" in the other engines.
            "own_share_below_half_outside_openai_chatgpt": all(own_share[e] < 0.5 for e in names if e != "openai_chatgpt"),
            "leader_domain_of_own_share_leader_mostly_self_reference": (
                2 * leaders[own_leader]["self_reference_count"] > leaders[own_leader]["count"]
            ),
            "top10_domains_share_below_half_of_vendor_citations_in_google_engines": all(
                top10_share[e] < 0.5 for e in google
            ),
        },
    }
    if study is not None:
        derived["study"] = dict(study)
    return derived


def build_business_figures(
    inputs: FrozenInputs, domain_map: dict[str, str] | None = None, study: dict[str, int] | None = None
) -> dict[str, Any]:
    domain_map = sc.load_domain_map() if domain_map is None else domain_map
    rows, _report = build_citation_panel(inputs)
    mentioned = mentioned_own_domains_by_run(inputs)

    rows_by_engine: dict[str, list[CitationRow]] = defaultdict(list)
    for r in rows:
        rows_by_engine[r.engine].append(r)

    engine_blocks = {engine: _engine_block(engine, rows_by_engine[engine], domain_map, mentioned)
                     for engine in sorted(rows_by_engine)}
    leftover = {
        engine: {k: v for k, v in rep.items() if k != "top_unmapped_domains"}
        for engine, rep in sc.leftover_report(rows, domain_map).items()
    }
    return {
        "figure_id": "S1_business",
        "status": "exploratory_derived",
        "note": NOTE,
        "class_order": sc.DISPLAY_CLASSES,
        "self_reference_parts": list(SELF_REFERENCE_PARTS),
        "engines": engine_blocks,
        "leftover_other": leftover,
        "derived": derive(engine_blocks, leftover, study),
    }
