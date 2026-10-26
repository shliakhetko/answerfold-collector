"""Display classes for the business view of S1's source mix.

Display-only relabel of the non-vendor-published citations in the citation
panel. Vendor-published stays exactly what S1 decided
(publisher_is_in_category True after the domain-ownership override); nothing
here can move that share.

A non-vendor citation gets its class from, in order:
1. the registrable domain in config/source_domain_classes.csv (so youtube.com
   is "Video and social" whatever A5 labelled the page);
2. otherwise A5's raw url_class: review site, forum and news keep their
   class; a vendor page or vendor listicle that is not vendor-published is a
   company site outside the category ("Other company sites");
3. otherwise "Other" (raw "other" on a domain the map does not know).

Citations with no url_registry match are not classified (S1 drops them from
the denominator too) and are left out here.

Report the leftover "Other" share and the top unmapped domains:
    uv run python -m analysis.phase_b.business.source_classes --data-root data
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

from analysis.coding.registrable_domain import registrable_domain
from analysis.phase_b.citation_panel import CitationRow
from collector import paths

VENDOR = "Vendor-published"
REVIEW = "Review sites"
FORUMS = "Forums"
NEWS = "News and press"
VIDEO_SOCIAL = "Video and social"
REGULATORS = "Regulators legal and standards"
BLOGS = "Blogs and publishing platforms"
COMPANY_SITES = "Other company sites"
OTHER = "Other"

# Fixed display order: vendor-published first, "Other" last.
DISPLAY_CLASSES = [VENDOR, REVIEW, FORUMS, NEWS, VIDEO_SOCIAL, REGULATORS, BLOGS, COMPANY_SITES, OTHER]
MAPPED_CLASSES = frozenset(DISPLAY_CLASSES) - {VENDOR, OTHER}

RAW_CLASS_FALLBACK = {
    "review site": REVIEW,
    "forum": FORUMS,
    "news": NEWS,
    "vendor page": COMPANY_SITES,
    "vendor listicle": COMPANY_SITES,
    "other": OTHER,
}

# The three engines of the main result; Claude is reported but sits outside it.
MAIN_ENGINES = ("openai_chatgpt", "google_ai_mode", "google_gemini")
OUTSIDE_ENGINES = ("anthropic_claude",)
LEFTOVER_TARGET = 0.03


def load_domain_map(path: Path | str = paths.SOURCE_DOMAIN_CLASSES_CSV) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            domain = row["domain"].strip().lower()
            cls = row["display_class"].strip()
            if cls not in MAPPED_CLASSES:
                raise ValueError(f"{path}: {domain!r} has unknown display class {cls!r}")
            if domain in mapping:
                raise ValueError(f"{path}: duplicate domain {domain!r}")
            mapping[domain] = cls
    return mapping


def row_domain(row: CitationRow) -> str:
    return registrable_domain(row.url_normalized) or row.url_normalized


def display_class(row: CitationRow, domain_map: dict[str, str]) -> str | None:
    """Display class of one citation; None when it has no url_registry match."""
    if row.url_class is None:
        return None
    if row.publisher_is_in_category is True:
        return VENDOR
    mapped = domain_map.get(row_domain(row))
    if mapped is not None:
        return mapped
    return RAW_CLASS_FALLBACK.get(row.url_class, OTHER)


def class_counts_by_engine(rows: list[CitationRow], domain_map: dict[str, str]) -> dict[str, dict[str, int]]:
    """engine -> display class -> citation count, classified citations only."""
    out: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        cls = display_class(row, domain_map)
        if cls is not None:
            out[row.engine][cls] += 1
    return {engine: dict(counts) for engine, counts in out.items()}


def leftover_report(
    rows: list[CitationRow], domain_map: dict[str, str], top_n: int = 15
) -> dict[str, dict]:
    """Per engine: classified citation count, leftover "Other" count and
    share, and the top unmapped domains inside that leftover."""
    classified: Counter = Counter()
    unmapped: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        cls = display_class(row, domain_map)
        if cls is None:
            continue
        classified[row.engine] += 1
        if cls == OTHER:
            unmapped[row.engine][row_domain(row)] += 1
    report = {}
    for engine in sorted(classified):
        other = sum(unmapped[engine].values())
        report[engine] = {
            "classified": classified[engine],
            "other": other,
            "other_share": other / classified[engine],
            "top_unmapped_domains": unmapped[engine].most_common(top_n),
        }
    return report


def format_report(report: dict[str, dict]) -> str:
    lines = []
    for engine, r in report.items():
        mark = "  [outside the main result]" if engine in OUTSIDE_ENGINES else ""
        lines.append(f"{engine}{mark}: Other {r['other']} of {r['classified']} = {r['other_share']:.2%}")
        for domain, n in r["top_unmapped_domains"]:
            lines.append(f"    {n:5d}  {domain}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from analysis.phase_b.citation_panel import build_citation_panel
    from analysis.phase_b.reading import PhaseBInputError, read_frozen_inputs, resolve_data_root

    parser = argparse.ArgumentParser(description="Leftover Other share per engine")
    parser.add_argument("--data-root", default=None, help="frozen data directory (else $GEO_DATA_ROOT)")
    args = parser.parse_args(argv)
    try:
        data_root = resolve_data_root(args.data_root)
        inputs = read_frozen_inputs(paths.REPO_ROOT, data_root)
    except PhaseBInputError as exc:
        print(f"stopped: {exc}", file=sys.stderr)
        return 1
    rows, _ = build_citation_panel(inputs)
    print(format_report(leftover_report(rows, load_domain_map())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
