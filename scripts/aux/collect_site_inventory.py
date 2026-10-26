#!/usr/bin/env python3
"""Task 6 -- company site inventory (one shallow pass per roster domain).

Fetch /sitemap.xml and /sitemap_index.xml, follow a sitemap index one level,
store the XML verbatim, and record the URL list only. When there is no
sitemap, fall back to a shallow same-site crawl capped at depth 2 and a page
limit -- again recording only the URL list. No page classification.

Writes data/02_aux/site_inventory/<domain>.jsonl (one record per domain) and
data/02_aux/site_inventory/raw_sitemaps/<domain>/*.xml.

Idempotent per domain. No API spend.

Usage:
    uv run python -m scripts.aux.collect_site_inventory --dry-run
    uv run python -m scripts.aux.collect_site_inventory --confirm
"""
from __future__ import annotations

import argparse
from pathlib import Path

from analysis.aux.http import PoliteSession
from analysis.aux.record import wrap
from analysis.aux.roster import DEFAULT_ROSTER_CSV, load_from_csv
from analysis.aux.sitemap import DEFAULT_CRAWL_PAGE_LIMIT, inventory_domain
from analysis.aux.storage import AuxStore, DEFAULT_AUX_ROOT, domain_key
from scripts.aux._run import rel, Progress, show_estimate_and_gate

REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECTOR = "collect_site_inventory"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roster", type=Path, default=DEFAULT_ROSTER_CSV)
    p.add_argument("--aux-root", type=Path, default=DEFAULT_AUX_ROOT)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--page-limit", type=int, default=DEFAULT_CRAWL_PAGE_LIMIT, help="crawl-fallback page cap")
    p.add_argument("--no-crawl-fallback", dest="crawl_fallback", action="store_false", default=True)
    p.add_argument("--host-interval", type=float, default=1.0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    store = AuxStore(args.aux_root)
    index_path = store.path("site_inventory", "index.jsonl")

    try:
        roster = load_from_csv(args.roster)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    domains: list[str] = []
    for e in roster:
        for d in e.domains:
            if d not in domains:
                domains.append(d)

    done = store.done_keys(index_path, lambda r: r.get("domain"))
    todo = domains if args.force else [d for d in domains if domain_key(d) not in done]
    if args.limit:
        todo = todo[: args.limit]

    proceed = show_estimate_and_gate(
        "Task 6 · Company site inventory",
        [
            f"roster domains: {len(domains)}",
            f"domains to inventory now: {len(todo)}",
            f"requests: ~2-4 per domain with a sitemap; up to {args.page_limit} for a crawl fallback",
            f"crawl fallback: {'on' if args.crawl_fallback else 'off'} (depth 2)",
            "API spend: $0 (direct HTTP)",
        ],
        confirmed=args.confirm,
        dry_run=args.dry_run,
    )
    if not proceed:
        return

    session = PoliteSession(min_host_interval_seconds=args.host_interval)
    progress = Progress(len(todo), every=5, label="domains")
    try:
        for domain in todo:
            fields, blobs = inventory_domain(
                session, domain, crawl_fallback=args.crawl_fallback, page_limit=args.page_limit
            )
            for blob_rel, data in blobs.items():
                store.write_bytes(blob_rel, data)
            # the URL list can be large -> its own file; the index row summarizes
            urls = fields.pop("urls")
            store.write_json(f"site_inventory/{domain_key(domain)}.urls.json", urls)
            fields["urls_file"] = f"site_inventory/{domain_key(domain)}.urls.json"
            store.append_record(index_path, wrap(COLLECTOR, fields, {"urls_file": fields["urls_file"]}))
            progress.tick(f"{domain} ({fields['url_count']} urls, {fields['source']})")
    finally:
        session.close()

    print(f"[site_inventory] done: {len(todo)} domains. index: {rel(index_path)}")


if __name__ == "__main__":
    main()
