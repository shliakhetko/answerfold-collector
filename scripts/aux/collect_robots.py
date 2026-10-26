#!/usr/bin/env python3
"""Task 4 -- crawler eligibility snapshot.

For every domain in the provisional roster: fetch /robots.txt, /llms.txt,
/llms-full.txt, /ai.txt and the homepage, storing every body verbatim plus
status, redirect chain, final URL, TLS issuer/expiry, response headers, the
meta-robots tag and the X-Robots-Tag header. robots.txt is NOT parsed here.

Writes data/02_aux/robots/index.jsonl + data/02_aux/robots/<domain>/... and records
the AI crawler agent list into data/02_aux/collector_config.json.

Idempotent per domain (skip domains already in the index unless --force).
No API spend -- direct HTTP only.

Usage:
    uv run python -m scripts.aux.collect_robots --dry-run
    uv run python -m scripts.aux.collect_robots --confirm
"""
from __future__ import annotations

import argparse
from pathlib import Path

from analysis.aux.crawler_agents import AI_CRAWLER_AGENTS
from analysis.aux.http import AUX_CONTACT_URL, USER_AGENT, PoliteSession
from analysis.aux.record import collector_version, wrap
from analysis.aux.robots_snapshot import WELL_KNOWN_FILES, snapshot_domain
from analysis.aux.roster import DEFAULT_ROSTER_CSV, load_from_csv
from analysis.aux.storage import AuxStore, DEFAULT_AUX_ROOT, domain_key
from scripts.aux._run import rel, Progress, show_estimate_and_gate

REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECTOR = "collect_robots"


def write_collector_config(store: AuxStore, extra: dict | None = None) -> None:
    cfg = {
        "collector_version": collector_version(),
        "user_agent": USER_AGENT,
        "contact_url": AUX_CONTACT_URL,
        "ai_crawler_agents": list(AI_CRAWLER_AGENTS),
        "well_known_files": list(WELL_KNOWN_FILES),
        "note": "Agent list is config for the Phase-A robots/llms parse; Task 4 stores raw, never parses.",
    }
    if extra:
        cfg.update(extra)
    store.write_json("collector_config.json", cfg)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roster", type=Path, default=DEFAULT_ROSTER_CSV)
    p.add_argument("--aux-root", type=Path, default=DEFAULT_AUX_ROOT)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--domain", action="append", default=None, help="restrict to these roster domains")
    p.add_argument("--host-interval", type=float, default=1.5, help="min seconds between requests to one host")
    p.add_argument("--force", action="store_true")
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    store = AuxStore(args.aux_root)
    index_path = store.path("robots", "index.jsonl")

    try:
        roster = load_from_csv(args.roster)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    domains: list[str] = []
    for entity in roster:
        for d in entity.domains:
            if args.domain and d not in {domain_key(x) for x in args.domain}:
                continue
            if d not in domains:
                domains.append(d)
    no_domain = [e.entity_id for e in roster if not e.domains]

    done = store.done_keys(index_path, lambda r: r.get("domain"))
    todo = domains if args.force else [d for d in domains if domain_key(d) not in done]
    if args.limit:
        todo = todo[: args.limit]

    reqs = len(todo) * (len(WELL_KNOWN_FILES) + 1)
    proceed = show_estimate_and_gate(
        "Task 4 · Crawler eligibility snapshot",
        [
            f"roster domains: {len(domains)} ({len(no_domain)} roster rows have no domain, skipped)",
            f"domains to fetch now: {len(todo)}",
            f"HTTP requests (~{len(WELL_KNOWN_FILES) + 1} per domain): ~{reqs}",
            "API spend: $0 (direct HTTP)",
            f"user-agent: {USER_AGENT}",
            f"min host interval: {args.host_interval}s",
        ],
        confirmed=args.confirm,
        dry_run=args.dry_run,
    )
    if not proceed:
        return

    write_collector_config(store, {"host_interval_seconds": args.host_interval})
    session = PoliteSession(min_host_interval_seconds=args.host_interval)
    progress = Progress(len(todo), every=10, label="domains")
    try:
        for domain in todo:
            fields, blobs = snapshot_domain(session, domain)
            for blob_rel, data in blobs.items():
                store.write_bytes(blob_rel, data)
            store.append_record(index_path, wrap(COLLECTOR, fields, {"blob_dir": fields["blob_dir"]}))
            progress.tick(domain)
    finally:
        session.close()

    print(f"[robots] done: {len(todo)} domains. index: {rel(index_path)}")
    if no_domain:
        print(f"[robots] roster rows with no domain (report in Task 9): {no_domain}")


if __name__ == "__main__":
    main()
