#!/usr/bin/env python3
"""Task 7 -- directory presence snapshot.

For every roster company, check G2, Capterra, TrustRadius, Trustpilot,
Product Hunt and Crunchbase. Profile URLs are constructed from a slug of the
company name (and the domain, for Trustpilot) and HTTP-verified -- no
SERP-assisted discovery.

Outcomes recorded distinctly: found / absent / blocked / error. `absent`
carries the exact URL templates tried (`search_method`) so it is
reproducible. Raw HTML stored for found + blocked pages. No review-count or
rating extraction.

Writes data/02_aux/directories/index.jsonl + data/02_aux/directories/bodies/.
Idempotent per (entity, directory). No API spend.

Usage:
    uv run python -m scripts.aux.collect_directories --dry-run
    uv run python -m scripts.aux.collect_directories --confirm
"""
from __future__ import annotations

import argparse
from pathlib import Path

from analysis.aux.directories import DIRECTORIES, check_directory
from analysis.aux.http import PoliteSession
from analysis.aux.record import wrap
from analysis.aux.roster import DEFAULT_ROSTER_CSV, load_from_csv
from analysis.aux.storage import AuxStore, DEFAULT_AUX_ROOT
from scripts.aux._run import rel, Progress, show_estimate_and_gate

REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECTOR = "collect_directories"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roster", type=Path, default=DEFAULT_ROSTER_CSV)
    p.add_argument("--aux-root", type=Path, default=DEFAULT_AUX_ROOT)
    p.add_argument("--limit", type=int, default=None, help="only the first N roster companies")
    p.add_argument("--directory", action="append", default=None, choices=DIRECTORIES)
    p.add_argument("--host-interval", type=float, default=2.0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    store = AuxStore(args.aux_root)
    index_path = store.path("directories", "index.jsonl")
    directories = args.directory or list(DIRECTORIES)

    try:
        roster = [e for e in load_from_csv(args.roster) if not e.entity_id.startswith("cand-")]
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    if args.limit:
        roster = roster[: args.limit]

    done = store.done_keys(index_path, lambda r: (r.get("entity_id"), r.get("directory")))
    units = [
        (e, d) for e in roster for d in directories
        if args.force or (e.entity_id, d) not in done
    ]

    proceed = show_estimate_and_gate(
        "Task 7 · Directory presence snapshot",
        [
            f"roster companies: {len(roster)}",
            f"directories: {directories}",
            f"(entity, directory) checks now: {len(units)}",
            f"requests: ~{len(units)}-{len(units) * 2} (1-2 templates each)",
            "API spend: $0 (direct HTTP). Expect many 403/blocked from Cloudflare-fronted sites.",
        ],
        confirmed=args.confirm,
        dry_run=args.dry_run,
    )
    if not proceed:
        return

    session = PoliteSession(min_host_interval_seconds=args.host_interval)
    progress = Progress(len(units), every=10, label="checks")
    outcomes: dict[str, int] = {}
    try:
        for entity, directory in units:
            fields, body = check_directory(
                session, directory, canonical_name=entity.canonical_name, primary_domain=entity.primary_domain
            )
            fields["entity_id"] = entity.entity_id
            fields["canonical_name"] = entity.canonical_name
            if body is not None:
                blob = f"directories/bodies/{entity.entity_id}-{directory}.html"
                store.write_bytes(blob, body)
                fields["body_blob"] = blob
            store.append_record(index_path, wrap(COLLECTOR, fields, {"body_blob": fields.get("body_blob")}))
            outcomes[fields["outcome"]] = outcomes.get(fields["outcome"], 0) + 1
            progress.tick(f"{entity.entity_id}/{directory}={fields['outcome']}")
    finally:
        session.close()

    print(f"[directories] done: {len(units)} checks. outcomes: {outcomes}")
    print(f"[directories] index: {rel(index_path)}")


if __name__ == "__main__":
    main()
