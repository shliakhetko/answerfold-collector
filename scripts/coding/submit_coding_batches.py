#!/usr/bin/env python3
"""Detached coding pipeline, step 1 of 2: submit A1's primary-extractor
Batch API jobs and exit. The machine can be shut down afterwards -- the
vendor keeps the jobs (and their results) for weeks.

Then, later (minutes to a day), run scripts/coding/collect_coding_batches.py
against the same --session-path: it polls the jobs, pulls the results into
the LLM cache, submits + collects the arbiter batch, and runs the rest of
the pipeline (A3 -> A4-prime -> A4 -> A5 -> B). collect can itself be
re-run any number of times and picks up where it left off.

Usage:
    uv run python -m scripts.coding.submit_coding_batches \\
        --runs-dir data/01_raw/runs --db-path data/03_coded/coding.sqlite3 \\
        --unbranded-intents 1,2,3,4,5,6 \\
        --inclusion-frame-path config/inclusion_frame.yaml
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from analysis.aggregate.load_runs import load_jsonl_dir
from analysis.coding.a1_answer_coding import submit_primary_batches
from analysis.coding.batch_session import BatchSession, default_session_path
from analysis.coding.cache import LLMCache
from analysis.coding.cited_url_resolution import DEFAULT_RESOLUTION_PATH as DEFAULT_CITED_URL_RESOLUTION_PATH
from analysis.coding.db import init_coding_schema
from scripts.coding.run_coding_pipeline import (
    ALT_DOMAINS_DEFAULT_SEED_PATH,
    DEFAULT_CACHE_DIR,
    DEFAULT_INCLUSION_FRAME_PATH,
    DEFAULT_SEED_LIST_PATH,
    STOP_AFTER_CHOICES,
)
from collector.runner.execute import get_instrument_commit
from collector.storage import api
from collector.storage.db import get_connection, init_db


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", type=Path, required=True)
    p.add_argument("--db-path", type=Path, required=True)
    p.add_argument("--unbranded-intents", type=str, required=True, help="comma-separated intent ids")
    p.add_argument("--inclusion-frame-path", type=Path, default=DEFAULT_INCLUSION_FRAME_PATH)
    p.add_argument("--seed-list-path", type=Path, default=DEFAULT_SEED_LIST_PATH)
    p.add_argument("--alt-domains-seed-path", type=Path, default=ALT_DOMAINS_DEFAULT_SEED_PATH)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--cited-url-resolution-path", type=Path, default=DEFAULT_CITED_URL_RESOLUTION_PATH)
    p.add_argument("--no-url-resolution", action="store_true")
    p.add_argument("--skip-entity-verification", action="store_true")
    p.add_argument("--skip-eligibility-floor", action="store_true",
                   help="carried to collect: emit cell_rates for every in-frame entity, no >=5-mention gate")
    p.add_argument("--stop-after", choices=STOP_AFTER_CHOICES, default=None,
                   help="carried to collect: stop the pipeline after this stage. 'a1' = mentions_raw "
                        "(no spend past the A1 arbiter); 'a3' = + entity resolution; 'a3b' = + entity "
                        "verification (Sonnet 'real company + homepage URL' pass -- last stage not needing "
                        "config/inclusion_frame.yaml); 'a4'/'a5' = + frame classification / URL classification. "
                        "Omit to run through B.")
    p.add_argument(
        "--limit", type=int, default=None,
        help="submit only the first N runs -- for a cheap end-to-end test before the full run.",
    )
    p.add_argument("--session-path", type=Path, default=None, help="default: <db-path>.batch-session.json")
    p.add_argument("--report-path", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    session_path = args.session_path or default_session_path(args.db_path)
    now = datetime.now(timezone.utc).isoformat()

    init_db(args.db_path)
    init_coding_schema(args.db_path)
    load_jsonl_dir(args.runs_dir, db_path=args.db_path, instrument_commit=get_instrument_commit())

    conn = get_connection(args.db_path)
    try:
        run_rows = api.list_runs(conn=conn)
    finally:
        conn.close()
    if args.limit is not None:
        run_rows = run_rows[: args.limit]

    cache = LLMCache(cache_dir=args.cache_dir / "a1_extractors")
    batch_ids = submit_primary_batches(run_rows, cache=cache)

    session = BatchSession(
        runs_dir=str(args.runs_dir),
        db_path=str(args.db_path),
        cache_dir=str(args.cache_dir),
        unbranded_intents=[int(x) for x in args.unbranded_intents.split(",") if x.strip()],
        inclusion_frame_path=str(args.inclusion_frame_path),
        seed_list_path=str(args.seed_list_path),
        alt_domains_seed_path=str(args.alt_domains_seed_path),
        cited_url_resolution_path=None if args.no_url_resolution else str(args.cited_url_resolution_path),
        skip_entity_verification=args.skip_entity_verification,
        report_path=str(args.report_path) if args.report_path else None,
        limit=args.limit,
        skip_eligibility_floor=args.skip_eligibility_floor,
        stop_after=args.stop_after,
        primary_batch_ids=batch_ids,
        created_at=now,
    )
    if batch_ids:
        session.advance("submitted", f"primary batches: {batch_ids}", now)
    else:
        session.advance("primaries_done", "every primary result already cached; nothing to submit", now)
    session.save(session_path)

    print(f"[submit] {len(run_rows)} runs" + (f" (--limit {args.limit})" if args.limit else ""))
    if batch_ids:
        for name, batch_id in batch_ids.items():
            print(f"[submit]   {name}: {batch_id}")
    else:
        print("[submit] nothing to submit -- primary extraction cache is already complete")
    print(f"[submit] session: {session_path}")
    print("[submit] you can shut the machine down now. next: uv run python -m scripts.coding.collect_coding_batches "
          f"--session-path {session_path}")


if __name__ == "__main__":
    main()
