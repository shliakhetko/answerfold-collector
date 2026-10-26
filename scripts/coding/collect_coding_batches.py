#!/usr/bin/env python3
"""Detached coding pipeline, step 2 of 2: finish a run started with
scripts/coding/submit_coding_batches.py.

Reads the --session-path written by submit, then advances as far as it can:

  submitted        -> poll the primary batches; when all are done, pull
                      their results into the LLM cache
  primaries_done   -> compute the disputed set, run the synchronous arbiter
                      (ARBITER_EXTRACTOR) over just those runs, into cache
  arbiter_done     -> run the local consensus math + A3 -> A4-prime -> A4
                      -> A5 -> B, write the pipeline report
  complete         -> nothing to do

Safe to re-run any number of times. Exits 0 when the pipeline is complete,
2 when it is still waiting on a vendor batch (re-run later), 1 on error.
The machine may be off between runs.

Note: the primaries (Claude + Gemini) run as one vendor Batch job each; the
arbiter runs synchronously here (OpenAI's Batch API is unusable on this
account, and it only touches the disputed runs). The final step
(arbiter_done -> complete) runs A3 synchronously (one call), then batches
A4-prime, A4 and A5 through the Anthropic Message Batches API -- each waits
inline up to --verification-timeout and, on timeout, exits 2 and resumes
that batch on the next run (no re-charge). B1-B3 are local math.

Usage:
    uv run python -m scripts.coding.collect_coding_batches --session-path data/03_coded/coding.sqlite3.batch-session.json
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from analysis.coding.a1_answer_coding import (
    ARBITER_EXTRACTOR,
    disputed_by_run_from_cache,
    fetch_primary_batches,
    primary_batch_statuses,
    run_arbiter_sync_into_cache,
)
from analysis.coding.batch_session import BatchSession
from analysis.coding.cache import LLMCache
from collector.storage import api
from collector.storage.db import get_connection
from scripts.coding.run_coding_pipeline import _print_summary, run_pipeline

STILL_WAITING = 2

# Terminal, successful statuses a primary batch can report -- OpenAI
# "completed", Gemini "JOB_STATE_SUCCEEDED", Anthropic "ended". Anything
# else terminal (e.g. "failed", "expired", "cancelled") means the vendor
# did not run the batch -- collect must abort loudly, never treat it as
# "done".
PRIMARY_BATCH_OK_STATUSES = frozenset({"completed", "JOB_STATE_SUCCEEDED", "ended"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session-path", type=Path, required=True)
    p.add_argument(
        "--arbiter-workers", type=int, default=8,
        help="thread-pool size for the synchronous arbiter calls in the primaries_done step. Default 8.",
    )
    p.add_argument(
        "--verification-timeout", type=float, default=3600.0,
        help="seconds to wait for the A4-prime (entity verification) batch in the finalize step. "
             "On timeout the run exits 2 and the batch resumes on the next collect (no re-charge). "
             "Default 3600.",
    )
    return p.parse_args(argv)


def _run_rows(db_path: str, limit: int | None) -> list[dict]:
    conn = get_connection(db_path)
    try:
        rows = api.list_runs(conn=conn)
    finally:
        conn.close()
    return rows[:limit] if limit is not None else rows


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    session = BatchSession.load(args.session_path)
    now = datetime.now(timezone.utc).isoformat()
    print(f"[collect] phase: {session.phase}")

    run_rows = _run_rows(session.db_path, session.limit)
    ext_cache = LLMCache(cache_dir=Path(session.cache_dir) / "a1_extractors")
    arb_cache = LLMCache(cache_dir=Path(session.cache_dir) / "a1_arbiter")

    if session.phase == "submitted":
        statuses = primary_batch_statuses(session.primary_batch_ids)
        not_ready = sorted(n for n, s in statuses.items() if s is None)
        if not_ready:
            print(f"[collect] primary batches still processing: {', '.join(not_ready)}")
            print("[collect] re-run collect later.")
            sys.exit(STILL_WAITING)
        failed = sorted(
            f"{n}={s}" for n, s in statuses.items() if str(s) not in PRIMARY_BATCH_OK_STATUSES
        )
        if failed:
            print(f"[collect] primary batch(es) did NOT succeed: {', '.join(failed)}")
            print("[collect] the vendor did not run the batch. Not advancing. Clear that "
                  "extractor's cache misses and resubmit (scripts.coding.submit_coding_batches) "
                  "before re-running collect.")
            sys.exit(1)
        print(f"[collect] primary batches done: {statuses}")
        fetch_primary_batches(run_rows, session.primary_batch_ids, cache=ext_cache)
        session.advance("primaries_done", f"fetched {list(session.primary_batch_ids)}", now)
        session.save(args.session_path)

    if session.phase == "primaries_done":
        disputed = disputed_by_run_from_cache(run_rows, cache=ext_cache)
        print(f"[collect] {len(disputed)} disputed runs -> synchronous {ARBITER_EXTRACTOR} arbiter")
        made = run_arbiter_sync_into_cache(
            run_rows, disputed, arbiter_cache=arb_cache, max_workers=args.arbiter_workers
        )
        session.advance("arbiter_done", f"arbiter (sync) made {made} calls, {len(disputed)} disputed runs", now)
        session.save(args.session_path)

    if session.phase == "arbiter_done":
        print("[collect] A1 batches complete. Running consensus + A3 -> A4-prime (batched) -> A4 -> A5 -> B ...")
        try:
            report = run_pipeline(
                runs_dir=Path(session.runs_dir),
                db_path=Path(session.db_path),
                unbranded_intent_ids=set(session.unbranded_intents),
                inclusion_frame_path=Path(session.inclusion_frame_path),
                seed_list_path=Path(session.seed_list_path),
                alt_domains_seed_path=Path(session.alt_domains_seed_path),
                cache_dir=Path(session.cache_dir),
                cited_url_resolution_path=(
                    Path(session.cited_url_resolution_path) if session.cited_url_resolution_path else None
                ),
                skip_entity_verification=session.skip_entity_verification,
                report_path=Path(session.report_path) if session.report_path else None,
                a1_offline=True,
                run_limit=session.limit,
                eligibility_floor=not session.skip_eligibility_floor,
                verification_batch_max_wait_seconds=args.verification_timeout,
                stop_after=session.stop_after,
            )
        except TimeoutError as exc:
            print(f"[collect] {exc}")
            print("[collect] the verification batch is still processing -- re-run collect later "
                  "(it resumes that batch, no re-charge).")
            sys.exit(STILL_WAITING)
        session.advance("complete", "pipeline finished", now)
        session.save(args.session_path)
        _print_summary(report)
        print("[collect] complete.")
        return

    if session.phase == "complete":
        print("[collect] already complete -- nothing to do.")
        return


if __name__ == "__main__":
    main()
