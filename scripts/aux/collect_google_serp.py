#!/usr/bin/env python3
"""Task 3 -- Google organic top 10, one SERP per (intent x paraphrase).

The window instrument only ever reached Google AI Mode (no organic SERP was
captured -- ranking_snapshot is empty). This fills that gap post-window: 35
SerpApi `google` searches, one per prompt string in config/prompts/prompts.csv, each
pinned to no_cache=true / location="United States" / gl=us / hl=en /
device=desktop, storing the entire raw JSON response.

Does NOT compute any per-company binary -- that is Phase A
(analysis/coding/b3b_serp_derivation.py).

Idempotent: a prompt whose raw JSON already exists is skipped unless --force.

Usage:
    uv run python -m scripts.aux.collect_google_serp --dry-run
    uv run python -m scripts.aux.collect_google_serp --confirm
"""
from __future__ import annotations

import argparse
from pathlib import Path

from analysis.aux.record import wrap
from analysis.aux.serp import APPROX_USD_PER_SEARCH, run_search
from analysis.aux.storage import AuxStore, DEFAULT_AUX_ROOT
from collector.config import ConfigError, SerpApiConfig
from collector.runner.load_prompts import load_prompts_csv
from scripts.aux._run import rel, Progress, show_estimate_and_gate

from collector import paths
DEFAULT_PROMPTS = paths.PROMPTS_CSV
COLLECTOR = "collect_google_serp"


def _raw_path(intent_id: int, paraphrase_id: int) -> str:
    return f"google_serp/i{intent_id}-p{paraphrase_id}.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    p.add_argument("--aux-root", type=Path, default=DEFAULT_AUX_ROOT)
    p.add_argument("--limit", type=int, default=None, help="only the first N prompts (staged runs)")
    p.add_argument("--force", action="store_true", help="re-fetch prompts already stored")
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    store = AuxStore(args.aux_root)
    index_path = store.path("google_serp", "index.jsonl")

    prompts = load_prompts_csv(args.prompts)
    if args.limit:
        prompts = prompts[: args.limit]

    todo = [
        pr for pr in prompts
        if args.force or not store.path(_raw_path(pr.intent_id, pr.paraphrase_id)).exists()
    ]
    skipped = len(prompts) - len(todo)

    est_low = len(todo) * 0.010
    est_high = len(todo) * APPROX_USD_PER_SEARCH
    proceed = show_estimate_and_gate(
        "Task 3 · Google organic SERP",
        [
            f"prompts in file: {len(prompts)}",
            f"already stored (skipped): {skipped}",
            f"SerpApi `google` calls to make: {len(todo)}",
            f"estimated spend: ${est_low:.2f}–${est_high:.2f} (~$0.010–0.015/search)",
            f"params: engine=google, location='United States', gl=us, hl=en, device=desktop, no_cache=true, num=10",
            f"output: {rel(index_path.parent)}/i<intent>-p<para>.json",
        ],
        confirmed=args.confirm,
        dry_run=args.dry_run,
    )
    if not proceed:
        return

    try:
        config = SerpApiConfig.from_env()
    except ConfigError as exc:
        raise SystemExit(str(exc))

    progress = Progress(len(todo), every=5, label="SERPs fetched")
    ok = failed = 0
    for pr in todo:
        result = run_search(pr.prompt_text, config=config)
        fields = {
            "intent_id": pr.intent_id,
            "paraphrase_id": pr.paraphrase_id,
            "branded": pr.branded,
            "query": pr.prompt_text,
            "ok": result.ok,
            "error": result.error,
            "elapsed_ms": result.elapsed_ms,
            "request_payload": result.request_payload,
            "raw_path": _raw_path(pr.intent_id, pr.paraphrase_id),
        }
        if result.raw is not None:
            store.write_json(_raw_path(pr.intent_id, pr.paraphrase_id), result.raw)
        record = wrap(COLLECTOR, fields, result.raw)
        # keep the big envelope only in the .json file; the index row points to it
        record["raw"] = {"stored_at": _raw_path(pr.intent_id, pr.paraphrase_id)} if result.raw is not None else None
        store.append_record(index_path, record)
        if result.ok:
            ok += 1
        else:
            failed += 1
            print(f"  ! i{pr.intent_id}-p{pr.paraphrase_id}: {result.error}")
        progress.tick(f"i{pr.intent_id}-p{pr.paraphrase_id}")

    print(f"[serp] done: {ok} ok, {failed} failed, {skipped} skipped. index: {rel(index_path)}")


if __name__ == "__main__":
    main()
