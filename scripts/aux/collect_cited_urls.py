#!/usr/bin/env python3
"""Task 5 -- cited URL resolution snapshot.

Enumerate every unique URL in any run's source list (~10,300) and resolve it
now: HEAD then GET following redirects, recording the final status, full
redirect chain, final registrable domain, content type and page title, and
storing the page body (capped, default 200 KB) for anything that resolves
200 + text/HTML.

Status is bucketed but never collapsed (dead / http_404 / http_403 / timeout
/ paywalled stay distinct). No domain-type classification. Feeds E1/E3/E4.

Idempotent: URLs already in data/02_aux/cited_urls/resolution.jsonl are skipped
unless --force. Use --limit / --domain for staged runs. Concurrency is
per-host-throttled so a big worker pool stays polite.

Usage:
    uv run python -m scripts.aux.collect_cited_urls --dry-run
    uv run python -m scripts.aux.collect_cited_urls --confirm --limit 50
    uv run python -m scripts.aux.collect_cited_urls --confirm
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

from analysis.aux.cited_urls import DEFAULT_RUNS_DIR, enumerate_cited_urls, resolve
from analysis.aux.http import PoliteSession
from analysis.aux.record import wrap
from analysis.aux.storage import AuxStore, DEFAULT_AUX_ROOT
from scripts.aux._run import rel, show_estimate_and_gate

REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECTOR = "collect_cited_urls"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    p.add_argument("--aux-root", type=Path, default=DEFAULT_AUX_ROOT)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--domain", action="append", default=None, help="only URLs whose host contains this string")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--host-interval", type=float, default=1.0)
    p.add_argument("--max-body-kb", type=int, default=200)
    p.add_argument("--force", action="store_true")
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    store = AuxStore(args.aux_root)
    index_path = store.path("cited_urls", "resolution.jsonl")

    urls = enumerate_cited_urls(args.runs_dir)
    if args.domain:
        needles = [d.lower() for d in args.domain]
        urls = [u for u in urls if any(n in (urlsplit(u["url"]).hostname or "").lower() for n in needles)]

    done = store.done_keys(index_path, lambda r: r.get("url"))
    todo = urls if args.force else [u for u in urls if u["url"] not in done]
    if args.limit:
        todo = todo[: args.limit]

    proceed = show_estimate_and_gate(
        "Task 5 · Cited URL resolution",
        [
            f"unique cited URLs total: {len(urls)}",
            f"already resolved (skipped): {len(urls) - len([u for u in urls if u['url'] not in done])}",
            f"URLs to resolve now: {len(todo)}",
            f"requests: ~{len(todo) * 2} (HEAD + GET)",
            f"workers: {args.workers}, min host interval: {args.host_interval}s",
            f"body cap: {args.max_body_kb} KB per 200/text-html page",
            "API spend: $0 (direct HTTP). Expect 1-4h for the full set.",
            f"output: {rel(index_path)} + cited_urls/bodies/<sha>.html",
        ],
        confirmed=args.confirm,
        dry_run=args.dry_run,
    )
    if not proceed:
        return

    session = PoliteSession(
        min_host_interval_seconds=args.host_interval,
        max_body_bytes=args.max_body_kb * 1024,
        capture_tls=False,  # Task 5 needs status/redirects/domain/title, not certs (that is Task 4)
    )
    context_by_url = {u["url"]: u for u in todo}
    buckets: dict[str, int] = {}
    done_n = 0

    def _work(url: str):
        fields, body = resolve(session, url)
        return url, fields, body

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_work, u["url"]) for u in todo]
            for fut in as_completed(futures):
                url, fields, body = fut.result()
                if body is not None:
                    store.write_bytes(fields["body_blob"], body)
                ctx = context_by_url.get(url, {})
                fields["cited_context"] = {
                    "first_engine": ctx.get("first_engine"),
                    "first_run_id": ctx.get("first_run_id"),
                    "first_day_index": ctx.get("first_day_index"),
                }
                store.append_record(index_path, wrap(COLLECTOR, fields, {"body_blob": fields.get("body_blob")}))
                buckets[fields["status_bucket"]] = buckets.get(fields["status_bucket"], 0) + 1
                done_n += 1
                if done_n % 100 == 0 or done_n == len(todo):
                    print(f"  [{done_n}/{len(todo)}] buckets={buckets}", flush=True)
    finally:
        session.close()

    print(f"[cited_urls] done: {done_n} resolved. buckets: {buckets}")
    print(f"[cited_urls] index: {rel(index_path)}")


if __name__ == "__main__":
    main()
