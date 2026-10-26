#!/usr/bin/env python3
"""Task 9 -- wrap.

  - reports/aux_collection_summary.md: what was collected, per-task counts,
    failures, and what is still missing.
  - Extend data/01_raw/MANIFEST.json with a `data/02_aux/` section (index files
    hashed individually; body-blob directories rolled up by count + bytes).
  - A "not collected -- proposed" list (listed only, never collected here).

Read-only except for the two output files.

Usage:
    uv run python -m scripts.aux.aux_summary
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analysis.aux.storage import AuxStore, DEFAULT_AUX_ROOT, iter_jsonl
from scripts.aux._run import rel

from collector import paths
DEFAULT_REPORT = paths.AUX_COLLECTION_SUMMARY_MD
MANIFEST_PATH = paths.RAW_MANIFEST

PROPOSED_NOT_COLLECTED = [
    "Wayback Machine / CDX archival snapshot for each cited URL that resolved dead or paywalled (Task 5), to recover content lost since the window.",
    "DNS + WHOIS (registrar, creation date, nameservers) for every roster domain -- cheap signal on vendor maturity, decays slowly but not captured.",
    "security.txt / .well-known/ enumeration alongside the robots/llms fetch (Task 4).",
    "Per-engine fan-out sub-queries pulled from the stored response envelopes (Gemini exposes web_search_queries; OpenAI/Anthropic expose issued queries in output/content) -- derivable now, not yet extracted.",
    "HTTP response timing / TTFB per cited URL, for a freshness/perf signal.",
    "A second directory pass via an authenticated or headless path for the sites that returned 403 in Task 7 (explicitly out of scope: HARD RULE 7, no bypassing).",
]


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _index_stats(path: Path) -> dict[str, Any]:
    rows = list(iter_jsonl(path)) if path.exists() else []
    return {"path": rel(path), "exists": path.exists(), "rows": len(rows), "rows_data": rows}


def summarize(store: AuxStore) -> dict[str, Any]:
    root = store.root
    out: dict[str, Any] = {"aux_root": str(root), "collectors": {}}

    # roster
    roster_csv = store.path("roster_provisional.csv")
    roster_rows = 0
    roster_manual = 0
    if roster_csv.exists():
        import csv

        with roster_csv.open(encoding="utf-8") as f:
            for r in csv.DictReader(line for line in f if not line.startswith("#")):
                roster_rows += 1
                if (r.get("needs_manual") or "").lower() in {"1", "true", "yes"}:
                    roster_manual += 1
    out["roster"] = {"rows": roster_rows, "needs_manual": roster_manual, "exists": roster_csv.exists()}

    # google serp
    serp = _index_stats(store.path("google_serp", "index.jsonl"))
    serp_ok = sum(1 for r in serp["rows_data"] if r.get("ok"))
    out["collectors"]["google_serp"] = {
        "rows": serp["rows"], "ok": serp_ok, "failed": serp["rows"] - serp_ok,
        "errors": [f"i{r['intent_id']}-p{r['paraphrase_id']}: {r.get('error')}" for r in serp["rows_data"] if not r.get("ok")],
    }

    # robots
    robots = _index_stats(store.path("robots", "index.jsonl"))
    out["collectors"]["robots"] = {"domains": robots["rows"]}

    # cited urls
    cited = _index_stats(store.path("cited_urls", "resolution.jsonl"))
    buckets = Counter(r.get("status_bucket") for r in cited["rows_data"])
    bodies = store.path("cited_urls", "bodies")
    body_files = list(bodies.glob("*.html")) if bodies.exists() else []
    out["collectors"]["cited_urls"] = {
        "resolved": cited["rows"],
        "status_buckets": dict(buckets),
        "bodies_stored": len(body_files),
        "bodies_bytes": sum(p.stat().st_size for p in body_files),
    }

    # site inventory
    inv = _index_stats(store.path("site_inventory", "index.jsonl"))
    with_sitemap = sum(1 for r in inv["rows_data"] if r.get("sitemap_exists"))
    out["collectors"]["site_inventory"] = {
        "domains": inv["rows"],
        "with_sitemap": with_sitemap,
        "crawl_fallback": inv["rows"] - with_sitemap,
        "total_urls": sum(r.get("url_count", 0) for r in inv["rows_data"]),
    }

    # directories
    dirs = _index_stats(store.path("directories", "index.jsonl"))
    outc = Counter(r.get("outcome") for r in dirs["rows_data"])
    per_dir = Counter(r.get("directory") for r in dirs["rows_data"])
    out["collectors"]["directories"] = {"checks": dirs["rows"], "outcomes": dict(outc), "by_directory": dict(per_dir)}

    return out


def build_manifest_aux(store: AuxStore) -> dict[str, Any]:
    root = store.root
    index_files: list[dict[str, Any]] = []
    for pattern in ("*.jsonl", "*/*.jsonl", "*/*.json", "*.csv", "*.json"):
        for p in sorted(root.glob(pattern)):
            if not p.is_file() or "bodies" in p.parts:
                continue
            n = sum(1 for _ in p.read_text(encoding="utf-8").splitlines() if _.strip()) if p.suffix == ".jsonl" else None
            index_files.append(
                {
                    "path": rel(p),
                    "sha256": _sha256_file(p),
                    "bytes": p.stat().st_size,
                    "lines": n,
                }
            )
    # de-dup (globs overlap)
    seen = set()
    deduped = []
    for e in index_files:
        if e["path"] in seen:
            continue
        seen.add(e["path"])
        deduped.append(e)

    blob_dirs: list[dict[str, Any]] = []
    for d in sorted(p for p in root.rglob("*") if p.is_dir() and any(c.is_file() for c in p.iterdir())):
        files = [c for c in d.rglob("*") if c.is_file()]
        if not files:
            continue
        blob_dirs.append(
            {
                "dir": rel(d),
                "file_count": len(files),
                "bytes": sum(c.stat().st_size for c in files),
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "index_files": deduped,
        "blob_dirs": blob_dirs,
    }


def render_md(summary: dict[str, Any]) -> str:
    c = summary["collectors"]
    L: list[str] = []
    w = L.append
    w("# Auxiliary collection summary")
    w("")
    w(f"_Generated {datetime.now(timezone.utc).isoformat()} · post-window batch · roster source: data/03_coded/coding_a1.sqlite3 `final_entity_list` (the frozen 415)_")
    w("")
    w("## What was collected")
    w("")
    r = summary["roster"]
    w(
        f"- **Provisional roster**: {r['rows']} rows "
        f"(415 in-frame companies + {r['needs_manual']} unresolved candidate; "
        f"{c['robots']['domains']} unique registrable domains after collapse)."
    )
    s = c["google_serp"]
    w(f"- **Task 3 · Google SERP**: {s['rows']} prompts fetched, {s['ok']} ok, {s['failed']} failed.")
    for e in s["errors"]:
        w(f"  - failed: {e}")
    w(f"- **Task 4 · Crawler eligibility**: {c['robots']['domains']} domains snapshotted.")
    cu = c["cited_urls"]
    w(f"- **Task 5 · Cited URL resolution**: {cu['resolved']} URLs resolved; {cu['bodies_stored']} bodies stored ({cu['bodies_bytes'] / 1e6:.1f} MB).")
    w(f"  - status buckets: {cu['status_buckets']}")
    inv = c["site_inventory"]
    w(f"- **Task 6 · Site inventory**: {inv['domains']} domains ({inv['with_sitemap']} with a sitemap, {inv['crawl_fallback']} via crawl fallback); {inv['total_urls']} URLs recorded.")
    d = c["directories"]
    w(f"- **Task 7 · Directory presence**: {d['checks']} (entity × directory) checks; outcomes {d['outcomes']}.")
    w("")
    w("## Failures")
    w("")
    w(f"- SERP failures: {c['google_serp']['failed']}")
    non_ok = ", ".join(f"{k}={v}" for k, v in cu["status_buckets"].items() if k != "ok")
    w(f"- Dead / blocked / other cited URLs: {non_ok or 'none'}")
    w(f"- Blocked directories: {c['directories']['outcomes'].get('blocked', 0)}; errors: {c['directories']['outcomes'].get('error', 0)}")
    w(f"- Roster rows still needing a manual domain: {r['needs_manual']}")
    w("")
    w("## Still missing")
    w("")
    w("- In-window auxiliary state (`domain_redirect_snapshot`, `ranking_snapshot`, `robots_ai_crawler_block`): unrecoverable; this batch holds only the post-window snapshot (window close +11-12 days).")
    w("- `instrument_commit` / `latency_ms` for the window: unrecoverable.")
    w(f"- Homepages for {r['needs_manual']} unresolved candidate row(s) in the roster (no `homepage_url` and no A4-prime verified URL).")
    w("")
    w("## Proposed but NOT collected (needs your go-ahead)")
    w("")
    for item in PROPOSED_NOT_COLLECTED:
        w(f"- {item}")
    w("")
    return "\n".join(L)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aux-root", type=Path, default=DEFAULT_AUX_ROOT)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    store = AuxStore(args.aux_root)
    summary = summarize(store)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_md(summary), encoding="utf-8")
    print(f"[summary] wrote {rel(args.report)}")

    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text())
    else:
        manifest = {"note": "raw window manifest not yet generated; run scripts.aux.freeze_window"}
    manifest["aux"] = build_manifest_aux(store)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[summary] updated {rel(args.manifest)} with the data/02_aux section")


if __name__ == "__main__":
    main()
