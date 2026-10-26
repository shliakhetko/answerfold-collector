#!/usr/bin/env python3
"""Task 1 -- collection integrity audit. READ ONLY.

Reads data/01_raw/runs/*.jsonl directly (not the DB) and writes reports/collection_audit.md.
Reports; does not fix anything.

  - Expected grid (engine x day x intent x paraphrase x rep) vs actual:
    every missing cell, every duplicate run_id, every errored / empty response.
  - Timestamp distribution per day per engine; every run outside the fixed
    fire hour (14:00 UTC / 17:00 Kyiv) with its offset.
  - model_returned vs model_requested per engine; any mid-window change.
  - search_fired null-rate / false-rate per engine per intent.
  - Runs with an empty source list; runs with response text below a floor.
  - Unique cited URL count and unique registrable domain count.
  - Errors / retries / rate-limit events logged in the JSONL.
  - The permanent gaps: instrument_commit and latency_ms never written; the
    once-per-window auxiliary tables never populated.

Usage:
    uv run python -m scripts.aux.audit_collection
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any

from analysis.aggregate.response_fields import extract_model_returned, extract_raw_response_text, extract_sources_rows
from analysis.aggregate.usage import extract_search_fired
from analysis.aux.record import KYIV_TZ
from collector.runner.load_prompts import load_prompts_csv
from collector.runner.experiment_config import load_experiment_config
from scripts.aux._run import rel

from collector import paths
DEFAULT_RUNS_DIR = paths.RUNS_DIR
DEFAULT_REPORT = paths.COLLECTION_AUDIT_MD
DEFAULT_PROMPTS = paths.PROMPTS_CSV
DEFAULT_EXPERIMENT = paths.EXPERIMENT_YAML

# Fixed fire hour for the study: 17:00 Europe/Kyiv == 14:00 UTC (EEST). A run
# block takes ~40 min, so anything starting past this counts as off-window.
TARGET_UTC_HOUR = 14
TARGET_WINDOW_MINUTES = 60
RESPONSE_TEXT_FLOOR_CHARS = 200


def load_runs(runs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(runs_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _offset_minutes(ts: str) -> float:
    dt = datetime.fromisoformat(ts).astimezone(timezone.utc)
    target = datetime.combine(dt.date(), time(TARGET_UTC_HOUR, 0), tzinfo=timezone.utc)
    return (dt - target).total_seconds() / 60.0


def build_report(runs_dir: Path, prompts_path: Path, experiment_path: Path) -> str:
    rows = load_runs(runs_dir)
    if not rows:
        return "# Collection integrity audit\n\nNo runs found.\n"

    try:
        prompts = load_prompts_csv(prompts_path)
        intents = sorted({p.intent_id for p in prompts})
        paraphrases_by_intent = defaultdict(set)
        for p in prompts:
            paraphrases_by_intent[p.intent_id].add(p.paraphrase_id)
        branded_intents = sorted({p.intent_id for p in prompts if p.branded})
    except FileNotFoundError:
        prompts = None
        intents = sorted({r["intent_id"] for r in rows})
        paraphrases_by_intent = defaultdict(set)
        for r in rows:
            paraphrases_by_intent[r["intent_id"]].add(r["paraphrase_id"])
        branded_intents = []

    try:
        exp = load_experiment_config(experiment_path)
        engines = [c.engine for c in exp.engines]
        model_requested_by_engine = {c.engine: c.model_requested for c in exp.engines}
        days = list(range(1, exp.days + 1))
        reps = list(range(1, exp.reps_per_day + 1))
    except FileNotFoundError:
        engines = sorted({r["engine"] for r in rows})
        model_requested_by_engine = {}
        days = sorted({r["day_index"] for r in rows})
        reps = sorted({r["rep_index"] for r in rows})

    # ---- expected vs actual grid --------------------------------------
    expected: set[str] = set()
    for engine in engines:
        for day in days:
            for intent in intents:
                for para in sorted(paraphrases_by_intent[intent]):
                    for rep in reps:
                        expected.add(f"{engine}-i{intent}-p{para}-d{day}-r{rep}")

    seen_counter = Counter(r["run_id"] for r in rows)
    actual = set(seen_counter)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    duplicates = sorted(rid for rid, n in seen_counter.items() if n > 1)

    errored = [r for r in rows if r.get("error")]
    empty_response = [r for r in rows if not r.get("error") and r.get("response") is None]

    # ---- timestamps -------------------------------------------------
    by_day_engine_ts: dict[tuple[int, str], list[str]] = defaultdict(list)
    for r in rows:
        by_day_engine_ts[(r["day_index"], r["engine"])].append(r["ts_utc"])
    off_window: list[dict[str, Any]] = []
    for (day, engine), tss in sorted(by_day_engine_ts.items()):
        first = min(tss)
        off = _offset_minutes(first)
        if abs(off) > TARGET_WINDOW_MINUTES:
            kyiv = datetime.fromisoformat(first).astimezone(KYIV_TZ)
            off_window.append(
                {"day": day, "engine": engine, "first_ts_utc": first, "first_ts_kyiv": kyiv.isoformat(), "offset_min": round(off, 1)}
            )

    # ---- model_returned vs model_requested -------------------------
    returned_by_engine_date: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    mismatches: list[dict[str, Any]] = []
    for r in rows:
        if r.get("error") or r.get("response") is None:
            continue
        engine = r["engine"]
        date = datetime.fromisoformat(r["ts_utc"]).date().isoformat()
        returned = extract_model_returned(engine, r["response"])
        if returned is not None:
            returned_by_engine_date[engine][date][returned] += 1
        requested = model_requested_by_engine.get(engine)
        if requested and returned and requested not in returned and returned not in requested:
            mismatches.append({"run_id": r["run_id"], "engine": engine, "requested": requested, "returned": returned})

    # ---- search_fired --------------------------------------------
    sf_counts: dict[tuple[str, int], Counter] = defaultdict(Counter)
    for r in rows:
        if r.get("error") or r.get("response") is None:
            sf_counts[(r["engine"], r["intent_id"])]["skipped"] += 1
            continue
        sf = extract_search_fired(r["engine"], r["response"])
        key = "null" if sf is None else ("true" if sf else "false")
        sf_counts[(r["engine"], r["intent_id"])][key] += 1

    # ---- sources / text floor -----------------------------------
    empty_sources: list[str] = []
    short_text: list[dict[str, Any]] = []
    all_urls: list[str] = []
    for r in rows:
        if r.get("error") or r.get("response") is None:
            continue
        srcs = extract_sources_rows(r["engine"], r["response"])
        all_urls.extend(s["url_raw"] for s in srcs if s.get("url_raw"))
        if not srcs:
            empty_sources.append(r["run_id"])
        text = extract_raw_response_text(r["engine"], r["response"]) or ""
        if len(text) < RESPONSE_TEXT_FLOOR_CHARS:
            short_text.append({"run_id": r["run_id"], "engine": r["engine"], "chars": len(text)})

    unique_urls = set(all_urls)
    unique_domains = set()
    for u in unique_urls:
        try:
            from urllib.parse import urlsplit

            host = (urlsplit(u).hostname or "").lower()
            if host.startswith("www."):
                host = host[4:]
            if host:
                unique_domains.add(host)
        except Exception:
            pass

    # ---- render ---------------------------------------------------
    L: list[str] = []
    w = L.append
    w("# Collection integrity audit")
    w("")
    w(f"_Generated {datetime.now(timezone.utc).isoformat()} · read-only · source: `{rel(runs_dir)}/*.jsonl`_")
    w("")
    w(f"- Runs on disk: **{len(rows)}** lines, **{len(actual)}** unique run_id")
    w(f"- Expected grid: {len(engines)} engines × {len(days)} days × {len(intents)} intents × 5 paraphrases × {len(reps)} reps = **{len(expected)}**")
    w(f"- Errored responses: **{len(errored)}** · empty (no response, no error): **{len(empty_response)}**")
    if branded_intents:
        w(f"- Branded intents (reported separately, never pooled): {branded_intents}")
    w("")

    w("## 1. Grid coverage")
    w("")
    w(f"- Missing cells: **{len(missing)}**")
    if missing:
        for rid in missing[:200]:
            w(f"  - `{rid}`")
        if len(missing) > 200:
            w(f"  - … and {len(missing) - 200} more")
    w(f"- Duplicate run_id (retries appended): **{len(duplicates)}**")
    for rid in duplicates:
        w(f"  - `{rid}` ×{seen_counter[rid]}")
    w(f"- Unexpected run_id (not in grid): **{len(unexpected)}**")
    for rid in unexpected[:50]:
        w(f"  - `{rid}`")
    w(f"- Errored / empty responses: **{len(errored) + len(empty_response)}**")
    for r in errored:
        w(f"  - `{r['run_id']}` — {str(r.get('error'))[:300]}")
    for r in empty_response:
        w(f"  - `{r['run_id']}` — response is null, no error field")
    w("")

    w("## 2. Timestamp distribution vs the fixed fire hour")
    w("")
    w(f"Target: first run of each day/engine block at **{TARGET_UTC_HOUR:02d}:00 UTC / 17:00 Europe/Kyiv**; flag if the block start is more than {TARGET_WINDOW_MINUTES} min off.")
    w("")
    w("| day | engine | first run (UTC) | first run (Kyiv) | offset (min) |")
    w("| --- | --- | --- | --- | --- |")
    for (day, engine), tss in sorted(by_day_engine_ts.items()):
        first = min(tss)
        off = _offset_minutes(first)
        flag = " ⚠️" if abs(off) > TARGET_WINDOW_MINUTES else ""
        kyiv = datetime.fromisoformat(first).astimezone(KYIV_TZ).strftime("%H:%M")
        w(f"| {day} | {engine} | {datetime.fromisoformat(first).astimezone(timezone.utc).strftime('%m-%d %H:%M')} | {kyiv} | {off:+.0f}{flag} |")
    w("")
    if off_window:
        w("**Off-window blocks:**")
        for o in off_window:
            w(f"- Day {o['day']} / {o['engine']}: started {o['first_ts_utc']} ({o['first_ts_kyiv']} Kyiv), offset {o['offset_min']:+} min.")
    else:
        w("_No off-window blocks._")
    w("")

    w("## 3. model_returned vs model_requested")
    w("")
    for engine in engines:
        requested = model_requested_by_engine.get(engine, "(unknown)")
        w(f"### {engine} — requested `{requested}`")
        per_date = returned_by_engine_date.get(engine, {})
        if not per_date:
            w("- no model string in any response envelope for this engine")
        for date in sorted(per_date):
            counts = ", ".join(f"`{m}`×{n}" for m, n in per_date[date].most_common())
            w(f"- {date}: {counts}")
        distinct = {m for c in per_date.values() for m in c}
        if len(distinct) > 1:
            w(f"- ⚠️ returned model string changed across the window: {sorted(distinct)}")
        w("")
    w(f"Direct requested/returned mismatches: **{len(mismatches)}**")
    for m in mismatches[:50]:
        w(f"- `{m['run_id']}`: requested `{m['requested']}` → returned `{m['returned']}`")
    w("")

    w("## 4. search_fired (derived, never stored)")
    w("")
    w("| engine | intent | true | false | null | skipped(err) |")
    w("| --- | --- | --- | --- | --- | --- |")
    for (engine, intent) in sorted(sf_counts):
        c = sf_counts[(engine, intent)]
        w(f"| {engine} | {intent} | {c['true']} | {c['false']} | {c['null']} | {c['skipped']} |")
    w("")
    per_engine = defaultdict(Counter)
    for (engine, _), c in sf_counts.items():
        per_engine[engine].update(c)
    for engine, c in sorted(per_engine.items()):
        total = c["true"] + c["false"] + c["null"]
        if total:
            w(f"- **{engine}**: false-rate {c['false'] / total:.1%}, null-rate {c['null'] / total:.1%} (n={total})")
    w("")

    w("## 5. Empty source lists and short responses")
    w("")
    w(f"- Runs with an empty source list (successful response, zero citations): **{len(empty_sources)}**")
    w(f"- Runs with response text below {RESPONSE_TEXT_FLOOR_CHARS} chars: **{len(short_text)}**")
    for s in short_text[:50]:
        w(f"  - `{s['run_id']}` ({s['engine']}, {s['chars']} chars)")
    w("")

    w("## 6. Cited URL / domain cardinality")
    w("")
    w(f"- Unique cited URLs across all source lists: **{len(unique_urls)}**")
    w(f"- Unique host names (naive registrable, www-stripped): **{len(unique_domains)}**")
    w("- Note: Gemini citations are `vertexaisearch…redirect` wrappers; the true publisher domain count is lower until Task 5 resolves them.")
    w("")

    w("## 7. Errors, retries, rate-limit events")
    w("")
    w(f"- Error rows in the JSONL: **{len(errored)}**")
    for r in errored:
        w(f"  - `{r['run_id']}` — {str(r.get('error'))[:400]}")
    w(f"- Duplicate run_id (the only retry signal in an append-only log): **{len(duplicates)}**")
    w("- No separate retry / rate-limit event log exists — the instrument delegates retries to each vendor SDK and records only the final outcome.")
    w("")

    w("## 8. Permanent gaps (cannot be fixed retroactively)")
    w("")
    w("- **`instrument_commit`** was never written to the JSONL line (documented gap vs INSTRUMENT.md §2). The git commit that produced each row is not recoverable per-row.")
    w("- **`latency_ms`** is computed by every adapter but discarded by the JSONL write path — lost for the entire window.")
    w("- **Once-per-window auxiliary state** (`domain_redirect_snapshot`, `ranking_snapshot`, `robots_ai_crawler_block`) was never collected during the window; no collector existed. The post-window equivalents in this batch reflect site state on the collection date, not during the window.")
    w("- **Population / inclusion frame** was not frozen before collection (INSTRUMENT.md §8.3 / §14).")
    w("")

    w("## Judgment calls for the ambiguity ledger")
    w("")
    w(f"- Response-text floor set at {RESPONSE_TEXT_FLOOR_CHARS} chars (arbitrary; {len(short_text)} runs fall below).")
    w(f"- Empty-source-list runs ({len(empty_sources)}) are kept as-is — a real answer with no citations is data, not an error.")
    w(f"- The {len(errored)} errored run(s) were never retried; the grid is treated as {len(expected) - len(missing)}/{len(expected)} present with {len(errored)} recorded failure(s).")
    w("")
    return "\n".join(L)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    p.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    p.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--stdout", action="store_true", help="also print the report to stdout")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(args.runs_dir, args.prompts, args.experiment_config)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(f"[audit] wrote {rel(args.report)} ({len(report.splitlines())} lines)")
    if args.stdout:
        print(report)


if __name__ == "__main__":
    main()
