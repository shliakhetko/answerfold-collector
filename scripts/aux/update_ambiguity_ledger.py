#!/usr/bin/env python3
"""Task 8 -- ambiguity ledger.

Append-only JSONL at data/02_aux/ambiguity_ledger.jsonl, rendered to
reports/ambiguity_ledger.md. Each entry: id, date, situation, decision,
reason. Re-running is idempotent (entries are keyed by `id`); pass --add to
append a new ad-hoc entry.

Usage:
    uv run python -m scripts.aux.update_ambiguity_ledger
    uv run python -m scripts.aux.update_ambiguity_ledger --add \\
        --date 2026-08-30 --situation "..." --decision "..." --reason "..."
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from analysis.aux.storage import AuxStore, DEFAULT_AUX_ROOT
from scripts.aux._run import rel

from collector import paths
DEFAULT_REPORT = paths.AMBIGUITY_LEDGER_MD

SEED_ENTRIES = [
    {
        "id": "population-definition-change-2026-08-22",
        "date": "2026-08-22",
        "situation": "The study population had been defined as a frozen roster of vendors chosen before collection.",
        "decision": "Switched to an open-ended, post-hoc discovered population bounded by a frozen inclusion frame: any vendor named in answer text is a candidate; membership is decided later by the frame, not by a pre-registered list.",
        "reason": "A frozen roster cannot capture vendors that only appear once the engines are queried; the discovered-population-with-frozen-frame design keeps the inclusion rule pre-committed while letting the roster grow.",
    },
    {
        "id": "schedule-deviation-day4-2026-08-23",
        "date": "2026-08-23",
        "situation": "The day-4 (2026-08-23) run block started at 17:01 UTC / 20:01 Europe/Kyiv instead of the fixed 14:00 UTC / 17:00 Kyiv fire hour -- an offset of ~181 minutes. All four engines are affected equally (they run in one block).",
        "decision": "The day-4 runs are kept as valid observations; the deviation is recorded here and surfaced in reports/collection_audit.md rather than the day being re-run or dropped.",
        "reason": "Re-running would introduce a second same-day sample under different conditions; dropping loses a full day. The fixed-hour rule (INSTRUMENT.md section 4) exists to hold time-of-day constant, and a one-off 3h shift on one day is a documented limitation, not a data-quality failure.",
    },
    {
        "id": "aux-batch-collected-post-window-2026-08-29",
        "date": "2026-08-29",
        "situation": "All auxiliary data in this batch (Google SERP grid, robots/llms files, cited-URL resolution, site inventories, directory presence) is collected 3 days after the window closed (2026-08-26).",
        "decision": "Every auxiliary record carries post_window=true and days_after_window_close, and reflects site state on the collection date -- it is used as a post-window snapshot, never backdated into the window.",
        "reason": "The once-per-window auxiliary collection was never run during the window (no collector existed). A post-window snapshot is the best available proxy; labelling it as such prevents it being mistaken for in-window state.",
    },
    {
        "id": "daily-redirect-snapshot-never-run",
        "date": "2026-08-29",
        "situation": "INSTRUMENT.md section 8.4 calls for a daily domain-redirect snapshot; it was never run for any company, and companies discovered mid-window would in any case have fewer daily snapshots than seed companies.",
        "decision": "No daily redirect history exists. Task 4's post-window fetch captures the redirect chain once, on the collection date, for every roster domain.",
        "reason": "The daily cadence cannot be reconstructed after the fact; a single post-window capture is what remains possible.",
    },
    {
        "id": "instrument-commit-and-latency-not-recorded",
        "date": "2026-08-29",
        "situation": "The JSONL write path never recorded instrument_commit (INSTRUMENT.md section 2) or latency_ms, though the adapters compute latency.",
        "decision": "Both are treated as permanently missing for the window. The collection-close tag pins the instrument code as a whole; per-row commit traceability is not recoverable.",
        "reason": "Neither value can be derived from an already-written line.",
    },
    {
        "id": "roster-source-is-trial-db",
        "date": "2026-08-29",
        "situation": "The verified roster depends on running the A1-A5 coding pipeline against the real window with a real inclusion frame; that has not happened. Per the user's decision the collectors point at data/_trial/coding_trial.sqlite3 (a different vertical: AI meeting note-takers) for now.",
        "decision": "The auxiliary collectors are built and validated against the trial roster; they are idempotent, so re-running them against the real roster later fetches only the new rows. Trial-vertical auxiliary data is a wiring artifact, not study data.",
        "reason": "Unblocks building and testing the collectors without waiting on the (billed) real-window pipeline run.",
    },
    {
        "id": "task5-resolved-urls-feed-coding-pipeline-2026-08-31",
        "date": "2026-08-31",
        "situation": "The coding pipeline consumed the raw cited URL. For Gemini that is a vertexaisearch grounding-redirect wrapper -- ~6.9k distinct URLs that carry no signal for A5's URL classifier and never match a vendor domain in B1's own_domain_cited join.",
        "decision": "run_coding_pipeline now loads data/02_aux/cited_urls/resolution.jsonl (Task 5) and, in memory, swaps each reachable citation's url_raw/domain for its resolved final destination before A3/A5/B1/B2. The raw citation is kept on url_cited and the full mapping persisted to the cited_url_resolution table. Citations that resolved 404/410/dead/timeout/redirect-loop are recorded by A5 as 'other' with no model call. --no-url-resolution restores the old behaviour.",
        "reason": "Halves A5's unique-URL count (~9.7k -> ~4.1k) and its spend, removes a large class of meaningless classifications, and fixes a B1 correctness gap (a Gemini citation of a vendor's own site was structurally impossible to detect). The collector sources table is untouched, so the raw record is unchanged.",
    },
    {
        "id": "a4prime-entity-verification-stage-2026-08-31",
        "date": "2026-08-31",
        "situation": "A3 resolves surface forms into entities and guesses a homepage only from co-cited domains; neither A3 nor A4 checks whether an entity is a real, discrete vendor. The trial run left real vendors (Sembly, Chorus, Bluedot) at meets_frame=0 purely because A4 had no homepage to fetch, while platform/feature/publisher/directory noise (Google Workspace, G2, 'Built-in Zoom notes') and likely hallucinations (Meatgeeek, Rev Winner) reached A4.",
        "decision": "Added A4-prime (module a3b_entity_verification.py) between A3 and A4: one search-backed call per entity returning is_real, entity_type (vendor/platform/feature/publisher/directory/generic/unknown), a discovered homepage_url, and a disambiguation note, persisted to the entity_verification table. A4 runs its frozen-frame check only on entities confirmed is_real AND entity_type=='vendor', using the discovered homepage; everything else is recorded meets_frame=False with no frame call. Missing verification = fail open (A4 proceeds as before). Off with --skip-entity-verification.",
        "reason": "The frozen coding spec has no reality/discreteness gate and no URL-discovery step, so real vendors were being excluded by a data gap rather than by judgment, and non-companies were entering the classification path. This stage is NOT in the frozen spec and changes what meets_frame=0 can mean -- hence this entry. It is only meaningful once the real inclusion frame and real-window coding run exist; against the trial DB it is wiring validation.",
    },
    {
        "id": "a1-detached-batch-run-2026-08-31",
        "date": "2026-08-31",
        "situation": "A full coding run against the 2,940-run window is ~12-15 h synchronous; the built-in --use-batch-api path still submits, polls and collects inside one process (must stay running, 1 h hard timeout).",
        "decision": "Added scripts/coding/submit_coding_batches.py + scripts/coding/collect_coding_batches.py: submit persists the A1 primary-extractor Batch API job ids to a session file and exits (machine can be off); collect resumes -- polls, pulls results into the same LLM cache, runs the arbiter batch, then A3->A4-prime->A4->A5->B. A1's consensus math and every later stage are byte-identical to the synchronous path; only the transport for the primary + arbiter extractor calls changes, and the model ids are the same.",
        "reason": "Lets the expensive, slow A1 extraction run unattended overnight without the machine staying awake, and removes the 1 h batch-timeout failure mode. No effect on what the pipeline computes.",
    },
    {
        "id": "response-text-floor-and-empty-sources",
        "date": "2026-08-29",
        "situation": "The audit needs a 'too short to be a real answer' threshold and a policy for successful runs with zero citations.",
        "decision": "Response-text floor set at 200 characters (reporting only, nothing dropped). Empty-source-list runs are kept as-is -- a real answer with no citations is data.",
        "reason": "Both are arbitrary reporting conveniences; keeping everything and flagging it preserves the raw-only discipline.",
    },
]


def load_existing(store: AuxStore) -> list[dict]:
    return store.read_records(store.path("ambiguity_ledger.jsonl"))


def _field(label: str, text: str) -> str:
    # A body that starts with a newline (list-style entries) renders the label
    # with no trailing space, so such entries round-trip byte for byte.
    sep = "" if text.startswith("\n") else " "
    return f"**{label}.**{sep}{text}"


def _sort_key(e: dict) -> tuple:
    # Date first. Within a day, entries with no "order" field sort by id (the
    # original rule); entries carrying an integer "order" (the 2026-10-04
    # backfill of hand-appended markdown entries, order = position in the
    # markdown) follow them in that order, so the markdown reproduces exactly.
    if "order" in e:
        return (e.get("date", ""), 1, e["order"], "")
    return (e.get("date", ""), 0, 0, e.get("id", ""))


def render_md(entries: list[dict]) -> str:
    L = ["# Ambiguity ledger", "", "_Date · situation · decision · reason. Append-only._", ""]
    for e in sorted(entries, key=_sort_key):
        L.append(f"## {e.get('date', '?')} — {e.get('id', '')}")
        L.append("")
        L.append(_field("Situation", e.get("situation", "")))
        L.append("")
        L.append(_field("Decision", e.get("decision", "")))
        L.append("")
        L.append(_field("Reason", e.get("reason", "")))
        L.append("")
    return "\n".join(L)


def md_headings(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.startswith("## ")]


def write_report_guarded(report: Path, new_text: str) -> list[str]:
    """Write `new_text` to `report` unless that would drop an existing heading.

    Returns the list of headings that would have disappeared (empty when the
    write happened). Warns on stdout; never raises for a heading loss.
    """
    if report.exists():
        old_headings = md_headings(report.read_text(encoding="utf-8"))
        new_set = set(md_headings(new_text))
        missing = [h for h in old_headings if h not in new_set]
        if missing:
            print(
                f"[ledger] WARNING: not writing {rel(report)} -- {len(missing)} existing "
                "heading(s) would disappear (the jsonl is missing those entries):"
            )
            for h in missing:
                print(f"[ledger]   missing: {h}")
            return missing
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(new_text, encoding="utf-8")
    return []


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aux-root", type=Path, default=DEFAULT_AUX_ROOT)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--add", action="store_true", help="append one ad-hoc entry from the flags below")
    p.add_argument("--id", default=None)
    p.add_argument("--date", default=date.today().isoformat())
    p.add_argument("--situation", default=None)
    p.add_argument("--decision", default=None)
    p.add_argument("--reason", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    store = AuxStore(args.aux_root)
    ledger_path = store.path("ambiguity_ledger.jsonl")

    existing = load_existing(store)
    known_ids = {e.get("id") for e in existing}

    to_append = [e for e in SEED_ENTRIES if e["id"] not in known_ids]

    if args.add:
        if not (args.situation and args.decision and args.reason):
            raise SystemExit("--add requires --situation, --decision, --reason")
        slug = args.id or f"adhoc-{args.date}-{abs(hash(args.situation)) % 10000}"
        if slug not in known_ids:
            to_append.append(
                {
                    "id": slug,
                    "date": args.date,
                    "situation": args.situation,
                    "decision": args.decision,
                    "reason": args.reason,
                }
            )

    for entry in to_append:
        store.append_record(ledger_path, entry)

    all_entries = load_existing(store)
    write_report_guarded(args.report, render_md(all_entries))

    print(f"[ledger] appended {len(to_append)} entr{'y' if len(to_append) == 1 else 'ies'}; {len(all_entries)} total")
    print(f"[ledger] {rel(ledger_path)}  +  {rel(args.report)}")


if __name__ == "__main__":
    main()
