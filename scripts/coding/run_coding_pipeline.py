#!/usr/bin/env python3
"""CLI entry point: run every Phase A/B coding-pipeline stage, end to end,
against one runs-dir of already-collected JSONL, all output landing in
SQLite (see analysis/coding/*.py for each stage's own docstring -- this
module only chains them).

Chains: load_jsonl_dir (analysis.aggregate.load_runs -- populates the
collector's own runs/sources tables) -> A1 (run_a1) -> A2 (the
surface_forms view, no write step) -> A3 (resolve_entities) -> alt_domains
(build_alt_domains) -> A4-prime / a3b (verify_entities, search-backed
is-real-vendor + homepage discovery; --skip-entity-verification to bypass)
-> A4 (classify_entities, real homepage fetch) -> A5
(classify_urls) -> B1 (build_per_run_company) -> B2 (build_per_run_rollups)
-> B3 (apply_eligibility_floor + build_cell_rates), every stage's output
persisted via analysis.coding.storage into the SAME database load_jsonl_dir
wrote runs/sources into -- see the SQL migration plan for why (joinable
results, no hand-parsed CSV/JSONL for anything computed).

With --use-batch-api (and always in the detached collect flow), the three
per-item Anthropic stages -- A4-prime verification, A4 classification, A5
URL classification -- run through the Message Batches API (50% cheaper,
async): each submits one batch, waits inline up to --verification-batch-timeout,
and on timeout exits so the next invocation resumes the parked batch with no
re-charge. A3 is a single call and always runs synchronously.

Assumes db_path's `runs` table only ever holds rows loaded from this one
runs_dir (true by construction: each db_path is dedicated to one runs_dir,
e.g. data/_trial/coding_trial.sqlite3 <-> data/_trial/runs/) -- B1-B3 process every run
currently in that table, not just ones from this invocation.

B3b (SERP derivation) and B4 (figures) are out of scope here: no
SERP-capture mechanism exists in collector/ yet, and config/figure_formulas.md
is still a content stub (see that file).

Every real LLM/HTTP call site is an injectable parameter (extractors,
entity_resolution_call_fn, classification_call_fn, url_classification_call_fn,
fetcher) -- same DI convention as scripts/collect/run_collection.py's
registry.ADAPTERS injection point -- so tests exercise the full chain with
zero network calls.

Usage:
    uv run python -m scripts.coding.run_coding_pipeline \\
        --runs-dir data/_trial/runs --db-path data/_trial/coding_trial.sqlite3 \\
        --unbranded-intents 1,2 \\
        --inclusion-frame-path config/trial/inclusion_frame.yaml
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from analysis.aggregate.load_runs import load_jsonl_dir
from analysis.coding import storage
from analysis.coding.a1_answer_coding import EXTRACTORS, PRIMARY_EXTRACTORS, run_a1
from analysis.coding.a3_entity_resolution import call_entity_resolution, resolve_entities
from analysis.coding.a3b_entity_verification import (
    build_verification_context,
    call_entity_verification,
    verify_entities,
)
from analysis.coding.a4_entity_classification import (
    DEFAULT_SEED_LIST_PATH,
    HomepageFetcher,
    call_entity_classification,
    classify_entities,
)
from analysis.coding.a5_url_classification import call_url_classification, classify_urls
from analysis.coding.alt_domains import DEFAULT_SEED_PATH as ALT_DOMAINS_DEFAULT_SEED_PATH
from analysis.coding.alt_domains import build_alt_domains
from analysis.coding.b1_per_run_company import _domain_of, build_per_run_company
from analysis.coding.b2_per_run_rollups import build_per_run_rollups
from analysis.coding.b3_aggregates import apply_eligibility_floor, build_cell_rates
from analysis.coding.cache import LLMCache
from analysis.coding.cited_url_resolution import (
    DEFAULT_RESOLUTION_PATH,
    load_resolution_map,
    resolve_sources_by_run,
)
from analysis.coding.db import init_coding_schema
from analysis.coding.homepage_fetch import RealHomepageFetcher
from collector import paths
from collector.runner.execute import get_instrument_commit
from collector.storage import api
from collector.storage.db import get_connection, init_db

DEFAULT_INCLUSION_FRAME_PATH = paths.INCLUSION_FRAME_YAML
DEFAULT_CACHE_DIR = paths.LLM_CACHE_DIR


@dataclass
class PipelineReport:
    # A1 (always populated)
    runs_processed: int
    runs_skipped_no_text: int
    mentions_written: int
    drop_rate_by_extractor: dict[str, float | None]
    arbiter_calls_made: int
    arbiter_candidates_checked: int
    arbiter_candidates_confirmed: int
    cited_url_resolution: dict[str, int]
    # None = full pipeline ran; "a1" = stopped after A1 (fields below are stubs)
    stopped_after: str | None = None
    # A3/A4/A4-prime/B (default = stopped early / not reached)
    entities_resolved: int = 0
    in_frame_count: int = 0
    out_of_frame_count: int = 0
    # segment is only meaningful for in-frame entities -- never pool
    # human_staffed with the AI-voice segments (config/inclusion_frame.yaml).
    segment_counts: dict[str, int] = field(default_factory=dict)
    qualifying_by_engine: dict[str, list[str]] = field(default_factory=dict)
    cell_rates: list[dict[str, Any]] = field(default_factory=list)
    entity_verification: dict[str, Any] = field(default_factory=lambda: {"ran": False})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _segment_counts(entity_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in entity_rows:
        if row.get("meets_frame") != "in":
            continue
        seg = row.get("segment") or "unknown"
        counts[seg] = counts.get(seg, 0) + 1
    return counts


def _verification_stats(
    verification_by_entity: dict[str, dict[str, Any]] | None, skip_entity_verification: bool
) -> dict[str, Any]:
    verification_rows = list((verification_by_entity or {}).values())
    by_type: dict[str, int] = {}
    for v in verification_rows:
        by_type[v.get("entity_type") or "unknown"] = by_type.get(v.get("entity_type") or "unknown", 0) + 1
    return {
        "ran": not skip_entity_verification,
        "verified": len(verification_rows),
        "real_vendors": sum(1 for v in verification_rows if v.get("is_real") and v.get("entity_type") == "vendor"),
        "excluded_non_vendor": sum(
            1 for v in verification_rows if not (v.get("is_real") and v.get("entity_type") == "vendor")
        ),
        "urls_discovered": sum(1 for v in verification_rows if v.get("verified_homepage_url")),
        "parse_failures": sum(1 for v in verification_rows if v.get("is_real") is None),
        "by_type": by_type,
    }


def _write_report(report: "PipelineReport", report_path: Path | None, db_path: Path) -> Path:
    path = report_path or db_path.parent / f"{db_path.stem}_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


STOP_AFTER_CHOICES = ("a1", "a3", "a3b", "a4", "a5")


def _sources_by_run(conn) -> dict[str, list[dict[str, Any]]]:
    rows = conn.execute("SELECT run_id, position, url_raw, domain, title FROM sources ORDER BY run_id, position").fetchall()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["run_id"], []).append(
            {"position": row["position"], "url_raw": row["url_raw"], "domain": row["domain"], "title": row["title"]}
        )
    return grouped


def run_pipeline(
    *,
    runs_dir: Path,
    db_path: Path,
    unbranded_intent_ids: set[int],
    inclusion_frame_path: Path,
    seed_list_path: Path = DEFAULT_SEED_LIST_PATH,
    alt_domains_seed_path: Path = ALT_DOMAINS_DEFAULT_SEED_PATH,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    cited_url_resolution_path: Path | None = DEFAULT_RESOLUTION_PATH,
    report_path: Path | None = None,
    instrument_commit: str | None = None,
    extractors: dict[str, Callable[..., Any]] = EXTRACTORS,
    entity_resolution_call_fn: Callable[..., Any] = call_entity_resolution,
    verification_call_fn: Callable[..., Any] = call_entity_verification,
    classification_call_fn: Callable[..., Any] = call_entity_classification,
    url_classification_call_fn: Callable[..., Any] = call_url_classification,
    fetcher: HomepageFetcher | None = None,
    skip_entity_verification: bool = False,
    use_batch_api: bool = False,
    a1_offline: bool = False,
    run_limit: int | None = None,
    eligibility_floor: bool = True,
    # inline wait budget per batched stage (A4-prime, A4, A5); on timeout the
    # stage raises TimeoutError and the caller re-runs later (batch resumes).
    verification_batch_max_wait_seconds: float = 3600.0,
    stop_after: str | None = None,
) -> PipelineReport:
    commit = instrument_commit or get_instrument_commit()
    fetcher = fetcher or RealHomepageFetcher()

    init_db(db_path)
    init_coding_schema(db_path)
    load_jsonl_dir(runs_dir, db_path=db_path, instrument_commit=commit)

    conn = get_connection(db_path)
    try:
        # per_run_company.entity_id and entity_verification.entity_id both
        # REFERENCE entities(entity_id) -- on a rerun against a db_path that
        # already has a previous run's output, resolve_entities()'s
        # clear_entities() (called below, in A3) would otherwise fail with a
        # FOREIGN KEY constraint error: the earlier run's rows still
        # reference the entities about to be deleted, and they aren't
        # cleared+rewritten until later in this function. Clearing them here,
        # before anything else runs, keeps every rerun starting from a clean
        # slate for this pipeline's own outputs (not runs/sources, which
        # load_jsonl_dir handles with its own incremental/skip-existing
        # semantics).
        storage.clear_per_run_company(conn=conn)
        storage.clear_entity_verification(conn=conn)

        run_rows = api.list_runs(conn=conn)  # already list[dict], see collector.storage.api
        if run_limit is not None:
            run_rows = run_rows[:run_limit]  # small-test knob; B1-B3 then only see these runs
        run_meta_by_id = {
            r["run_id"]: {"engine": r["engine"], "intent_id": r["intent_id"], "day_index": r["day_index"]}
            for r in run_rows
        }
        sources_by_run = _sources_by_run(conn)

        # Task 5 overlay: swap each reachable citation for its resolved
        # final URL before A3/A5/B1/B2 run. Collapses Gemini's grounding-
        # redirect wrappers (~6.9k distinct, zero-signal URLs) down to their
        # real publishers -- roughly halves A5's unique-URL count -- and
        # lets B1's own_domain_cited join actually match a Gemini citation
        # of a vendor's own site. No-op (returns the sources unchanged) when
        # the aux batch never ran. See analysis/coding/cited_url_resolution.py.
        resolution_map = load_resolution_map(cited_url_resolution_path)
        sources_by_run, resolution_stats = resolve_sources_by_run(sources_by_run, resolution_map)
        storage.clear_cited_url_resolution(conn=conn)
        if resolution_map:
            storage.insert_cited_url_resolution(
                [
                    {
                        "cited_url": rc.cited_url,
                        "final_url": rc.final_url,
                        "final_domain": rc.final_domain,
                        "status_bucket": rc.status_bucket,
                    }
                    for rc in resolution_map.values()
                ],
                conn=conn,
            )

        # A1 -- arbiter_cache is a separate directory (not folded into
        # a1_extractors/) so scripts/coding/report_coding_pipeline_cost.py reports
        # the arbiter's cost (ARBITER_EXTRACTOR, run synchronously) as its
        # own stage, separate from the two primary Batch jobs.
        a1_stats = run_a1(
            run_rows,
            cache=LLMCache(cache_dir=cache_dir / "a1_extractors"),
            arbiter_cache=LLMCache(cache_dir=cache_dir / "a1_arbiter"),
            extractors=extractors,
            use_batch_api=use_batch_api or a1_offline,  # offline finalize is the batched code path
            offline=a1_offline,
            conn=conn,
        )

        def _stop(label: str, *, verification_by_entity=None) -> PipelineReport:
            conn.commit()
            entity_rows = storage.list_entities(conn=conn)
            rep = PipelineReport(
                runs_processed=a1_stats.runs_processed,
                runs_skipped_no_text=len(run_rows) - a1_stats.runs_processed,
                mentions_written=a1_stats.mentions_written,
                drop_rate_by_extractor={n: a1_stats.drop_rate(n) for n in PRIMARY_EXTRACTORS},
                arbiter_calls_made=a1_stats.arbiter_calls_made,
                arbiter_candidates_checked=a1_stats.arbiter_candidates_checked,
                arbiter_candidates_confirmed=a1_stats.arbiter_candidates_confirmed,
                cited_url_resolution=resolution_stats.to_dict(),
                stopped_after=label,
                entities_resolved=len(entity_rows),
                in_frame_count=sum(1 for r in entity_rows if r.get("meets_frame") == "in"),
                out_of_frame_count=sum(1 for r in entity_rows if r.get("meets_frame") not in (None, "in")),
                segment_counts=_segment_counts(entity_rows),
                entity_verification=_verification_stats(verification_by_entity, skip_entity_verification),
            )
            _write_report(rep, report_path, db_path)
            return rep

        if stop_after == "a1":
            return _stop("a1")  # finally: closes conn

        # A2 is the surface_forms view (analysis/coding/schema.sql) -- no
        # write step; A3 queries it directly via storage.list_surface_forms.

        # A3
        resolve_entities(
            sources_by_run=sources_by_run,
            cache=LLMCache(cache_dir=cache_dir / "a3_entity_resolution"),
            call_fn=entity_resolution_call_fn,
            conn=conn,
        )

        # alt_domains (frozen-output gap-fill, see analysis/coding/alt_domains.py)
        build_alt_domains(alt_domains_seed_path, conn=conn)

        if stop_after == "a3":
            return _stop("a3")

        # A4-prime (a3b) -- search-backed: is this a real, discrete vendor,
        # and what is its homepage? Gates whether A4 runs its frozen-frame
        # check on the entity at all. Not in the frozen spec (see the
        # ambiguity ledger); --skip-entity-verification turns it off.
        verification_by_entity: dict[str, dict[str, Any]] | None = None
        if not skip_entity_verification:
            verification_context = build_verification_context(
                storage.list_entities(conn=conn), storage.list_mentions_raw(conn=conn), sources_by_run
            )
            verify_entities(
                context_by_entity=verification_context,
                cache=LLMCache(cache_dir=cache_dir / "a3b_entity_verification"),
                call_fn=verification_call_fn,
                use_batch=use_batch_api or a1_offline,  # A4-prime batches whenever A1 does
                batch_max_wait_seconds=verification_batch_max_wait_seconds,
                conn=conn,
            )
            verification_by_entity = {
                r["entity_id"]: r for r in storage.list_entity_verification(conn=conn)
            }

        if stop_after == "a3b":
            return _stop("a3b", verification_by_entity=verification_by_entity)

        # A4 -- real homepage fetch, enriches the entities rows A3 wrote.
        classify_entities(
            inclusion_frame_path=inclusion_frame_path,
            seed_list_path=seed_list_path,
            fetcher=fetcher,
            cache=LLMCache(cache_dir=cache_dir / "a4_entity_classification"),
            call_fn=classification_call_fn,
            verification_by_entity=verification_by_entity,
            use_batch=use_batch_api or a1_offline,  # A4 batches whenever A1 does
            batch_max_wait_seconds=verification_batch_max_wait_seconds,
            conn=conn,
        )

        if stop_after == "a4":
            return _stop("a4", verification_by_entity=verification_by_entity)

        # A5
        all_sources = [source for sources in sources_by_run.values() for source in sources]
        classify_urls(
            all_sources,
            cache=LLMCache(cache_dir=cache_dir / "a5_url_classification"),
            call_fn=url_classification_call_fn,
            use_batch=use_batch_api or a1_offline,  # A5 batches whenever A1 does
            batch_max_wait_seconds=verification_batch_max_wait_seconds,
            conn=conn,
        )

        if stop_after == "a5":
            return _stop("a5", verification_by_entity=verification_by_entity)

        # B1/B2/B3 -- pure joins/arithmetic, unchanged by the SQL migration;
        # only their inputs (queried here) and outputs (persisted below) moved.
        entity_registry_rows = storage.list_entities(conn=conn)
        alt_domains_rows = storage.list_alt_domains(conn=conn)
        url_registry_rows = storage.list_url_registry(conn=conn)
        mentions_raw_rows = storage.list_mentions_raw(conn=conn)

        # url_registry is keyed by full normalized URL (may include a path);
        # B2's url_class_by_domain is keyed by bare domain -- collapse down,
        # first-seen wins on a same-domain collision.
        url_class_by_domain: dict[str, str] = {}
        for row in url_registry_rows:
            domain = _domain_of(row["url_normalized"])
            url_class_by_domain.setdefault(domain, row["url_class"])

        per_run_company_rows = build_per_run_company(
            run_rows, mentions_raw_rows, entity_registry_rows, sources_by_run, alt_domains_rows
        )
        per_run_rollup_rows = build_per_run_rollups(
            per_run_company_rows, run_rows, mentions_raw_rows, sources_by_run, entity_registry_rows, url_class_by_domain
        )
        # The floor set is always computed (cheap, pure) and persisted to
        # eligibility_qualifying for reference -- eligibility_floor=False only
        # stops it gating which cells build_cell_rates emits.
        eligibility = apply_eligibility_floor(per_run_company_rows, run_meta_by_id, unbranded_intent_ids)
        cells = build_cell_rates(
            per_run_company_rows, run_meta_by_id, eligibility, restrict_to_eligible=eligibility_floor
        )

        storage.clear_per_run_company(conn=conn)
        if per_run_company_rows:
            storage.insert_per_run_company(per_run_company_rows, conn=conn)
        storage.clear_per_run_rollups(conn=conn)
        if per_run_rollup_rows:
            storage.insert_per_run_rollups(per_run_rollup_rows, conn=conn)
        storage.clear_eligibility_qualifying(conn=conn)
        storage.insert_eligibility_qualifying(eligibility.qualifying_by_engine, conn=conn)
        storage.clear_cell_rates(conn=conn)
        if cells:
            storage.insert_cell_rates(cells, conn=conn)

        conn.commit()
    finally:
        conn.close()

    in_frame_count = sum(1 for row in entity_registry_rows if row.get("meets_frame") == "in")
    out_of_frame_count = len(entity_registry_rows) - in_frame_count

    verification_stats = _verification_stats(verification_by_entity, skip_entity_verification)

    report = PipelineReport(
        runs_processed=a1_stats.runs_processed,
        runs_skipped_no_text=len(run_rows) - a1_stats.runs_processed,
        mentions_written=a1_stats.mentions_written,
        drop_rate_by_extractor={name: a1_stats.drop_rate(name) for name in PRIMARY_EXTRACTORS},
        arbiter_calls_made=a1_stats.arbiter_calls_made,
        arbiter_candidates_checked=a1_stats.arbiter_candidates_checked,
        arbiter_candidates_confirmed=a1_stats.arbiter_candidates_confirmed,
        entities_resolved=len(entity_registry_rows),
        in_frame_count=in_frame_count,
        out_of_frame_count=out_of_frame_count,
        segment_counts=_segment_counts(entity_registry_rows),
        qualifying_by_engine={engine: sorted(ids) for engine, ids in eligibility.qualifying_by_engine.items()},
        cell_rates=[asdict(c) for c in cells],
        cited_url_resolution=resolution_stats.to_dict(),
        entity_verification=verification_stats,
    )

    _write_report(report, report_path, db_path)
    return report


def _print_summary(report: PipelineReport) -> None:
    print(f"runs processed: {report.runs_processed} (skipped, no text: {report.runs_skipped_no_text})")
    print(f"mentions written (consensus): {report.mentions_written}")
    if report.stopped_after:
        print(f"** stopped after {report.stopped_after.upper()} -- later stages not run **")
    for name, rate in report.drop_rate_by_extractor.items():
        print(f"  drop rate [{name}]: {rate if rate is not None else 'n/a'}")
    print(
        f"arbiter: {report.arbiter_calls_made} calls, "
        f"{report.arbiter_candidates_confirmed}/{report.arbiter_candidates_checked} disputed candidates confirmed"
    )
    print(
        f"entities resolved: {report.entities_resolved} "
        f"(in-frame: {report.in_frame_count}, out-of-frame: {report.out_of_frame_count})"
    )
    if report.segment_counts:
        print(f"  in-frame segments: {report.segment_counts}")
    ev = report.entity_verification
    if ev.get("ran"):
        print(
            f"entity verification (A4-prime): {ev['real_vendors']} real vendors, "
            f"{ev['excluded_non_vendor']} excluded, {ev['urls_discovered']} homepages found"
            + (f", {ev['parse_failures']} parse failures" if ev.get("parse_failures") else "")
        )
        print(f"  by type: {ev.get('by_type', {})}")
    else:
        print("entity verification (A4-prime): skipped")
    for engine, entity_ids in report.qualifying_by_engine.items():
        print(f"  eligibility-floor qualifying [{engine}]: {entity_ids}")
    print(f"cell rates computed: {len(report.cell_rates)}")
    cur = report.cited_url_resolution
    print(
        f"cited-URL resolution: {cur['citations_swapped']}/{cur['citations_total']} citations swapped, "
        f"{cur['citations_dead']} dead; unique URLs for A5 {cur['unique_urls_before']} -> {cur['unique_urls_after']}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument(
        "--unbranded-intents", type=str, required=True, help="comma-separated intent ids, e.g. 1,2,3,4,5,6"
    )
    parser.add_argument("--inclusion-frame-path", type=Path, default=DEFAULT_INCLUSION_FRAME_PATH)
    parser.add_argument("--seed-list-path", type=Path, default=DEFAULT_SEED_LIST_PATH)
    parser.add_argument("--alt-domains-seed-path", type=Path, default=ALT_DOMAINS_DEFAULT_SEED_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--cited-url-resolution-path", type=Path, default=DEFAULT_RESOLUTION_PATH,
        help="Task 5 cited-URL resolution JSONL (data/02_aux/cited_urls/resolution.jsonl). "
             "Reachable citations are swapped for their resolved final URL before A3/A5/B1/B2. "
             "Missing file is fine -- the pipeline then runs on the raw citations.",
    )
    parser.add_argument(
        "--no-url-resolution", action="store_true",
        help="Ignore --cited-url-resolution-path entirely; feed A3/A5/B1/B2 the raw cited URLs "
             "(Gemini grounding-redirect wrappers included).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="process only the first N runs (small-test knob; B1-B3 then only see those runs).",
    )
    parser.add_argument(
        "--verification-batch-timeout", type=float, default=3600.0,
        help="with --use-batch-api: seconds to wait inline for each batched stage (A4-prime "
             "verification, A4 classification, A5 URL classification). On timeout the run exits "
             "and resumes that batch on the next invocation (no re-charge). Default 3600.",
    )
    parser.add_argument(
        "--stop-after", choices=STOP_AFTER_CHOICES, default=None,
        help="run only up to this stage, then stop. 'a1' = consensus mentions (mentions_raw). "
             "'a3' = + entity resolution. 'a3b' = + entity verification (the Sonnet 'real company + "
             "homepage URL' pass) -- the last stage that does NOT need config/inclusion_frame.yaml. "
             "'a4' = + frozen-frame classification (needs the real frame). 'a5' = + URL classification. "
             "Omit to run through B.",
    )
    parser.add_argument(
        "--skip-eligibility-floor", action="store_true",
        help="do not gate cell_rates by the spec's '>=5 mentions across unbranded intents' floor -- "
             "emit a cell for every in-frame entity x intent x engine (zero-mention cells included). "
             "The floor set is still computed and stored in eligibility_qualifying for reference.",
    )
    parser.add_argument(
        "--skip-entity-verification", action="store_true",
        help="Skip A4-prime (a3b): the search-backed is-this-a-real-vendor / find-its-homepage "
             "pass between A3 and A4. A4 then classifies every resolved entity as it did before "
             "this stage existed.",
    )
    parser.add_argument(
        "--use-batch-api", action="store_true",
        help="A1 submits each primary extractor's runs as one vendor Batch API job instead of one "
             "synchronous call per run -- a flat 50%% discount at the cost of async turnaround (see "
             "run_a1's docstring). The arbiter always runs synchronously over just the disputed runs. "
             "Off by default so a small/interactive run stays synchronous.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    unbranded_intent_ids = {int(x) for x in args.unbranded_intents.split(",") if x.strip()}
    report = run_pipeline(
        runs_dir=args.runs_dir,
        db_path=args.db_path,
        unbranded_intent_ids=unbranded_intent_ids,
        inclusion_frame_path=args.inclusion_frame_path,
        seed_list_path=args.seed_list_path,
        alt_domains_seed_path=args.alt_domains_seed_path,
        cache_dir=args.cache_dir,
        cited_url_resolution_path=None if args.no_url_resolution else args.cited_url_resolution_path,
        skip_entity_verification=args.skip_entity_verification,
        use_batch_api=args.use_batch_api,
        run_limit=args.limit,
        eligibility_floor=not args.skip_eligibility_floor,
        verification_batch_max_wait_seconds=args.verification_batch_timeout,
        stop_after=args.stop_after,
    )
    _print_summary(report)


if __name__ == "__main__":
    main()
