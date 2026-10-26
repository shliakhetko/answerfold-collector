#!/usr/bin/env python3
"""Phase B entry command.

Single command, no manual steps, no execution order required of the
operator: frozen upstream artifacts in, every registered figure out, as
one serialized file. Running it twice on unchanged inputs produces
byte-identical figures (see analysis/phase_b/serialize.py's docstring for
the one field -- generated_at -- that legitimately differs). This is the
reproducibility claim; a skeptical reader runs exactly this.

Usage:
    GEO_DATA_ROOT=/path/to/frozen/data uv run python -m scripts.results.run_phase_b
    uv run python -m scripts.results.run_phase_b --data-root /path/to/frozen/data [--output FILE]

The data path has NO default: unset or missing stops the build. Every
manifest (Phase A, raw window, aux, ui_test) is verified against the files
under the data path before anything is computed; any mismatch or missing
file stops the build.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analysis.phase_b import domain_ownership, intent_map
from analysis.phase_b.citation_panel import build_citation_panel
from analysis.phase_b.daily_panel import build_daily_panel
from analysis.phase_b.figures import FIGURES
from analysis.phase_b.rates_panel import build_rates_panel
from analysis.phase_b.crawler_precheck import crawler_precheck
from analysis.phase_b.reading import PhaseBInputError, read_frozen_inputs, resolve_data_root
from analysis.phase_b.run_company_panel import build_run_company_panel
from analysis.phase_b.serialize import write_seam_file

from collector import paths
REPO_ROOT = paths.REPO_ROOT
OUTPUT_PATH = paths.FIGURES_JSON


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase B build")
    parser.add_argument("--data-root", default=None, help="frozen data directory (else $GEO_DATA_ROOT; no default)")
    parser.add_argument("--output", default=None, help="figures.json output path (default: data/04_results/figures.json)")
    args = parser.parse_args(argv if argv is not None else [])
    output_path = Path(args.output) if args.output else OUTPUT_PATH
    try:
        data_root = resolve_data_root(args.data_root)
        print(f"data root: {data_root}")
        inputs = read_frozen_inputs(REPO_ROOT, data_root)
    except PhaseBInputError as exc:
        print(f"Phase B stopped: {exc}", file=sys.stderr)
        return 1

    print(f"read {len(inputs.raw_runs)} raw runs, {len(inputs.entities)} entities, {len(inputs.mentions_raw)} mentions")

    # Frog block first action: day/rep completeness, printed before anything
    # else (even the intent map) -- "build the daily panel and check every
    # engine has all 7 day_index values with 3 reps each. Print missing
    # cells." A known, already-documented gap (see daily_panel.
    # KNOWN_INCOMPLETE_CELLS) is reported but does not fail this print step;
    # an unwhitelisted one is a hard stop for the whole block.
    _daily_cells, daily_panel_report = build_daily_panel(inputs)
    completeness = daily_panel_report.completeness
    if completeness.unwhitelisted_missing:
        print(f"missing cells (UNEXPECTED, {len(completeness.unwhitelisted_missing)}):", file=sys.stderr)
        for gap in completeness.unwhitelisted_missing:
            print(f"  {gap}", file=sys.stderr)
        print(
            "Per the task's stop rule: write this failure to reports/ambiguity_ledger.md and stop -- "
            "do not proceed to build V2/A2/O1/O2 in this pass.",
            file=sys.stderr,
        )
        return 1
    print(f"missing cells: none unexpected (checked {completeness.total_cells_expected} engine x intent x paraphrase x day cells)")
    if completeness.known_incomplete:
        print(f"  known, already-documented gaps ({len(completeness.known_incomplete)}): {completeness.known_incomplete}")

    # Intent map: read + assert first, before any other number (2026-09-24,
    # "do not hardcode {1..6}" review feedback) -- see
    # analysis/phase_b/intent_map.py's docstring for why this asserts
    # against citation_panel.UNBRANDED_INTENT_IDS rather than replacing it.
    try:
        resolved_intent_map = intent_map.read_intent_map(REPO_ROOT)
        intent_map.assert_intent_map(resolved_intent_map)
    except intent_map.IntentMapError as exc:
        print(f"Phase B stopped: {exc}", file=sys.stderr)
        return 1
    print(f"intent map (intent_id -> branded): {resolved_intent_map}")

    # Rates panel qualifying counts: printed next, still before any other
    # intermediate or figure number, per the task's "qualifying company
    # count per engine, first" rule.
    _rate_cells, rates_report = build_rates_panel(inputs)
    print(f"rates panel qualifying companies by engine (>=5 mentions, unbranded intents): {rates_report.qualifying_by_engine}")

    _ownership_rows, build_report = domain_ownership.build_source_ownership(
        inputs.mentions_raw,
        inputs.entities,
        inputs.alt_domains_rows,
        inputs.sources_by_run,
        inputs.raw_runs,
        inputs.entity_verification_rows,
    )
    print("source-ownership intermediate build report:")
    print(f"  rows produced:              {build_report.rows_produced}")
    print(f"  distinct responses covered: {build_report.distinct_responses_covered}")
    print(f"  distinct entities covered:  {build_report.distinct_entities_covered}")
    print(f"  true/false/null:            {build_report.owned_true_false_null}")
    print(f"  entities with no domain:    {build_report.entities_with_no_domain}")
    print(f"  entities with alternates:   {build_report.entities_with_alternates}")
    print(f"  matches via alternate:      {build_report.matches_via_alternate}")
    print(f"  matches via subdomain:      {build_report.matches_via_subdomain}")
    print(f"  responses w/ zero sources:  {build_report.responses_with_zero_sources}")
    print(f"  malformed inputs:           {build_report.malformed_input_count}")
    print(f"  denominator before/after exclusion: {build_report.denominator_before_exclusion} / {build_report.denominator_after_exclusion}")
    print(f"  coverage reconcile:         {build_report.coverage_reconcile_note}")
    print(f"  owned=True share by engine: {build_report.owned_true_share_by_engine}")
    print(
        f"  gemini citations total/resolved/unresolved: {build_report.gemini_citations_total} / "
        f"{build_report.gemini_citations_resolved} / {build_report.gemini_citations_unresolved}"
    )
    print(f"  gemini unresolved note:     {build_report.gemini_unresolved_note}")
    print(
        f"  gemini owned-share current/tight-upper/naive-upper: "
        f"{build_report.gemini_owned_share_current:.1%} / {build_report.gemini_owned_share_tight_upper_bound:.1%} / "
        f"{build_report.gemini_owned_share_naive_upper_bound:.1%}"
    )
    print(
        f"  gemini title recovery residual (post-fix, expect ~0): "
        f"{build_report.gemini_title_recovery_residual_recoverable_rows} rows still recoverable -> "
        f"{build_report.gemini_title_recovery_residual_owned_share:.1%} if they were"
    )
    print(f"  gemini title recovery note: {build_report.gemini_title_recovery_note}")
    print(
        f"  entities_with_no_domain 560->0 split: dropped(out-of-frame)={build_report.entities_dropped_by_in_frame_fix} "
        f"via_homepage_fallback={build_report.entities_gained_domain_via_homepage_fallback} "
        f"via_alt_domains_only={build_report.entities_gained_domain_via_alt_domains_only} "
        f"still_none={build_report.entities_still_with_no_domain_after_fix}"
    )

    _citation_rows, citation_panel_report = build_citation_panel(inputs)
    print("citation-panel intermediate build report:")
    print(f"  runs in scope (usable, intents 1-6): {citation_panel_report.runs_in_scope}")
    print(f"  rows produced:                       {citation_panel_report.rows_produced}")
    print(f"  url_class counts:                    {citation_panel_report.url_class_counts}")
    print(
        f"  unclassified (title_fallback/other): {citation_panel_report.citations_unclassified_title_fallback} / "
        f"{citation_panel_report.citations_unclassified_other}"
    )

    _run_company_rows, _eligibility, run_company_panel_report = build_run_company_panel(inputs)
    print("run-company-panel intermediate build report:")
    print(f"  rows produced (full cross product):  {run_company_panel_report.rows_produced}")
    print(f"  mentioned rows:                       {run_company_panel_report.mentioned_rows}")
    print(f"  own_domain_cited true/false/null:     {run_company_panel_report.own_domain_cited_true_false_null}")
    print(f"  rows flipped false->none (vs B1):     {run_company_panel_report.rows_flipped_false_to_none}")
    print(f"  qualifying entities by engine:        {run_company_panel_report.qualifying_entities_by_engine}")

    figure_results = {figure_id: compute(inputs) for figure_id, compute in FIGURES.items()}

    write_seam_file(
        output_path,
        inputs_manifest=inputs.inputs_manifest,
        figure_results=figure_results,
        intermediates={
            "intent_map": resolved_intent_map,
            "source_ownership": build_report.to_json(),
            "citation_panel": citation_panel_report.to_json(),
            "run_company_panel": run_company_panel_report.to_json(),
            "rates_panel": rates_report.to_json(),
            "daily_panel": daily_panel_report.to_json(),
            "crawler_precheck": crawler_precheck(inputs.roster_rows, inputs.robots_by_domain),
        },
    )
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
