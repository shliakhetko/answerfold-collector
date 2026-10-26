"""K1 -- Company mention concentration (PRE-SPECIFIED TABLE, pinned
EXPERIMENT.md / Notion Part 3.2).

Unit: the mention (not the citation -- K2's unit -- a different figure at
a different level). Denominator: all mentions in the six unbranded
intents, per engine -- built from analysis.phase_b.rates_panel.
build_rates_panel's RateCell.n_mentioned, summed per (entity, engine)
across intents, reused rather than a fresh join over run_company_panel.

Figures: top-5 companies' share of all mentions in that engine, and the
count of companies needed to reach 50% (the quotable integer).

**Flat-baseline comparison, both shown, neither picked** (2026-09-24,
K1 flat-baseline ambiguity, logged in the ambiguity ledger): the spec's own
comparison number ("a flat distribution across 50 companies puts the top 5
at 10%") is a worked illustrative example, not a formula tied to this
study's real counts -- so both a literal reading (5 / 415, the frozen
in-frame population) and a per-engine reading (5 / distinct companies
actually mentioned at least once in that engine) are computed and printed
against the observed share. Neither is silently preferred.

"N out-of-frame brands named" per engine: unlike every other B1-derived
figure, this join is built from surface_form_to_entity_id over the FULL,
unfiltered entity_registry (not in_frame_entities()) -- a mention of an
out-of-frame or unresolved entity's surface form has no entry in the usual
in-frame-only map and would otherwise vanish silently. Bucketed by
meets_frame (out/borderline/unresolved), plus a residual bucket for
surface forms matching no registry row at all. Counts distinct entities,
not raw mention volume, per the spec's own phrasing ("N ... brands").
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from analysis.coding.b1_per_run_company import surface_form_to_entity_id
from analysis.phase_b.citation_panel import UNBRANDED_INTENT_IDS
from analysis.phase_b.exclusions import usable_run_ids
from analysis.phase_b.figure_shape import FigureResult
from analysis.phase_b.rates_panel import build_rates_panel
from analysis.phase_b.reading import FrozenInputs

FIGURE_ID = "K1"
TOP_N = 5
# EXPERIMENT.md K1, verbatim: "a flat distribution across 50 companies puts
# the top 5 at 10%" -- the spec's own worked illustrative number, kept as a
# fixed reference alongside the per-engine recomputation, never replaced by
# it (see module docstring).
FLAT_BASELINE_FIXED_N = 415  # frozen in-frame population (round2-fix-c-population-size-confirmed)
FLAT_BASELINE_SPEC_EXAMPLE_N = 50
FLAT_BASELINE_SPEC_EXAMPLE_SHARE = TOP_N / FLAT_BASELINE_SPEC_EXAMPLE_N


def _concentration_from_counts(counts: dict[str, int]) -> dict[str, Any]:
    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top_n_count = sum(c for _, c in ranked[:TOP_N])
    top_n_share = (top_n_count / total) if total else None

    count_to_50pct = None
    cumulative_share_curve: list[float] = []
    cumulative = 0
    for i, (_, c) in enumerate(ranked, start=1):
        cumulative += c
        cumulative_share_curve.append(cumulative / total if total else 0.0)
        if count_to_50pct is None and total and cumulative / total >= 0.5:
            count_to_50pct = i

    return {
        "total_mentions": total,
        "distinct_companies_mentioned": len(ranked),
        "top_n_share": top_n_share,
        "count_to_50pct": count_to_50pct,
        "top_n_ranked": [{"entity_id": e, "count": c, "share": c / total} for e, c in ranked[:TOP_N]] if total else [],
        "cumulative_share_curve": cumulative_share_curve,
    }


def _out_of_frame_named_by_engine(
    inputs: FrozenInputs, usable_ids: set[str], unbranded_intent_ids: frozenset[int]
) -> dict[str, dict[str, Any]]:
    sf_to_entity_all = surface_form_to_entity_id(inputs.entities)
    entities_by_id = {e["entity_id"]: e for e in inputs.entities}
    run_meta = {r["run_id"]: r for r in inputs.raw_runs}

    entities_by_engine_bucket: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    unmatched_by_engine: dict[str, set[str]] = defaultdict(set)

    for m in inputs.mentions_raw:
        run_id = m.get("run_id")
        if run_id not in usable_ids:
            continue
        meta = run_meta.get(run_id)
        if meta is None or meta.get("intent_id") not in unbranded_intent_ids:
            continue
        engine = meta.get("engine")
        entity_id = sf_to_entity_all.get(m.get("surface_form"))
        if entity_id is None:
            unmatched_by_engine[engine].add(m.get("surface_form"))
            continue
        entity = entities_by_id.get(entity_id)
        frame = entity.get("meets_frame") if entity else None
        if frame == "in":
            continue  # in-frame -- not an out-of-frame brand, not counted here
        bucket = frame if frame in ("out", "borderline", "unresolved") else "unresolved"
        entities_by_engine_bucket[engine][bucket].add(entity_id)

    engines = set(entities_by_engine_bucket) | set(unmatched_by_engine)
    result: dict[str, dict[str, Any]] = {}
    for engine in engines:
        buckets = entities_by_engine_bucket.get(engine, {})
        all_out_of_frame = buckets.get("out", set()) | buckets.get("borderline", set()) | buckets.get("unresolved", set())
        result[engine] = {
            "out": len(buckets.get("out", set())),
            "borderline": len(buckets.get("borderline", set())),
            "unresolved": len(buckets.get("unresolved", set())),
            "unmatched_surface_forms": len(unmatched_by_engine.get(engine, set())),
            "total_out_of_frame_entities": len(all_out_of_frame),
        }
    return result


def compute(inputs: FrozenInputs) -> FigureResult:
    cells, _rates_report = build_rates_panel(inputs)
    if not cells:
        return FigureResult(
            figure_id=FIGURE_ID,
            status="untestable",
            value=None,
            denominator_before_exclusion=None,
            denominator_after_exclusion=None,
            n_included=None,
            n_excluded=None,
            threshold=None,
            reason_untestable="zero rate cells (no usable run in the unbranded intents mentioning any in-frame entity)",
        )

    mentions_by_engine_entity: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for c in cells:
        mentions_by_engine_entity[c.engine][c.entity_id] += c.n_mentioned

    engines = sorted(mentions_by_engine_entity)
    usable_ids = usable_run_ids(inputs.raw_runs)
    out_of_frame_named = _out_of_frame_named_by_engine(inputs, usable_ids, UNBRANDED_INTENT_IDS)

    value: dict[str, float | None] = {}
    denominator_before_exclusion: dict[str, int] = {}
    denominator_after_exclusion: dict[str, int] = {}
    n_included: dict[str, int] = {}
    n_excluded: dict[str, dict[str, int]] = {}
    count_to_50pct: dict[str, int | None] = {}
    top_n_ranked: dict[str, list[dict[str, Any]]] = {}
    cumulative_share_curve: dict[str, list[float]] = {}
    flat_baseline_recomputed_by_engine: dict[str, float | None] = {}
    distinct_companies_mentioned_by_engine: dict[str, int] = {}

    for engine in engines:
        counts = {e: n for e, n in mentions_by_engine_entity[engine].items() if n > 0}
        stats = _concentration_from_counts(counts)

        value[engine] = stats["top_n_share"]
        denominator_before_exclusion[engine] = stats["total_mentions"]
        denominator_after_exclusion[engine] = stats["total_mentions"]
        n_included[engine] = stats["total_mentions"]
        n_excluded[engine] = {}
        count_to_50pct[engine] = stats["count_to_50pct"]
        top_n_ranked[engine] = stats["top_n_ranked"]
        cumulative_share_curve[engine] = stats["cumulative_share_curve"]
        distinct_companies_mentioned_by_engine[engine] = stats["distinct_companies_mentioned"]
        flat_baseline_recomputed_by_engine[engine] = (
            TOP_N / stats["distinct_companies_mentioned"] if stats["distinct_companies_mentioned"] else None
        )

    return FigureResult(
        figure_id=FIGURE_ID,
        status="ok",
        value=value,
        denominator_before_exclusion=denominator_before_exclusion,
        denominator_after_exclusion=denominator_after_exclusion,
        n_included=n_included,
        n_excluded=n_excluded,
        threshold=None,  # pre-specified table, no prediction (Notion Part 3.0/3.2)
        reason_untestable=None,
        notes={
            "count_to_50pct": count_to_50pct,
            "top_5_ranked": top_n_ranked,
            "cumulative_share_curve": cumulative_share_curve,
            "distinct_companies_mentioned_by_engine": distinct_companies_mentioned_by_engine,
            "flat_baseline_fixed": {
                "n": FLAT_BASELINE_FIXED_N,
                "share": TOP_N / FLAT_BASELINE_FIXED_N,
            },
            "flat_baseline_spec_example": {
                "n": FLAT_BASELINE_SPEC_EXAMPLE_N,
                "share": FLAT_BASELINE_SPEC_EXAMPLE_SHARE,
            },
            "flat_baseline_recomputed_by_engine": flat_baseline_recomputed_by_engine,
            "out_of_frame_named_by_engine": out_of_frame_named,
        },
    )
