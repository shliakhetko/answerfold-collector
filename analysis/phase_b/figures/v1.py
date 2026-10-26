"""V1 -- Vertical spread against engine spread (CONFIRMATORY, pinned
EXPERIMENT.md / Notion Part 3.1).

Claim: a company's mention rate varies more across vertical cuts than
across engines. Unit: the qualifying company (the eligibility floor
applies to every per-company figure, Notion Part 3.0).

Does NOT reuse analysis.coding.b3_aggregates.vertical_vs_engine_spread --
that function computes max-min across every engine/intent for an entity,
which is neither this spec's statistic (SD, not range) nor its scope
(vertical spread restricted to intents 2/3/4, engine spread restricted to
intent 1 only, per Notion Part 3.1's own careful wording: "so the intent
is held fixed"). Reuses analysis.phase_b.rates_panel.build_rates_panel for
mention_rate, not recomputed here.

- vertical_spread (per company): population SD of mention_rate across
  intents {2, 3, 4} inside one engine, then averaged (arithmetic mean)
  over every engine.
- engine_spread (per company): population SD of mention_rate across every
  engine, computed on intent 1 only.
- ratio = vertical_spread / engine_spread. statistic = MEDIAN across
  companies (spec, verbatim: "one company with a near-zero engine spread
  produces an enormous ratio and would carry the whole result").
- Exclusion: engine_spread == 0 -- counted and reported separately from
  any missing-cell data-availability gap (a different, non-spec reason a
  company's ratio can't be computed at all).
- Two qualifying-company definitions, both computed (spec, verbatim, log
  in ledger): PRIMARY = qualifies in at least one engine (union across
  engines). SENSITIVITY = qualifies in all four engines (intersection).
  Primary is `value`; sensitivity is a full parallel record in notes,
  never silently dropped.

2026-09-24 round-4 fixes (report only -- this statistic is NOT redefined
by any of the below, per the owner's explicit instruction):
- per_entity now also carries the two raw spread components
  (vertical_spread, engine_spread), not only their ratio -- the paper
  figure plots these directly as x/y scatter coordinates rather than a
  log2-transformed ratio histogram.
- n_above_line/n_below_line/title_sentence: "above the diagonal"
  (engine_spread > vertical_spread, ratio < 1) means engine matters more
  than industry for that company; "below" is the reverse. Computed once
  here so the render script draws a title without computing anything of
  its own.
- excluded_because_zero_mentions_intent1_all_engines /
  intent1_mention_rate_distribution / zero_ratio_median_check: verification
  the owner asked for on the two largest exclusion questions (103 of 157
  companies excluded for engine_spread==0; 11 companies with ratio==0.0
  that an earlier log2-histogram render silently omitted from its x-axis,
  though never from the actual median calculation below). Confirmed
  against real data before writing this: all 103 have exactly 0.0 mean
  intent-1 mention rate in every engine; the 11 zero-ratio companies were
  always inside `ratios` (a computed ratio of exactly 0.0 has
  excluded_reason=None by construction) so the median is unchanged, not
  recomputed.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from analysis.phase_b.figure_shape import FigureResult
from analysis.phase_b.rates_panel import build_rates_panel
from analysis.phase_b.reading import FrozenInputs

FIGURE_ID = "V1"
VERTICAL_INTENT_IDS = (2, 3, 4)
# EXPERIMENT.md's own intent table (same names analysis/phase_b/figures/a1.py
# uses for its SERP-slot table) -- reused for display, not invented ad hoc.
INTENT_THEME_NAMES = {2: "dental", 3: "legal", 4: "home services"}
ENGINE_SPREAD_INTENT_ID = 1
# Notion Part 3.1 (V1), verbatim: "Prediction: median at or above 1.5.
# Falsifier: median at or below 1.0, meaning engine moves the number as
# much or more. Band 1.0 to 1.5 inconclusive."
MEDIAN_PREDICTION_AT_OR_ABOVE = 1.5
MEDIAN_FALSIFIER_AT_OR_BELOW = 1.0
# EXPERIMENT.md V1, verbatim: "Exclusion: company excluded if engine
# spread is zero. Count of exclusions reported." No exclusion rule exists
# for a zero RATIO -- a company whose vertical spread happens to be
# exactly zero is a valid, informative data point (industry does not
# matter at all for that company) and belongs in the median.
PRE_REG_EXCLUSION_RULE_QUOTE = (
    "EXPERIMENT.md V1, verbatim: \"Exclusion: company excluded if engine spread is zero. Count of "
    "exclusions reported.\" No exclusion rule exists for a zero ratio -- a company whose vertical "
    "spread is exactly zero (identical mention rate across the three vertical intents) is a valid, "
    "informative data point (industry does not matter at all for that company) and belongs in the "
    "median, not a hidden data point."
)
# Round 5 fix: state the asymmetry plainly rather than leaving it implicit
# in the two rules above.
ASYMMETRY_SENTENCE = (
    "The two zero-spread cases are handled asymmetrically, by design: a company with zero engine "
    "spread is excluded from the median (pre-registered), but a company with zero vertical spread "
    "is kept in it -- zero engine spread means the ratio can't be computed at all (division by "
    "zero), while zero vertical spread is itself a real, computable answer (industry doesn't move "
    "this company's rate)."
)


def _verdict(median: float | None) -> str | None:
    if median is None:
        return None
    if median >= MEDIAN_PREDICTION_AT_OR_ABOVE:
        return "supported"
    if median <= MEDIAN_FALSIFIER_AT_OR_BELOW:
        return "falsified"
    return "inconclusive"


def _entity_ratio(
    entity_id: str, engines: list[str], rate_by_key: dict[tuple[str, int, str], float]
) -> tuple[float | None, str | None, float | None, float | None]:
    """(ratio, exclusion_reason, vertical_spread, engine_spread). The two
    raw components are returned whenever computable, even if the ratio
    itself isn't (e.g. vertical_spread is still reported for a company
    excluded on engine_spread_zero) -- exclusion_reason is None only when
    a ratio was computed."""
    engine_sds: list[float] = []
    for engine in engines:
        values = [rate_by_key.get((entity_id, intent_id, engine)) for intent_id in VERTICAL_INTENT_IDS]
        if any(v is None for v in values):
            continue
        engine_sds.append(statistics.pstdev(values))
    if not engine_sds:
        return None, "missing_vertical_cells", None, None
    vertical_spread = statistics.fmean(engine_sds)

    engine_values = [rate_by_key.get((entity_id, ENGINE_SPREAD_INTENT_ID, engine)) for engine in engines]
    if any(v is None for v in engine_values) or len(engine_values) < 2:
        return None, "missing_engine_spread_cells", vertical_spread, None
    engine_spread = statistics.pstdev(engine_values)
    if engine_spread == 0:
        return None, "engine_spread_zero", vertical_spread, engine_spread
    return vertical_spread / engine_spread, None, vertical_spread, engine_spread


def _summarize(
    entity_ids: list[str], engines: list[str], rate_by_key: dict[tuple[str, int, str], float]
) -> dict[str, Any]:
    ratios: list[float] = []
    excluded_by_reason: dict[str, int] = defaultdict(int)
    per_entity: list[dict[str, Any]] = []
    for entity_id in sorted(entity_ids):
        ratio, reason, vertical_spread, engine_spread = _entity_ratio(entity_id, engines, rate_by_key)
        if ratio is None:
            excluded_by_reason[reason] += 1
        else:
            ratios.append(ratio)
        per_entity.append(
            {
                "entity_id": entity_id,
                "ratio": ratio,
                "excluded_reason": reason,
                "vertical_spread": vertical_spread,
                "engine_spread": engine_spread,
            }
        )

    median = statistics.median(ratios) if ratios else None
    n_above_line = sum(1 for r in ratios if r < 1)  # engine_spread > vertical_spread -- engine matters more
    n_below_line = sum(1 for r in ratios if r > 1)  # vertical_spread > engine_spread -- industry matters more
    n_on_line = sum(1 for r in ratios if r == 1)
    title_sentence = (
        f"For {n_above_line} of {len(ratios)} companies, which AI engine you ask matters more than "
        "which industry you ask about."
        if ratios
        else "No companies had a computable ratio."
    )

    return {
        "median": median,
        "verdict": _verdict(median),
        "n_companies_considered": len(entity_ids),
        "n_included": len(ratios),
        "n_excluded_by_reason": dict(excluded_by_reason),
        "n_excluded_engine_spread_zero": excluded_by_reason.get("engine_spread_zero", 0),
        "per_entity": per_entity,
        "n_above_line_engine_matters_more": n_above_line,
        "n_below_line_industry_matters_more": n_below_line,
        "n_on_line": n_on_line,
        "title_sentence": title_sentence,
    }


def _mean_rate(entity_id: str, intent_id: int, engines: list[str], rate_by_key: dict[tuple[str, int, str], float]) -> float | None:
    vals = [rate_by_key.get((entity_id, intent_id, engine)) for engine in engines]
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


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

    engines = sorted({c.engine for c in cells})
    rate_by_key = {(c.entity_id, c.intent_id, c.engine): c.mention_rate for c in cells}

    qualifying_engines_by_entity: dict[str, set[str]] = defaultdict(set)
    for c in cells:
        if c.qualifies:
            qualifying_engines_by_entity[c.entity_id].add(c.engine)

    primary_entities = [e for e, qual_engines in qualifying_engines_by_entity.items() if len(qual_engines) >= 1]
    sensitivity_entities = [
        e for e, qual_engines in qualifying_engines_by_entity.items() if qual_engines == set(engines)
    ]

    if not primary_entities:
        return FigureResult(
            figure_id=FIGURE_ID,
            status="untestable",
            value=None,
            denominator_before_exclusion=None,
            denominator_after_exclusion=None,
            n_included=None,
            n_excluded=None,
            threshold=None,
            reason_untestable="zero qualifying companies in any engine",
        )

    primary = _summarize(primary_entities, engines, rate_by_key)
    sensitivity = _summarize(sensitivity_entities, engines, rate_by_key)

    # Fix 2 verification (report only -- see module docstring).
    excluded_entities = [e["entity_id"] for e in primary["per_entity"] if e["excluded_reason"] == "engine_spread_zero"]
    excluded_all_zero = [
        eid for eid in excluded_entities
        if all(rate_by_key.get((eid, ENGINE_SPREAD_INTENT_ID, engine)) == 0.0 for engine in engines)
    ]
    excluded_nonzero_equal = [eid for eid in excluded_entities if eid not in excluded_all_zero]
    if not excluded_nonzero_equal:
        exclusion_sentence = (
            f"n excluded: {len(excluded_entities)} of {primary['n_companies_considered']}, because they "
            "were never named in the general 'best' question on any engine"
        )
    else:
        exclusion_sentence = (
            f"n excluded: {len(excluded_entities)} of {primary['n_companies_considered']} for zero engine "
            f"spread on intent 1 -- {len(excluded_all_zero)} because they were never named in the general "
            f"'best' question on any engine, {len(excluded_nonzero_equal)} because their intent-1 mention "
            "rate happened to be identical (but nonzero) across every engine"
        )

    included_entities = [e["entity_id"] for e in primary["per_entity"] if e["ratio"] is not None]
    incl_rates = [r for r in (_mean_rate(e, ENGINE_SPREAD_INTENT_ID, engines, rate_by_key) for e in included_entities) if r is not None]
    excl_rates = [r for r in (_mean_rate(e, ENGINE_SPREAD_INTENT_ID, engines, rate_by_key) for e in excluded_entities) if r is not None]

    zero_ratio_entities = [e["entity_id"] for e in primary["per_entity"] if e["ratio"] == 0.0]

    # Round 5 fix: print the actual dental/legal/home-services rates behind
    # each zero-ratio company (a ratio of 0.0 means vertical_spread == 0,
    # i.e. an identical mean rate across all three verticals -- this shows
    # what that identical rate actually is, and whether it's identical
    # because it's always zero).
    zero_ratio_vertical_rates: dict[str, dict[str, float | None]] = {}
    zero_ratio_all_zero_industry_entities: list[str] = []
    rate_group_by_key: dict[tuple[float | None, ...], list[str]] = defaultdict(list)
    for entity_id in sorted(zero_ratio_entities):
        rates_by_intent = {
            INTENT_THEME_NAMES[intent_id]: _mean_rate(entity_id, intent_id, engines, rate_by_key)
            for intent_id in VERTICAL_INTENT_IDS
        }
        zero_ratio_vertical_rates[entity_id] = rates_by_intent
        if all(v == 0.0 for v in rates_by_intent.values()):
            zero_ratio_all_zero_industry_entities.append(entity_id)
        rate_group_by_key[tuple(rates_by_intent[n] for n in INTENT_THEME_NAMES.values())].append(entity_id)

    # Grouped by identical (dental, legal, home services) rate triple, for
    # a compact display -- printing 11 identical rows would be noise; the
    # full per-entity breakdown is still in vertical_rates_by_entity above.
    zero_ratio_rate_groups = [
        {
            "dental": key[0], "legal": key[1], "home_services": key[2],
            "all_zero_industry": all(v == 0.0 for v in key),
            "entity_ids": entity_ids,
        }
        for key, entity_ids in sorted(rate_group_by_key.items(), key=lambda kv: -len(kv[1]))
    ]

    if zero_ratio_all_zero_industry_entities:
        zero_industry_absent_sentence = (
            f"{len(zero_ratio_all_zero_industry_entities)} of the {len(zero_ratio_entities)} zero-ratio "
            "companies were never mentioned in any of the three vertical intents on any engine -- for "
            "them, industry was absent from the data, not tested at a non-zero rate."
        )
    else:
        zero_industry_absent_sentence = None

    # Round 5 fix: an EXPLORATORY median with the 0/0/0 (industry-absent)
    # companies removed, computed after seeing the data -- ledgered as
    # exploratory, post-data. The pre-registered value/verdict above (over
    # every included ratio, 0/0/0 companies included) is untouched.
    ratios_excluding_zero_industry = [
        e["ratio"]
        for e in primary["per_entity"]
        if e["ratio"] is not None and e["entity_id"] not in zero_ratio_all_zero_industry_entities
    ]
    exploratory_median_excl_zero_industry = (
        statistics.median(ratios_excluding_zero_industry) if ratios_excluding_zero_industry else None
    )

    return FigureResult(
        figure_id=FIGURE_ID,
        status="ok",
        value=primary["median"],
        denominator_before_exclusion=primary["n_companies_considered"],
        denominator_after_exclusion=primary["n_included"],
        n_included=primary["n_included"],
        n_excluded=primary["n_excluded_by_reason"],
        threshold={
            "prediction_at_or_above": MEDIAN_PREDICTION_AT_OR_ABOVE,
            "falsifier_at_or_below": MEDIAN_FALSIFIER_AT_OR_BELOW,
        },
        reason_untestable=None,
        notes={
            "verdict": primary["verdict"],
            "n_excluded_engine_spread_zero": primary["n_excluded_engine_spread_zero"],
            "per_entity": primary["per_entity"],
            "n_above_line_engine_matters_more": primary["n_above_line_engine_matters_more"],
            "n_below_line_industry_matters_more": primary["n_below_line_industry_matters_more"],
            "n_on_line": primary["n_on_line"],
            "title_sentence": primary["title_sentence"],
            "sensitivity_all_4_engines": sensitivity,
            "engines": engines,
            "vertical_intent_ids": list(VERTICAL_INTENT_IDS),
            "engine_spread_intent_id": ENGINE_SPREAD_INTENT_ID,
            "excluded_because_zero_mentions_intent1_all_engines": {
                "n_excluded_total": len(excluded_entities),
                "n_all_zero": len(excluded_all_zero),
                "n_nonzero_equal_rate": len(excluded_nonzero_equal),
                "sentence": exclusion_sentence,
            },
            "intent1_mention_rate_distribution": {
                "included": {
                    "n": len(incl_rates),
                    "mean": statistics.fmean(incl_rates) if incl_rates else None,
                    "median": statistics.median(incl_rates) if incl_rates else None,
                },
                "excluded": {
                    "n": len(excl_rates),
                    "mean": statistics.fmean(excl_rates) if excl_rates else None,
                    "median": statistics.median(excl_rates) if excl_rates else None,
                },
            },
            "zero_ratio_median_check": {
                "n_zero_ratio": len(zero_ratio_entities),
                "entity_ids": zero_ratio_entities,
                "always_included_in_median": True,
                "pre_reg_exclusion_rule_quote": PRE_REG_EXCLUSION_RULE_QUOTE,
                "vertical_rates_by_entity": zero_ratio_vertical_rates,
                "rate_groups": zero_ratio_rate_groups,
                "n_all_zero_industry": len(zero_ratio_all_zero_industry_entities),
                "all_zero_industry_entity_ids": zero_ratio_all_zero_industry_entities,
                "zero_industry_absent_sentence": zero_industry_absent_sentence,
            },
            "asymmetry_sentence": ASYMMETRY_SENTENCE,
            "exploratory_median_excluding_zero_industry": {
                "median": exploratory_median_excl_zero_industry,
                "verdict": _verdict(exploratory_median_excl_zero_industry),
                "n_included": len(ratios_excluding_zero_industry),
                "n_excluded_zero_industry": len(zero_ratio_all_zero_industry_entities),
                "is_exploratory": True,
                "computed_post_data": True,
                "note": (
                    "EXPLORATORY, computed after seeing the data -- does not replace the pre-registered "
                    "median/verdict above, which includes every zero-ratio company."
                ),
            },
        },
    )
