"""A1 -- SEO top-10 vs mention rate (CONFIRMATORY, pinned EXPERIMENT.md /
Notion Part 3.1).

Claim: a company's Google top-10 organic presence for an intent's head
phrase does not predict its mention rate for that intent. Unit: company x
intent, per engine, qualifying companies only (the eligibility floor
applies to every per-company figure, Notion Part 3.0).

google_top10 is derived by analysis.coding.b3b_serp_derivation.derive_serp_binary
from analysis.phase_b.serp_loader's head-phrase (paraphrase P01) SERP
capture, joined to entity domains -- reused verbatim, not reimplemented.
mention_rate comes from analysis.phase_b.rates_panel.build_rates_panel,
reused verbatim, not recomputed.

Intent-inclusion rule (spec, verbatim): "an intent enters the pooled
estimate only if between 20% and 80% of qualifying companies are top-10
for it. Below or above that the binary carries no information." Applied
per engine, since "qualifying companies" is itself a per-engine set.
Excluded intents are named in notes, per engine, with the fraction that
triggered the exclusion.

SERP collection limitation (analysis.phase_b.serp_loader.
SERP_COLLECTION_LIMITATION) is carried on the figure record itself, not
only in the ambiguity ledger -- the task requires it show up on the
rendered figure.

An engine with too few qualifying companies, too few included intents, or
no variance in google_top10/mention_rate to compute a real correlation is
marked untestable for A1 specifically (its own entry in
notes["untestable_engines"]) without stopping the whole figure -- other
engines still compute normally.

**2026-09-24 review feedback: "untestable in every engine" needed context,
not just a fact.** Google returns at most 10 organic slots per intent;
with 83/53/92/73 qualifying companies across the four engines, the
maximum possible top-10 fraction (10 / n_qualifying) tops out at
10.9%-18.9% -- always below the 20% inclusion floor. That ceiling is
computed and stated on the figure (notes["structural_ceiling_by_engine"]),
a per-intent SERP-slot breakdown checks whether a single company ever
holds more than one of Google's matched slots
(notes["intent_top10_slot_detail"]), observed-vs-maximum is reported for
every unbranded intent, not only the excluded ones
(notes["top10_fraction_by_engine_intent"]), and one EXPLORATORY panel
(notes["exploratory_top10_vs_not"]) reports mean mention rate for top-10
vs not-top-10 companies pooled across ALL unbranded intents -- no 20-80%
inclusion filter, since that filter is a confirmatory-only rule and this
panel makes no confirmatory claim. The confirmatory value/status/threshold
fields are untouched; this is purely additive context, never worded as
support for or against A1's confirmatory claim.

**2026-09-24 round-4 cleanup (second pass):** the four near-identical
per-engine ceiling sentences are replaced by a single summary table
(notes["structural_ceiling_by_engine"] gained "highest_observed_fraction"
so the table can show max-possible next to highest-observed for every
engine, no arbitrary cutoff). The `gap_beyond_slot_limit_by_engine` flag
(which only ever fired for one of four engines behind a 50%-of-ceiling
cutoff) is removed entirely -- the summary table already shows this
comparison as raw numbers for every engine uniformly. The SERP-slot table
gained intent theme names (from EXPERIMENT.md's own intent table) and a
"non-vendor slots" count. The exploratory panel's entity_ids are now
listed per engine (notes["exploratory_top10_vs_not"][engine]["entity_ids"]),
and its disclaimer text was rewritten to state the real, actionable
limitation (group too small to test, big brands rank on Google and get
named by AI for related reasons) instead of the vaguer "claims no
direction."
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from scipy.stats import pointbiserialr

from analysis.coding.b1_per_run_company import _domain_of, entity_domains, in_frame_entities
from analysis.coding.b3b_serp_derivation import derive_serp_binary
from analysis.phase_b.citation_panel import UNBRANDED_INTENT_IDS
from analysis.phase_b.figure_shape import FigureResult
from analysis.phase_b.rates_panel import build_rates_panel
from analysis.phase_b.reading import FrozenInputs
from analysis.phase_b.serp_loader import SERP_COLLECTION_LIMITATION

FIGURE_ID = "A1"
# Notion Part 3.1 (A1), verbatim: "Prediction: r below 0.3. Falsifier: r at
# or above 0.5, the conventional large-effect line. Band 0.3 to 0.5
# inconclusive."
PREDICTION_BELOW = 0.3
FALSIFIER_AT_OR_ABOVE = 0.5
# Notion Part 3.1 (A1), verbatim: "an intent enters the pooled estimate
# only if between 20% and 80% of qualifying companies are top-10 for it."
INTENT_INCLUSION_LOW = 0.20
INTENT_INCLUSION_HIGH = 0.80
MIN_SAMPLES_FOR_CORRELATION = 3  # below this, r is not a meaningful statistic
TOP10_SLOTS = 10  # Google's organic top-10 cap -- the structural ceiling's numerator
# 2026-09-24 round-4 rewrite: states the real, actionable limitation
# (group size, a real confound) instead of the vaguer "claims no direction."
EXPLORATORY_DISCLAIMER = (
    "Descriptive only. Top-10 companies show higher mention rates here, but the top-10 group is too small "
    "to test anything (its pair counts are in exploratory_top10_vs_not), and big brands tend to rank on "
    "Google and get named by AI for the same reasons. Not evidence for or against C3."
)
# EXPERIMENT.md's own intent table (Post-Collection Coding Pipeline Spec /
# prompts.csv's prompt set) -- reused for display, not invented ad hoc.
INTENT_THEME_NAMES = {
    1: "general", 2: "dental", 3: "legal", 4: "home services", 5: "pricing", 6: "capability",
}
# Verified today against every one of the 6 unbranded intents' raw SERP
# captures: all 6 include a Google AI Overview box, and 3 of 6 also show
# an inline video carousel -- both are known to reduce the organic-result
# count SerpApi returns below the classic 10, which is why raw slots run
# 7-9 here, never a full 10.
SERP_SLOT_EXPLANATION = (
    "Raw slots run 7-9, never a full 10, because every one of these SERPs includes a Google AI Overview "
    "box (all 6 unbranded intents), and 3 of 6 also show an inline video carousel -- both reduce the "
    "organic-result count Google (and therefore SerpApi) returns."
)


def _verdict(r: float | None) -> str | None:
    if r is None:
        return None
    if r < PREDICTION_BELOW:
        return "supported"
    if r >= FALSIFIER_AT_OR_ABOVE:
        return "falsified"
    return "inconclusive"


def _structural_ceiling_sentence(engine: str, n_qual: int, ceiling: float) -> str:
    return (
        f"At most {TOP10_SLOTS} Google slots exist per intent; with {n_qual} qualifying companies in "
        f"{engine}, the maximum possible top-10 fraction is {ceiling:.1%}, below the {INTENT_INCLUSION_LOW:.0%} "
        "floor the inclusion rule requires -- untestable here is a consequence of this design, not a "
        "finding about SEO visibility."
    )


def _intent_top10_slot_detail(inputs: FrozenInputs) -> dict[int, dict[str, Any]]:
    """Per unbranded intent (engine-independent -- Google's top 10 for an
    intent doesn't depend on which AI engine is being scored): how many of
    the <=10 organic-result slots Google actually returned
    (n_raw_slots -- confirmed on this data to be 7-9, never a full 10), how
    many resolve to ANY in-frame entity's domain
    (n_slots_matched_any_in_frame_entity), how many DISTINCT entities that
    represents (n_distinct_companies_matched), and how many matched slots
    are "extra" for a company that already holds one
    (n_duplicate_company_slots = the difference) -- a positive number means
    at least one company held more than one of Google's matched slots for
    that intent (e.g. two pages on its own domain), which would otherwise
    silently inflate a naive per-slot count into looking like more
    distinct companies appeared than actually did.

    Reuses the same domain-matching helpers
    analysis/coding/b3b_serp_derivation.py's derive_serp_binary() itself
    uses (in_frame_entities, entity_domains, _domain_of) -- no new
    matching rule invented."""
    entities = in_frame_entities(inputs.entities)
    domains_by_entity = entity_domains(entities, inputs.alt_domains_rows)

    detail: dict[int, dict[str, Any]] = {}
    for intent_id in sorted(UNBRANDED_INTENT_IDS):
        rows = [r for r in inputs.serp_raw_rows if r["intent_id"] == intent_id]
        matched_entity_ids: list[str] = []
        for row in rows:
            domain = _domain_of(row["url"])
            for entity_id, domains in domains_by_entity.items():
                if domain in domains:
                    matched_entity_ids.append(entity_id)
                    break
        distinct = set(matched_entity_ids)
        n_matched = len(matched_entity_ids)
        detail[intent_id] = {
            "intent_name": INTENT_THEME_NAMES.get(intent_id, "?"),
            "n_raw_slots": len(rows),
            "n_slots_matched_any_in_frame_entity": n_matched,
            "n_slots_non_vendor": len(rows) - n_matched,
            "n_distinct_companies_matched": len(distinct),
            "n_duplicate_company_slots": n_matched - len(distinct),
        }
    return detail


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

    serp_binary = derive_serp_binary(inputs.serp_raw_rows, inputs.entities, inputs.alt_domains_rows)
    top10_pairs = {(row["entity_id"], row["intent_id"]) for row in serp_binary}

    rate_by_cell: dict[tuple[str, int, str], float] = {
        (c.entity_id, c.intent_id, c.engine): c.mention_rate for c in cells
    }
    qualifying_by_engine: dict[str, set[str]] = defaultdict(set)
    engines_seen: set[str] = set()
    for c in cells:
        engines_seen.add(c.engine)
        if c.qualifies:
            qualifying_by_engine[c.engine].add(c.entity_id)

    value: dict[str, float | None] = {}
    denominator_before_exclusion: dict[str, int] = {}
    denominator_after_exclusion: dict[str, int] = {}
    n_included: dict[str, int] = {}
    n_excluded: dict[str, dict[str, Any]] = {}
    verdict_by_engine: dict[str, str | None] = {}
    excluded_intents_by_engine: dict[str, list[dict[str, Any]]] = {}
    included_intents_by_engine: dict[str, list[int]] = {}
    top10_fraction_by_engine_intent: dict[str, dict[int, float]] = {}
    top10_vs_not_by_engine: dict[str, dict[str, float | None]] = {}
    scatter_points_by_engine: dict[str, list[dict[str, Any]]] = {}
    structural_ceiling_by_engine: dict[str, dict[str, Any] | None] = {}
    exploratory_top10_vs_not: dict[str, dict[str, Any]] = {}
    untestable_engines: dict[str, str] = {}

    for engine in sorted(engines_seen):
        qualifying = qualifying_by_engine.get(engine, set())
        n_qual = len(qualifying)
        before = n_qual * len(UNBRANDED_INTENT_IDS)
        denominator_before_exclusion[engine] = before

        if n_qual == 0:
            value[engine] = None
            verdict_by_engine[engine] = None
            n_included[engine] = 0
            denominator_after_exclusion[engine] = 0
            n_excluded[engine] = {"reason": "zero qualifying companies"}
            untestable_engines[engine] = "zero qualifying companies for this engine"
            structural_ceiling_by_engine[engine] = None
            exploratory_top10_vs_not[engine] = {
                "mean_top10": None, "n_top10": 0, "mean_not_top10": None, "n_not_top10": 0, "entity_ids": [],
                "top10_rates": [],
            }
            continue

        ceiling = min(TOP10_SLOTS, n_qual) / n_qual

        included_intents: list[int] = []
        excluded: list[dict[str, Any]] = []
        fractions: dict[int, float] = {}
        for intent_id in sorted(UNBRANDED_INTENT_IDS):
            top10_count = sum(1 for e in qualifying if (e, intent_id) in top10_pairs)
            frac = top10_count / n_qual
            fractions[intent_id] = frac
            if INTENT_INCLUSION_LOW <= frac <= INTENT_INCLUSION_HIGH:
                included_intents.append(intent_id)
            else:
                excluded.append({"intent_id": intent_id, "top10_fraction": frac})
        excluded_intents_by_engine[engine] = excluded
        included_intents_by_engine[engine] = included_intents
        top10_fraction_by_engine_intent[engine] = fractions

        highest_observed_fraction = max(fractions.values())
        structural_ceiling_by_engine[engine] = {
            "n_qualifying": n_qual,
            "max_slots": TOP10_SLOTS,
            "theoretical_max_fraction": ceiling,
            "highest_observed_fraction": highest_observed_fraction,
            "sentence": _structural_ceiling_sentence(engine, n_qual, ceiling),
        }

        # EXPLORATORY (2026-09-24): pooled across ALL unbranded intents, no
        # 20-80% inclusion filter -- see EXPLORATORY_DISCLAIMER. A separate
        # pooling from the confirmatory x/y below, which stays scoped to
        # included_intents only.
        expl_top10_rates: list[float] = []
        expl_top10_entity_ids: set[str] = set()
        expl_not_rates: list[float] = []
        for i in sorted(UNBRANDED_INTENT_IDS):
            for e in sorted(qualifying):
                rate = rate_by_cell.get((e, i, engine))
                if rate is None:
                    continue
                if (e, i) in top10_pairs:
                    expl_top10_rates.append(rate)
                    expl_top10_entity_ids.add(e)
                else:
                    expl_not_rates.append(rate)
        exploratory_top10_vs_not[engine] = {
            "mean_top10": (sum(expl_top10_rates) / len(expl_top10_rates)) if expl_top10_rates else None,
            "n_top10": len(expl_top10_rates),
            "mean_not_top10": (sum(expl_not_rates) / len(expl_not_rates)) if expl_not_rates else None,
            "n_not_top10": len(expl_not_rates),
            "entity_ids": sorted(expl_top10_entity_ids),
            # Round 5 fix: the render script used to plot this group's mean
            # cloned once per pair as fake "dots" -- the real per-pair rates
            # were computed above but never left this function. Now they do.
            "top10_rates": list(expl_top10_rates),
        }

        x: list[int] = []  # google_top10, 0/1
        y: list[float] = []  # mention_rate
        points: list[dict[str, Any]] = []
        for intent_id in included_intents:
            for entity_id in sorted(qualifying):
                rate = rate_by_cell.get((entity_id, intent_id, engine))
                if rate is None:
                    continue
                top10 = 1 if (entity_id, intent_id) in top10_pairs else 0
                x.append(top10)
                y.append(rate)
                points.append(
                    {"entity_id": entity_id, "intent_id": intent_id, "google_top10": bool(top10), "mention_rate": rate}
                )
        scatter_points_by_engine[engine] = points

        n_included[engine] = len(x)
        denominator_after_exclusion[engine] = len(x)
        n_excluded[engine] = {"excluded_intent_pairs": before - len(x)}

        top10_rates = [rate for flag, rate in zip(x, y) if flag == 1]
        not_top10_rates = [rate for flag, rate in zip(x, y) if flag == 0]
        mean_top10 = sum(top10_rates) / len(top10_rates) if top10_rates else None
        mean_not = sum(not_top10_rates) / len(not_top10_rates) if not_top10_rates else None
        ratio = (mean_top10 / mean_not) if (mean_top10 is not None and mean_not) else None
        top10_vs_not_by_engine[engine] = {"mean_top10": mean_top10, "mean_not": mean_not, "ratio": ratio}

        distinct_x = set(x)
        if not included_intents:
            # Sharper than the generic "insufficient variation" message
            # below: this specific case (zero intents ever cleared the
            # 20-80% band) has a known structural cause, so say it (2026-09-24
            # review feedback -- "untestable" must not read as an SEO finding).
            value[engine] = None
            verdict_by_engine[engine] = None
            untestable_engines[engine] = (
                f"no intent cleared the 20-80% inclusion band -- structural ceiling: maximum possible "
                f"top-10 fraction is {ceiling:.1%} given {n_qual} qualifying companies and only "
                f"{TOP10_SLOTS} Google slots per intent; untestable by design, not a finding about SEO."
            )
        elif len(x) < MIN_SAMPLES_FOR_CORRELATION or len(distinct_x) < 2 or len(set(y)) < 2:
            value[engine] = None
            verdict_by_engine[engine] = None
            untestable_engines[engine] = (
                f"insufficient variation to compute a correlation (n={len(x)}, distinct google_top10 values="
                f"{len(distinct_x)}, distinct mention_rate values={len(set(y))})"
            )
        else:
            r, _p = pointbiserialr(x, y)
            value[engine] = float(r)
            verdict_by_engine[engine] = _verdict(float(r))

    # Round 5 fix: a fixed, engine-independent count of the exploratory
    # panel's top-10 pool -- Google's top-10 for an (entity, intent) pair
    # doesn't depend on which AI engine is being scored, so the pool is the
    # same pairs/companies for every engine's panel, unlike n_top10 above
    # (which varies 8-11 per engine because "qualifying" is a per-engine
    # set). Union across engines, not one engine's set, so a company that
    # only qualifies for one engine still counts toward the pool.
    any_engine_qualifying = set().union(*qualifying_by_engine.values()) if qualifying_by_engine else set()
    fixed_top10_pairs = {
        (e, i) for e in sorted(any_engine_qualifying) for i in UNBRANDED_INTENT_IDS if (e, i) in top10_pairs
    }
    top10_pairs_definition = {
        "n_pairs": len(fixed_top10_pairs),
        "n_companies": len({e for e, _i in fixed_top10_pairs}),
    }

    # Deliberately does NOT collapse to a blank status="untestable" record
    # even when every engine ends up untestable (e.g. every intent falls
    # outside the 20-80% inclusion band on real data, a genuine finding
    # about this sparse-visibility category, not a data-absence bug): the
    # per-engine excluded-intent fractions and the SERP-timing limitation
    # must still show up on the rendered figure per the task's own gate
    # requirement, which a collapsed record would silently drop.
    n_testable_engines = sum(1 for v in value.values() if v is not None)
    n_engines_seen = len(engines_seen)
    confirmatory_summary_sentence = (
        f"Confirmatory result: untestable in all {n_engines_seen} engines."
        if n_testable_engines == 0
        else f"Confirmatory result: testable in {n_testable_engines} of {n_engines_seen} engines."
    )

    return FigureResult(
        figure_id=FIGURE_ID,
        status="ok",
        value=value,
        denominator_before_exclusion=denominator_before_exclusion,
        denominator_after_exclusion=denominator_after_exclusion,
        n_included=n_included,
        n_excluded=n_excluded,
        threshold={"prediction_below": PREDICTION_BELOW, "falsifier_at_or_above": FALSIFIER_AT_OR_ABOVE},
        reason_untestable=None,
        notes={
            "verdict_by_engine": verdict_by_engine,
            "excluded_intents_by_engine": excluded_intents_by_engine,
            "included_intents_by_engine": included_intents_by_engine,
            "top10_fraction_by_engine_intent": top10_fraction_by_engine_intent,
            "top10_vs_not_by_engine": top10_vs_not_by_engine,
            "scatter_points_by_engine": scatter_points_by_engine,
            "untestable_engines": untestable_engines,
            "serp_collection_limitation": SERP_COLLECTION_LIMITATION,
            "structural_ceiling_by_engine": structural_ceiling_by_engine,
            "intent_top10_slot_detail": _intent_top10_slot_detail(inputs),
            "serp_slot_explanation": SERP_SLOT_EXPLANATION,
            "exploratory_top10_vs_not": exploratory_top10_vs_not,
            "top10_pairs_definition": top10_pairs_definition,
            "exploratory_disclaimer": EXPLORATORY_DISCLAIMER,
            "confirmatory_summary_sentence": confirmatory_summary_sentence,
        },
    )
