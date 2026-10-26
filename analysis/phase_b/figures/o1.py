"""O1 -- Top quartile stability (PRE-SPECIFIED TABLE, pinned EXPERIMENT.md /
Notion Part 3.2).

Reads the same pooled company-day rate as A2
(analysis.phase_b.daily_panel.company_day_rates /
qualifying_entities_by_engine) -- both figures need "a company's mention
rate that day," and both should agree on what that means.

- Set: per engine, per day, the top HEAD_SIZE=13 qualifying companies by
  that day's rate ("Keep 13 as pre-registered even though N is 415" --
  the ambiguity ledger already carries this exact tension, entry
  o1-integer-kept-k1-baseline-recomputed-2026-09-12; this figure
  cross-references that entry rather than re-litigating it, per
  instruction). Ranking pool for a given day is whichever qualifying
  companies actually have a rate that day -- a company absent that day
  (no usable cell) is simply not ranked, never imputed.
- Tail: the bottom half of the engine's qualifying companies (n_qualifying
  // 2, floor), same ranking.
- Statistic: mean Jaccard overlap across the 6 CONSECUTIVE day pairs
  (1,2)...(6,7), computed separately for the head sets and the tail sets.
- Figure-specific floor: an engine with fewer than HEAD_SIZE qualifying
  companies can't form the top-13 set at all and is untestable for O1
  specifically -- this is O1's own construction floor, distinct from
  EXPERIMENT.md 3.0's generic >=10 pre-registered eligibility floor (V2/A2
  use that one; O1 uses its own because its unit is a fixed-size set).
- Labels (2026-09-27, round 3): from the noise-only null -- inside its
  5th-95th percentile = consistent_with_noise (see LABEL_RULE). The raw
  head-vs-tail difference is descriptive only. The pre-registered
  "stable head, churning tail" band (head >= 0.8 with tail <= 0.5) is
  kept as a boolean in notes["prereg_band_met_by_engine"], not a label.

2026-09-27 review fix -- TIE-BREAK BUG. The original ranking broke ties
(the same rate value) alphabetically, every day. The tail in particular
has a large tied block (mean 13.7-16.1 companies per engine, mostly at
rate exactly 0 -- see notes["tie_diagnostics"]), so the same
alphabetically-first companies occupied the tail every single day
regardless of whether anything real happened -- inflating tail Jaccard by
construction, not by measurement. Fix: `_ranked_companies_random` breaks
ties randomly instead; the reported head_mean_jaccard/tail_mean_jaccard
are now the AVERAGE over N_TIE_BREAK_DRAWS=1000 independent random
tie-break draws (fixed seed, TIE_BREAK_SEED), not one deterministic
ranking. head_sets_by_day/tail_sets_by_day are KEPT (one illustrative
alphabetical realization, for manual inspection) but no longer back the
reported statistic.

2026-09-27 review fix -- NOISE-ONLY NULL. The permutation null (below)
answers "is this better than pure chance." It does not answer the
question that actually matters for a "you are unstable" product claim:
if each company's TRUE rate never changed all week, would sampling noise
alone (few reps per day) produce this much apparent set churn? For each
company, holds its rate fixed at the POOLED (aggregate) proportion across
the week, simulates each day's usable reps as an independent Binomial
draw at that fixed rate, and rebuilds head/tail Jaccard from the
simulated week (same random tie-break machinery, one draw per replicate).
Reported in notes["noise_only_null"], summarized the same way as the
permutation null: observed near this null's mean = the apparent churn is
consistent with sampling noise, not real behavioral change.

- 2026-09-26 review fix -- permutation null: raw Jaccard numbers (0.5 to
  0.7) have no meaning without a baseline. For each engine, for k in
  {HEAD_SIZE, n_tail} (sizes differ per engine and per head/tail -- head
  and tail are NOT directly comparable without each having its own null,
  since a larger k against the same population tends toward a higher
  expected Jaccard purely by combinatorics), simulate N_PERMUTATION_ITERATIONS
  draws of two independent random size-k subsets of the engine's
  n_qualifying companies and Jaccard each draw -- stationary across
  day-pairs by construction (company identity doesn't matter, only set
  size), so one simulation per (engine, k), not per day-pair. Fixed seed
  for reproducibility (run_phase_b.py's byte-identical-rerun guarantee).
- 2026-09-26 review fix -- tie-rate diagnostic: per engine per day, the
  rate value sitting at the head/tail cutoff boundary and how many
  companies are exactly tied there (many companies likely tie at rate=0,
  making tail membership partly an artifact of the alphabetical
  tie-break rather than a real ranking) -- this diagnostic is what led to
  finding the tie-break bug fixed above; it remains accurate regardless
  of tie-break method, since it describes the raw data's tie structure,
  not a chosen resolution of it.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

import numpy as np

from analysis.phase_b.daily_panel import (
    CONSECUTIVE_DAY_PAIRS,
    build_daily_panel,
    company_day_rates,
    qualifying_entities_by_engine,
)
from analysis.phase_b.figure_shape import FigureResult
from analysis.phase_b.reading import FrozenInputs

FIGURE_ID = "O1"
HEAD_SIZE = 13  # pre-registered, kept fixed regardless of population size -- see module docstring
O1_INTEGER_LEDGER_REF = "o1-integer-kept-k1-baseline-recomputed-2026-09-12"
HEAD_STABLE_AT_OR_ABOVE = 0.8
TAIL_CHURNING_AT_OR_BELOW = 0.5
PERMUTATION_SEED = 20260926
N_PERMUTATION_ITERATIONS = 10_000
TIE_BREAK_SEED = 20260927
N_TIE_BREAK_DRAWS = 1_000
NOISE_NULL_SEED = 20260927
N_NOISE_NULL_REPLICATES = 1_000
INTERPRETATION_THRESHOLDS_SOURCE = (
    "EXPERIMENT.md, '#### O1. Top quartile stability': \"Interpretation bands, labelled as interpretation: "
    "head at or above 0.8 with tail at or below 0.5 is the stable-head-churning-tail picture.\" -- literal "
    "pre-registered wording, not a display choice."
)
HEAD_TAIL_NULL_COMPARABILITY_NOTE = (
    "Head (k=13, fixed) and tail (k=n_qualifying//2, varies by engine) are compared against DIFFERENT chance "
    "baselines even under pure randomness -- a larger k against the same N population tends toward a higher "
    "expected Jaccard purely by combinatorics. Head's raw value is not directly comparable to tail's raw "
    "value without each side's own permutation null (see notes['permutation_null'])."
)
LABEL_RULE = (
    "Labels come from the NOISE-ONLY null (each company held at its pooled rate, binomial sampling noise "
    "only): observed inside the null's 5th-95th percentile = consistent_with_noise; above p95 = above_noise; "
    "below p5 = below_noise. The raw head-vs-tail difference is a descriptive column only (head and tail are "
    "different-sized sets), never a label."
)
TIE_BREAK_METHOD_NOTE = (
    "head_mean_jaccard/tail_mean_jaccard are averaged over 1,000 independent random tie-break draws (fixed "
    "seed) -- NOT computed from the single fixed-order realization in head_sets_by_day/tail_sets_by_day, "
    "which is kept only as one illustrative, human-inspectable listing (see the ledger for why a fixed "
    "tie order was replaced)."
)
NOISE_ONLY_NULL_NOTE = (
    "noise_only_null holds each company's rate fixed at its pooled (whole-week) proportion and simulates each "
    "day's usable reps as an independent Binomial draw at that fixed rate -- i.e. a week with NO real "
    "day-to-day change, only sampling noise from a small rep count. Observed near this null's mean means the "
    "apparent set churn is consistent with noise, not real behavioral change; observed clearly above it means "
    "real churn beyond noise. This is the null that matters for a stability/instability product claim -- the "
    "random-subset permutation null above only checks against pure chance, not against 'no real drift.'"
)


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 1.0  # two empty sets: trivially identical, never a division by zero
    return len(a & b) / len(union)


def _ranked_companies(day_rate_by_entity: dict[str, float]) -> list[str]:
    # Descending by rate; ties broken alphabetically by entity_id --
    # deterministic, used ONLY for head_sets_by_day/tail_sets_by_day (an
    # illustrative reference listing) and tie_diagnostics. NOT used for
    # the reported Jaccard statistic -- see _ranked_companies_random.
    return [e for e, _r in sorted(day_rate_by_entity.items(), key=lambda kv: (-kv[1], kv[0]))]


def _ranked_companies_random(day_rate_by_entity: dict[str, float], rng: np.random.Generator) -> list[str]:
    # Descending by rate; ties (identical rate value) broken by an
    # independent random permutation each call -- companies with a
    # strictly higher rate always precede those with a lower one,
    # deterministically; only the order WITHIN a tied group is randomized.
    groups: dict[float, list[str]] = defaultdict(list)
    for e, r in day_rate_by_entity.items():
        groups[r].append(e)
    ranked: list[str] = []
    for r in sorted(groups.keys(), reverse=True):
        members = groups[r]
        if len(members) > 1:
            members = [str(m) for m in rng.permutation(members)]
        ranked.extend(members)
    return ranked


def _head_tail_jaccard_one_realization(
    rate_by_day: dict[int, dict[str, float]], head_size: int, n_tail: int, rng: np.random.Generator
) -> tuple[dict[str, float], dict[str, float]]:
    """One random-tie-break draw: builds head/tail sets per day, returns
    the 6 consecutive-day-pair Jaccards for each. Shared by the tie-break
    average (real rates, many draws) and the noise-only null (simulated
    rates, one draw per replicate)."""
    head_by_day: dict[int, set[str]] = {}
    tail_by_day: dict[int, set[str]] = {}
    for day_index, day_rates in rate_by_day.items():
        ranked = _ranked_companies_random(day_rates, rng)
        head_by_day[day_index] = set(ranked[:head_size])
        tail_by_day[day_index] = set(ranked[-n_tail:]) if n_tail > 0 else set()

    head_jaccards: dict[str, float] = {}
    tail_jaccards: dict[str, float] = {}
    for d1, d2 in CONSECUTIVE_DAY_PAIRS:
        pair_key = f"{d1}-{d2}"
        if d1 in head_by_day and d2 in head_by_day:
            head_jaccards[pair_key] = _jaccard(head_by_day[d1], head_by_day[d2])
        if d1 in tail_by_day and d2 in tail_by_day:
            tail_jaccards[pair_key] = _jaccard(tail_by_day[d1], tail_by_day[d2])
    return head_jaccards, tail_jaccards


def _tie_break_averaged_jaccard(
    rate_by_day: dict[int, dict[str, float]], head_size: int, n_tail: int, rng: np.random.Generator, n_draws: int
) -> tuple[dict[str, float], dict[str, float]]:
    head_sums: dict[str, float] = defaultdict(float)
    tail_sums: dict[str, float] = defaultdict(float)
    head_counts: dict[str, int] = defaultdict(int)
    tail_counts: dict[str, int] = defaultdict(int)
    for _ in range(n_draws):
        h, t = _head_tail_jaccard_one_realization(rate_by_day, head_size, n_tail, rng)
        for k, v in h.items():
            head_sums[k] += v
            head_counts[k] += 1
        for k, v in t.items():
            tail_sums[k] += v
            tail_counts[k] += 1
    head_avg = {k: head_sums[k] / head_counts[k] for k in head_sums}
    tail_avg = {k: tail_sums[k] / tail_counts[k] for k in tail_sums}
    return head_avg, tail_avg


def _simulate_noise_only_realization(
    real_counts_by_entity_day: dict[str, dict[int, tuple[int, int]]],
    pooled_rate_by_entity: dict[str, float],
    rng: np.random.Generator,
) -> dict[int, dict[str, float]]:
    rate_by_day: dict[int, dict[str, float]] = defaultdict(dict)
    for entity_id, day_counts in real_counts_by_entity_day.items():
        p = pooled_rate_by_entity[entity_id]
        for day_index, (_n_mentioned, n_total) in day_counts.items():
            if n_total <= 0:
                continue
            k = rng.binomial(n_total, p)
            rate_by_day[day_index][entity_id] = k / n_total
    return rate_by_day


def _null_jaccard_samples(n_population: int, k: int, n_iterations: int, rng: np.random.Generator) -> list[float]:
    if k == 0 or k > n_population:
        return []
    samples = []
    for _ in range(n_iterations):
        a = set(rng.choice(n_population, size=k, replace=False).tolist())
        b = set(rng.choice(n_population, size=k, replace=False).tolist())
        union = a | b
        samples.append(len(a & b) / len(union) if union else 1.0)
    return samples


def _summarize_null(null_samples: list[float], observed: float | None) -> dict[str, Any] | None:
    if not null_samples or observed is None:
        return None
    ranked = sorted(null_samples)
    percentile = sum(1 for v in ranked if v <= observed) / len(ranked)
    return {
        "n_iterations": len(ranked),
        "null_mean_jaccard": statistics.fmean(ranked),
        "null_p05_jaccard": ranked[int(0.05 * (len(ranked) - 1))],
        "null_p95_jaccard": ranked[int(0.95 * (len(ranked) - 1))],
        "observed_mean_jaccard": observed,
        "observed_percentile_in_null": percentile,
    }


CONSISTENT_WITH_NOISE = "consistent_with_noise"
ABOVE_NOISE = "above_noise"
BELOW_NOISE = "below_noise"


def _noise_label(observed: float | None, null_summary: dict[str, Any] | None) -> str | None:
    """Label from the NOISE-ONLY null (not the random-subset permutation
    null): observed inside the null's 5th-95th percentile band is
    consistent_with_noise; above p95 = above_noise (more stable than noise
    alone produces); below p5 = below_noise."""
    if observed is None or not null_summary:
        return None
    if observed > null_summary["null_p95_jaccard"]:
        return ABOVE_NOISE
    if observed < null_summary["null_p05_jaccard"]:
        return BELOW_NOISE
    return CONSISTENT_WITH_NOISE


def _engine_label(head_label: str | None, tail_label: str | None) -> str | None:
    if head_label is None or tail_label is None:
        return None
    if head_label == CONSISTENT_WITH_NOISE and tail_label == CONSISTENT_WITH_NOISE:
        return CONSISTENT_WITH_NOISE
    return f"head_{head_label}__tail_{tail_label}"


def _tie_block(day_rates: dict[str, float], boundary_rate: float | None, included: set[str]) -> dict[str, Any]:
    if boundary_rate is None:
        return {"boundary_rate": None, "tie_block_size": 0, "tie_excluded_count": 0}
    tie_block = [c for c, r in day_rates.items() if r == boundary_rate]
    n_included = sum(1 for c in tie_block if c in included)
    return {"boundary_rate": boundary_rate, "tie_block_size": len(tie_block), "tie_excluded_count": len(tie_block) - n_included}


def compute(inputs: FrozenInputs) -> FigureResult:
    cells, _report = build_daily_panel(inputs)
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
            reason_untestable="zero daily-panel cells (no usable, qualifying, unbranded-intent data)",
        )

    rates = company_day_rates(cells)
    qualifying = qualifying_entities_by_engine(cells)
    engines = sorted(qualifying.keys())

    # engine -> day_index -> entity_id -> rate
    rate_by_engine_day: dict[str, dict[int, dict[str, float]]] = {engine: {} for engine in engines}
    # engine -> entity_id -> day_index -> (n_mentioned, n_total) -- the raw
    # counts, needed for the noise-only null's pooled rate and real n_d.
    raw_by_engine_entity: dict[str, dict[str, dict[int, tuple[int, int]]]] = {engine: defaultdict(dict) for engine in engines}
    for (entity_id, engine, day_index), (n_mentioned, n_total) in rates.items():
        rate_by_engine_day.setdefault(engine, {}).setdefault(day_index, {})[entity_id] = n_mentioned / n_total
        raw_by_engine_entity[engine][entity_id][day_index] = (n_mentioned, n_total)

    value: dict[str, dict[str, float] | None] = {}
    untestable_engines: dict[str, str] = {}
    n_qualifying_by_engine: dict[str, int] = {}
    n_tail_by_engine: dict[str, int] = {}
    head_sets_by_engine: dict[str, dict[int, list[str]]] = {}
    tail_sets_by_engine: dict[str, dict[int, list[str]]] = {}
    head_jaccard_by_pair_by_engine: dict[str, dict[str, float]] = {}
    tail_jaccard_by_pair_by_engine: dict[str, dict[str, float]] = {}
    interpretation_band: dict[str, str | None] = {}
    noise_labels_by_engine: dict[str, dict[str, str | None]] = {}
    raw_head_vs_tail: dict[str, dict[str, Any] | None] = {}
    prereg_band_met_by_engine: dict[str, bool] = {}
    permutation_null_by_engine: dict[str, dict[str, Any] | None] = {}
    noise_only_null_by_engine: dict[str, dict[str, Any] | None] = {}
    tie_diagnostics_by_engine: dict[str, dict[int, dict[str, Any]]] = {}
    tie_diagnostics_summary_by_engine: dict[str, dict[str, Any] | None] = {}

    # Three independent RNG streams, one per purpose, each consumed in
    # sorted-engine order -- fixed, documented order for reproducibility
    # (run_phase_b.py's byte-identical-rerun guarantee).
    permutation_rng = np.random.default_rng(PERMUTATION_SEED)
    tie_break_rng = np.random.default_rng(TIE_BREAK_SEED)
    noise_rng = np.random.default_rng(NOISE_NULL_SEED)

    for engine in engines:
        n_qual = len(qualifying.get(engine, set()))
        n_qualifying_by_engine[engine] = n_qual
        if n_qual < HEAD_SIZE:
            value[engine] = None
            untestable_engines[engine] = f"only {n_qual} qualifying companies (need >= {HEAD_SIZE} to form the head set)"
            interpretation_band[engine] = None
            permutation_null_by_engine[engine] = None
            noise_only_null_by_engine[engine] = None
            tie_diagnostics_summary_by_engine[engine] = None
            continue

        n_tail = n_qual // 2
        n_tail_by_engine[engine] = n_tail

        # Deterministic (alphabetical) pass: illustrative reference sets +
        # tie diagnostics only -- does NOT feed the reported statistic.
        head_by_day: dict[int, set[str]] = {}
        tail_by_day: dict[int, set[str]] = {}
        tie_diagnostics_by_day: dict[int, dict[str, Any]] = {}
        for day_index, day_rates in rate_by_engine_day[engine].items():
            ranked = _ranked_companies(day_rates)
            head_set = set(ranked[:HEAD_SIZE])
            tail_set = set(ranked[-n_tail:]) if n_tail > 0 else set()
            head_by_day[day_index] = head_set
            tail_by_day[day_index] = tail_set

            head_boundary_rate = day_rates[ranked[HEAD_SIZE - 1]] if len(ranked) >= HEAD_SIZE else None
            tail_boundary_rate = day_rates[ranked[-n_tail]] if n_tail > 0 and len(ranked) >= n_tail else None
            tie_diagnostics_by_day[day_index] = {
                "head": _tie_block(day_rates, head_boundary_rate, head_set),
                "tail": _tie_block(day_rates, tail_boundary_rate, tail_set),
            }

        head_sets_by_engine[engine] = {d: sorted(s) for d, s in head_by_day.items()}
        tail_sets_by_engine[engine] = {d: sorted(s) for d, s in tail_by_day.items()}
        tie_diagnostics_by_engine[engine] = tie_diagnostics_by_day
        tie_diagnostics_summary_by_engine[engine] = {
            "mean_head_tie_block_size": statistics.fmean(d["head"]["tie_block_size"] for d in tie_diagnostics_by_day.values()),
            "mean_tail_tie_block_size": statistics.fmean(d["tail"]["tie_block_size"] for d in tie_diagnostics_by_day.values()),
            "days_with_head_tie_at_boundary": sum(1 for d in tie_diagnostics_by_day.values() if d["head"]["tie_block_size"] > 1),
            "days_with_tail_tie_at_boundary": sum(1 for d in tie_diagnostics_by_day.values() if d["tail"]["tie_block_size"] > 1),
        }

        # The reported statistic: averaged over N_TIE_BREAK_DRAWS random
        # tie-break realizations of the REAL rates.
        head_jaccards, tail_jaccards = _tie_break_averaged_jaccard(
            rate_by_engine_day[engine], HEAD_SIZE, n_tail, tie_break_rng, N_TIE_BREAK_DRAWS
        )
        head_jaccard_by_pair_by_engine[engine] = head_jaccards
        tail_jaccard_by_pair_by_engine[engine] = tail_jaccards

        head_mean = statistics.fmean(head_jaccards.values()) if head_jaccards else None
        tail_mean = statistics.fmean(tail_jaccards.values()) if tail_jaccards else None
        value[engine] = {"head_mean_jaccard": head_mean, "tail_mean_jaccard": tail_mean}

        # Random-subset permutation null: one simulation per (engine, k) --
        # stationary across day-pairs by construction.
        null_head_samples = _null_jaccard_samples(n_qual, HEAD_SIZE, N_PERMUTATION_ITERATIONS, permutation_rng)
        null_tail_samples = _null_jaccard_samples(n_qual, n_tail, N_PERMUTATION_ITERATIONS, permutation_rng)
        permutation_null_by_engine[engine] = {
            "head": _summarize_null(null_head_samples, head_mean),
            "tail": _summarize_null(null_tail_samples, tail_mean),
        }

        # Noise-only null: each company held at its pooled (whole-week)
        # rate, real per-day rep counts, Binomial sampling noise only.
        pooled_rate_by_entity: dict[str, float] = {}
        for entity_id, day_counts in raw_by_engine_entity[engine].items():
            total_mentioned = sum(c[0] for c in day_counts.values())
            total_reps = sum(c[1] for c in day_counts.values())
            pooled_rate_by_entity[entity_id] = total_mentioned / total_reps if total_reps else 0.0

        noise_head_means: list[float] = []
        noise_tail_means: list[float] = []
        for _ in range(N_NOISE_NULL_REPLICATES):
            simulated_rate_by_day = _simulate_noise_only_realization(
                raw_by_engine_entity[engine], pooled_rate_by_entity, noise_rng
            )
            h, t = _head_tail_jaccard_one_realization(simulated_rate_by_day, HEAD_SIZE, n_tail, noise_rng)
            if h:
                noise_head_means.append(statistics.fmean(h.values()))
            if t:
                noise_tail_means.append(statistics.fmean(t.values()))
        noise_only_null_by_engine[engine] = {
            "head": _summarize_null(noise_head_means, head_mean),
            "tail": _summarize_null(noise_tail_means, tail_mean),
        }

        head_label = _noise_label(head_mean, noise_only_null_by_engine[engine]["head"])
        tail_label = _noise_label(tail_mean, noise_only_null_by_engine[engine]["tail"])
        noise_labels_by_engine[engine] = {"head": head_label, "tail": tail_label}
        interpretation_band[engine] = _engine_label(head_label, tail_label)
        # Descriptive only -- never a label: raw head vs. tail difference
        # (different set sizes, so not directly comparable; see
        # head_tail_null_comparability_note).
        if head_mean is None or tail_mean is None:
            raw_head_vs_tail[engine] = None
        else:
            raw_head_vs_tail[engine] = {
                "head_minus_tail": head_mean - tail_mean,
                "direction": "head_higher" if head_mean > tail_mean else ("tail_higher" if tail_mean > head_mean else "equal"),
            }
        prereg_band_met_by_engine[engine] = (
            head_mean is not None and tail_mean is not None
            and head_mean >= HEAD_STABLE_AT_OR_ABOVE and tail_mean <= TAIL_CHURNING_AT_OR_BELOW
        )

    return FigureResult(
        figure_id=FIGURE_ID,
        status="ok",
        value=value,
        denominator_before_exclusion=n_qualifying_by_engine,
        denominator_after_exclusion=n_qualifying_by_engine,
        n_included=n_qualifying_by_engine,
        n_excluded={"untestable_engines": untestable_engines},
        threshold=None,  # interpretation band, not a pre-registered pass/fail threshold -- see notes
        reason_untestable=None,
        notes={
            "untestable_engines": untestable_engines,
            "n_qualifying_by_engine": n_qualifying_by_engine,
            "n_tail_by_engine": n_tail_by_engine,
            "head_size": HEAD_SIZE,
            "head_sets_by_day": head_sets_by_engine,
            "tail_sets_by_day": tail_sets_by_engine,
            "head_jaccard_by_day_pair": head_jaccard_by_pair_by_engine,
            "tail_jaccard_by_day_pair": tail_jaccard_by_pair_by_engine,
            "interpretation_band": interpretation_band,
            "noise_labels_by_engine": noise_labels_by_engine,
            "label_rule": LABEL_RULE,
            "raw_head_vs_tail_descriptive": raw_head_vs_tail,
            "prereg_band_met_by_engine": prereg_band_met_by_engine,
            "interpretation_thresholds": {
                "head_stable_at_or_above": HEAD_STABLE_AT_OR_ABOVE,
                "tail_churning_at_or_below": TAIL_CHURNING_AT_OR_BELOW,
            },
            "o1_integer_note": (
                f"top-{HEAD_SIZE} kept per the pre-registered integer, not recomputed against the full "
                f"population -- see reports/ambiguity_ledger.md, {O1_INTEGER_LEDGER_REF}."
            ),
            "permutation_null": permutation_null_by_engine,
            "permutation_seed": PERMUTATION_SEED,
            "head_tail_null_comparability_note": HEAD_TAIL_NULL_COMPARABILITY_NOTE,
            "tie_diagnostics": tie_diagnostics_by_engine,
            "tie_diagnostics_summary": tie_diagnostics_summary_by_engine,
            "interpretation_thresholds_source": INTERPRETATION_THRESHOLDS_SOURCE,
            "tie_break_seed": TIE_BREAK_SEED,
            "n_tie_break_draws": N_TIE_BREAK_DRAWS,
            "tie_break_method_note": TIE_BREAK_METHOD_NOTE,
            "noise_only_null": noise_only_null_by_engine,
            "noise_null_seed": NOISE_NULL_SEED,
            "n_noise_null_replicates": N_NOISE_NULL_REPLICATES,
            "noise_only_null_note": NOISE_ONLY_NULL_NOTE,
        },
    )
