"""V2 -- Drift against noise (PRE-SPECIFIED TABLE, product decision, pinned
EXPERIMENT.md / Notion Part 3.2).

Unit: (entity_id, intent_id, engine, paraphrase_id). Reads
analysis.phase_b.daily_panel.build_daily_panel, which already restricts to
usable runs, the six unbranded intents, and qualifying companies -- none of
that is re-filtered here.

- Noise floor (within-day): at fixed unit and day, the usable reps found
  for that day (nominally 3, never assumed -- "a missed day stays missed"
  applies to reps too) give a rate out of n_d. PRIMARY within-day variance
  per day = rate*(1-rate) -- the raw variance of the n_d binary
  observations themselves, pooled (mean) across the days present for that
  unit. SENSITIVITY within-day variance per day = rate*(1-rate)/n_d -- the
  sampling variance of the MEAN of n_d Bernoulli draws (EXPERIMENT.md's own
  callout: "a daily rate over 3 reps carries within-variance/3, not the
  full within variance"), generalized to /n_d so the one real n=2 day
  (google_gemini/i6/p4/d7 -- see the ambiguity ledger) is handled correctly
  with no special case. Reported as a full parallel row in notes["sensitivity"]
  -- never swapped into value, per the task's explicit instruction.
- Between-day variance: population variance (statistics.pvariance) of the
  unit's daily rates, across the days actually present (never imputed). A
  unit with fewer than 2 days present has no between-day variance and is
  excluded (reason "fewer_than_2_days").
- excess_drift = sqrt(max(0, between_var - within_var)) * 100, in
  percentage points of mention rate -- rates are fractions, so the *100
  scaling happens only as the very last step, after the floor and sqrt.
- Per engine: value = MEDIAN excess drift across that engine's units.
  EXPERIMENT.md does not state median vs. mean for V2 itself (unlike V1,
  which explicitly says "median not mean" for its own stated reason) --
  median is used here for the same robustness reasoning, as an
  interpolation confirmed with the study owner and logged in the ambiguity
  ledger, not a literal quote. Verdict banded at >=5 ("alert_worthy") /
  <=2 ("no_product") / between ("real_but_too_small_to_alert"), per
  EXPERIMENT.md verbatim.
- MIN_QUALIFYING_FOR_V2 = 10 (EXPERIMENT.md 3.0's pre-registered
  eligibility-floor number): an engine below it is untestable for V2
  specifically, without failing the whole figure -- same per-engine-
  untestable pattern as figures/a3.py's MIN_QUALIFYING_COMPANIES.

2026-09-27 review fix -- NOISE-ONLY NULL for the >=5pt alert-worthy
share. Holds each unit's rate fixed at its POOLED (whole-window)
proportion (total mentions / total usable reps across the days present),
simulates each day's real usable-rep count as an independent Binomial
draw at that fixed rate (i.e. a week with NO real drift, only sampling
noise), and recomputes the same excess-drift formula (including the
floor at 0) on the simulated week -- vectorized per unit with numpy,
looped once per unit. Each unit's null-crossing probability (share of
1,000 simulated replicates landing >=5pts) is averaged across an engine's
units to get notes["share_alert_worthy_null_by_engine"], reported next to
the observed share: if observed isn't clearly above the null, the
alert-worthy tail is consistent with sampling noise, not real signal.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any

import numpy as np

from analysis.phase_b.daily_panel import build_daily_panel
from analysis.phase_b.figure_shape import FigureResult
from analysis.phase_b.reading import FrozenInputs

FIGURE_ID = "V2"
MIN_QUALIFYING_FOR_V2 = 10  # EXPERIMENT.md 3.0's pre-registered eligibility floor
ALERT_WORTHY_AT_OR_ABOVE = 5.0
NO_PRODUCT_AT_OR_BELOW = 2.0
SIX_CHANGE_POINTS_LIMITATION = (
    "7 days gives 6 change points. One model update inside the window is indistinguishable from "
    "ongoing drift -- V2 can show that movement exists, it cannot show that movement is regular."
)
# 2026-09-26 review fix: a company never mentioned (all_zero) or always
# mentioned (all_hundred) across the days present has zero variance both
# within and between days BY CONSTRUCTION -- it can never contribute
# nonzero excess drift. These units are legitimate data (a company that
# never appears cannot drift), but they mechanically inflate how much
# weight the floor-at-0 gets in the median. Counted and reported per
# engine, plus a second median computed with all_zero units excluded (see
# the ambiguity ledger, v2-all-zero-all-hundred-and-nonzero-sensitivity-cut-2026-09-26).
SENSITIVITY_EXPLANATION = (
    "The sensitivity row uses a different WITHIN-DAY VARIANCE FORMULA, not a different population of "
    "units: primary within-day variance is the raw per-rep Bernoulli variance rate*(1-rate); sensitivity "
    "divides that by the day's usable rep count, rate*(1-rate)/n_d -- the sampling variance of the day's "
    "MEAN rate rather than of one rep. Because within-day variance is always subtracted from between-day "
    "variance before the floor and square root, a smaller within-day number (sensitivity) can only push "
    "excess drift up or leave it unchanged, never down. On this data both formulas land on the identical "
    "median in every engine because the primary median is already floored at exactly zero for a "
    "majority-zero-variance population -- the sensitivity row is real, different, unit-by-unit "
    "arithmetic; it simply cannot move a median that is already pinned at the floor unless enough "
    "individual units cross it, which they do not here."
)
REAL_LABEL = "real"
CONSISTENT_WITH_NOISE_LABEL = "consistent_with_noise"
V2_NOISE_NULL_SEED = 20260927
N_NOISE_NULL_REPLICATES = 1_000
NOISE_ONLY_NULL_NOTE = (
    "share_alert_worthy_null_by_engine holds each unit's rate fixed at its pooled (whole-window) proportion "
    "and simulates each day's real usable-rep count as an independent Binomial draw at that fixed rate -- a "
    "week with NO real drift, only sampling noise -- then recomputes the same excess-drift formula (floor "
    "included). If the observed share isn't clearly above this null, the alert-worthy tail is consistent with "
    "sampling noise, not real signal."
)


def _verdict(points: float | None) -> str | None:
    if points is None:
        return None
    if points >= ALERT_WORTHY_AT_OR_ABOVE:
        return "alert_worthy"
    if points <= NO_PRODUCT_AT_OR_BELOW:
        return "no_product"
    return "real_but_too_small_to_alert"


def _unit_excess_drift(days: dict[int, list[bool]]) -> dict[str, Any] | None:
    """days: day_index -> list of mentioned bools actually found for that
    day (however many usable reps that turned out to be -- never assumed
    to be 3). None if fewer than 2 days are present (between-day variance
    is undefined with a single day)."""
    if len(days) < 2:
        return None

    daily_rates: dict[int, float] = {}
    primary_within_by_day: dict[int, float] = {}
    sensitivity_within_by_day: dict[int, float] = {}
    reps_by_day: dict[int, int] = {}
    mentioned_by_day: dict[int, int] = {}
    for day_index, reps in days.items():
        n_d = len(reps)
        k_d = sum(1 for m in reps if m)
        rate_d = k_d / n_d
        daily_rates[day_index] = rate_d
        primary_within_by_day[day_index] = rate_d * (1 - rate_d)
        sensitivity_within_by_day[day_index] = (rate_d * (1 - rate_d)) / n_d
        reps_by_day[day_index] = n_d
        mentioned_by_day[day_index] = k_d

    between_var = statistics.pvariance(list(daily_rates.values()))
    primary_within = statistics.fmean(primary_within_by_day.values())
    sensitivity_within = statistics.fmean(sensitivity_within_by_day.values())

    raw_variance_diff = between_var - primary_within  # UNCLIPPED -- can be negative
    primary_excess = math.sqrt(max(0.0, raw_variance_diff)) * 100
    sensitivity_excess = math.sqrt(max(0.0, between_var - sensitivity_within)) * 100
    # Sign-preserving generalization of the primary transform (sqrt(max(0,x))*100
    # for x>=0) -- shows how far below zero the unfloored estimate would have
    # gone, in the same percentage-point scale as excess_drift_points.
    signed_excess_drift_points = math.copysign(math.sqrt(abs(raw_variance_diff)), raw_variance_diff) * 100

    rates = list(daily_rates.values())
    return {
        "n_days_used": len(days),
        "reps_used_by_day": reps_by_day,
        "mentioned_by_day": mentioned_by_day,
        "daily_rates": daily_rates,
        "between_day_variance": between_var,
        "primary_within_day_variance": primary_within,
        "sensitivity_within_day_variance": sensitivity_within,
        "excess_drift_points": primary_excess,
        "sensitivity_excess_drift_points": sensitivity_excess,
        "all_zero": all(r == 0.0 for r in rates),
        "all_hundred": all(r == 1.0 for r in rates),
        "raw_variance_diff": raw_variance_diff,
        "signed_excess_drift_points": signed_excess_drift_points,
    }


def _noise_only_crossing_vector(
    reps_by_day: dict[int, int], mentioned_by_day: dict[int, int], rng: np.random.Generator, n_replicates: int
) -> np.ndarray | None:
    """Vectorized per unit: holds the unit's rate fixed at its pooled
    (whole-window) proportion, simulates each day's real usable-rep count
    as an independent Binomial draw at that rate, and returns the share
    of replicates whose (floored) excess drift is >= ALERT_WORTHY_AT_OR_ABOVE.
    Returns the boolean per-replicate vector (so an engine-level null
    DISTRIBUTION of the alert-worthy share can be built by summing
    across units replicate by replicate). None if fewer than 2 days (mirrors _unit_excess_drift's own guard)."""
    days = sorted(reps_by_day.keys())
    if len(days) < 2:
        return None
    n_arr = np.array([reps_by_day[d] for d in days], dtype=np.int64)
    total_n = int(n_arr.sum())
    if total_n == 0:
        return None
    pooled_rate = sum(mentioned_by_day.values()) / total_n

    draws = rng.binomial(n_arr, pooled_rate, size=(n_replicates, len(days)))
    simulated_rates = draws / n_arr
    between_var = simulated_rates.var(axis=1)  # population variance, ddof=0
    within_var = (simulated_rates * (1 - simulated_rates)).mean(axis=1)
    excess = np.sqrt(np.clip(between_var - within_var, 0.0, None)) * 100
    return excess >= ALERT_WORTHY_AT_OR_ABOVE


def _noise_comparison(observed_share: float, null_counts: Any, n_units_null: int) -> dict[str, Any] | None:
    """Observed >=5pt share against the engine-level noise-only null
    DISTRIBUTION (one share per replicate = units crossing / units).
    Label is "real" ONLY if observed > the null's 95th percentile;
    anything else is "consistent_with_noise" (never "no drift")."""
    if null_counts is None or n_units_null == 0:
        return None
    null_shares = np.sort(null_counts / n_units_null)
    null_mean = float(null_shares.mean())
    p95 = float(null_shares[int(0.95 * (len(null_shares) - 1))])
    percentile = float(np.mean(null_shares <= observed_share))
    return {
        "null_share_alert_worthy": null_mean,
        "observed_share_alert_worthy": observed_share,
        "ratio_observed_to_null": (observed_share / null_mean) if null_mean > 0 else None,
        "null_p95_share": p95,
        "observed_percentile_in_null": percentile,
        "n_replicates": int(len(null_shares)),
        "label": REAL_LABEL if observed_share > p95 else CONSISTENT_WITH_NOISE_LABEL,
    }


def compute(inputs: FrozenInputs) -> FigureResult:
    cells, report = build_daily_panel(inputs)
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

    # (entity_id, intent_id, engine, paraphrase_id) -> day_index -> [mentioned,...]
    grouped: dict[tuple[str, int, str, int], dict[int, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for cell in cells:
        key = (cell.entity_id, cell.intent_id, cell.engine, cell.paraphrase_id)
        grouped[key][cell.day_index].append(cell.mentioned)

    engines = sorted(report.qualifying_by_engine.keys())
    per_unit: list[dict[str, Any]] = []
    n_excluded_by_reason: dict[str, int] = defaultdict(int)
    units_by_engine: dict[str, list[dict[str, Any]]] = defaultdict(list)

    noise_null_rng = np.random.default_rng(V2_NOISE_NULL_SEED)
    # engine -> per-replicate count of units crossing >=5pts under the null
    null_crossing_counts: dict[str, Any] = {}
    null_unit_counts: dict[str, int] = {}
    for (entity_id, intent_id, engine, paraphrase_id), days in grouped.items():
        result = _unit_excess_drift(days)
        if result is None:
            n_excluded_by_reason["fewer_than_2_days"] += 1
            continue
        crossing_vector = _noise_only_crossing_vector(
            result["reps_used_by_day"], result["mentioned_by_day"], noise_null_rng, N_NOISE_NULL_REPLICATES
        )
        crossing_probability = float(np.mean(crossing_vector)) if crossing_vector is not None else None
        if crossing_vector is not None:
            null_crossing_counts[engine] = null_crossing_counts.get(engine, 0) + crossing_vector.astype(np.int64)
            null_unit_counts[engine] = null_unit_counts.get(engine, 0) + 1
        record = {
            "entity_id": entity_id,
            "intent_id": intent_id,
            "engine": engine,
            "paraphrase_id": paraphrase_id,
            "noise_only_crossing_probability": crossing_probability,
            **result,
        }
        per_unit.append(record)
        units_by_engine[engine].append(record)

    value: dict[str, float | None] = {}
    verdict_by_engine: dict[str, str | None] = {}
    n_included: dict[str, int] = {}
    untestable_engines: dict[str, str] = {}
    sensitivity_value: dict[str, float | None] = {}
    n_all_zero_by_engine: dict[str, int | None] = {}
    n_all_hundred_by_engine: dict[str, int | None] = {}
    share_all_zero_by_engine: dict[str, float | None] = {}
    median_nonzero_by_engine: dict[str, float | None] = {}
    n_nonzero_units_by_engine: dict[str, int | None] = {}
    verdict_nonzero_only_by_engine: dict[str, str | None] = {}
    n_alert_worthy_by_engine: dict[str, int | None] = {}
    share_alert_worthy_by_engine: dict[str, float | None] = {}
    n_negative_before_floor_by_engine: dict[str, int | None] = {}
    signed_excess_drift_summary_by_engine: dict[str, dict[str, float] | None] = {}
    share_alert_worthy_null_by_engine: dict[str, float | None] = {}
    noise_comparison_by_engine: dict[str, dict[str, Any] | None] = {}

    for engine in engines:
        n_qual = report.qualifying_by_engine.get(engine, 0)
        units = units_by_engine.get(engine, [])
        n_included[engine] = len(units)
        if n_qual < MIN_QUALIFYING_FOR_V2 or not units:
            value[engine] = None
            verdict_by_engine[engine] = None
            sensitivity_value[engine] = None
            n_all_zero_by_engine[engine] = None
            n_all_hundred_by_engine[engine] = None
            share_all_zero_by_engine[engine] = None
            median_nonzero_by_engine[engine] = None
            n_nonzero_units_by_engine[engine] = None
            verdict_nonzero_only_by_engine[engine] = None
            n_alert_worthy_by_engine[engine] = None
            share_alert_worthy_by_engine[engine] = None
            n_negative_before_floor_by_engine[engine] = None
            signed_excess_drift_summary_by_engine[engine] = None
            share_alert_worthy_null_by_engine[engine] = None
            noise_comparison_by_engine[engine] = None
            untestable_engines[engine] = (
                f"only {n_qual} qualifying companies (need >= {MIN_QUALIFYING_FOR_V2})"
                if n_qual < MIN_QUALIFYING_FOR_V2
                else "no unit had >= 2 usable days"
            )
            continue

        median_excess = statistics.median(u["excess_drift_points"] for u in units)
        median_sensitivity = statistics.median(u["sensitivity_excess_drift_points"] for u in units)
        value[engine] = median_excess
        sensitivity_value[engine] = median_sensitivity
        verdict_by_engine[engine] = _verdict(median_excess)

        n_all_zero = sum(1 for u in units if u["all_zero"])
        n_all_hundred = sum(1 for u in units if u["all_hundred"])
        n_all_zero_by_engine[engine] = n_all_zero
        n_all_hundred_by_engine[engine] = n_all_hundred
        share_all_zero_by_engine[engine] = n_all_zero / len(units)

        nonzero_units = [u for u in units if not u["all_zero"]]
        n_nonzero_units_by_engine[engine] = len(nonzero_units)
        median_nonzero = statistics.median(u["excess_drift_points"] for u in nonzero_units) if nonzero_units else None
        median_nonzero_by_engine[engine] = median_nonzero
        verdict_nonzero_only_by_engine[engine] = _verdict(median_nonzero)

        n_alert_worthy = sum(1 for u in units if u["excess_drift_points"] >= ALERT_WORTHY_AT_OR_ABOVE)
        n_alert_worthy_by_engine[engine] = n_alert_worthy
        share_alert_worthy_by_engine[engine] = n_alert_worthy / len(units)

        crossing_probs = [u["noise_only_crossing_probability"] for u in units if u["noise_only_crossing_probability"] is not None]
        share_alert_worthy_null_by_engine[engine] = statistics.fmean(crossing_probs) if crossing_probs else None

        noise_comparison_by_engine[engine] = _noise_comparison(
            share_alert_worthy_by_engine[engine],
            null_crossing_counts.get(engine),
            null_unit_counts.get(engine, 0),
        )

        n_negative_before_floor = sum(1 for u in units if u["raw_variance_diff"] < 0)
        n_negative_before_floor_by_engine[engine] = n_negative_before_floor
        signed_values = [u["signed_excess_drift_points"] for u in units]
        signed_excess_drift_summary_by_engine[engine] = {
            "median": statistics.median(signed_values),
            "mean": statistics.fmean(signed_values),
            "min": min(signed_values),
            "max": max(signed_values),
            "n_negative": n_negative_before_floor,
            "share_negative": n_negative_before_floor / len(units),
        }

    return FigureResult(
        figure_id=FIGURE_ID,
        status="ok",
        value=value,
        denominator_before_exclusion=len(grouped),
        denominator_after_exclusion=len(per_unit),
        n_included=n_included,
        n_excluded=dict(n_excluded_by_reason),
        threshold={
            "alert_worthy_at_or_above": ALERT_WORTHY_AT_OR_ABOVE,
            "no_product_at_or_below": NO_PRODUCT_AT_OR_BELOW,
        },
        reason_untestable=None,
        notes={
            "verdict_by_engine": verdict_by_engine,
            "sensitivity": sensitivity_value,
            "sensitivity_explanation": SENSITIVITY_EXPLANATION,
            "untestable_engines": untestable_engines,
            "per_unit": per_unit,
            "limitation": SIX_CHANGE_POINTS_LIMITATION,
            "min_qualifying_for_testable": MIN_QUALIFYING_FOR_V2,
            "n_all_zero_by_engine": n_all_zero_by_engine,
            "n_all_hundred_by_engine": n_all_hundred_by_engine,
            "share_all_zero_by_engine": share_all_zero_by_engine,
            "median_excess_drift_nonzero_only": median_nonzero_by_engine,
            "n_nonzero_units_by_engine": n_nonzero_units_by_engine,
            "verdict_nonzero_only_by_engine": verdict_nonzero_only_by_engine,
            "n_alert_worthy_by_engine": n_alert_worthy_by_engine,
            "share_alert_worthy_by_engine": share_alert_worthy_by_engine,
            "n_negative_before_floor_by_engine": n_negative_before_floor_by_engine,
            "n_negative_before_floor_total": sum(v for v in n_negative_before_floor_by_engine.values() if v is not None),
            "signed_excess_drift_summary_by_engine": signed_excess_drift_summary_by_engine,
            "share_alert_worthy_null_by_engine": share_alert_worthy_null_by_engine,
            "noise_comparison_by_engine": noise_comparison_by_engine,
            "noise_only_null_seed": V2_NOISE_NULL_SEED,
            "n_noise_null_replicates": N_NOISE_NULL_REPLICATES,
            "noise_only_null_note": NOISE_ONLY_NULL_NOTE,
        },
    )
