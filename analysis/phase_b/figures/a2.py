"""A2 -- Cross-engine drift correlation (PRE-SPECIFIED TABLE, pinned
EXPERIMENT.md / Notion Part 3.2).

Unit: day-over-day change in a company's mention rate, company level,
pooled across the six unbranded intents (EXPERIMENT.md, verbatim). The
company-day rate itself pools every unbranded intent, paraphrase, and
usable rep into one ratio per (company, engine, day) --
analysis.phase_b.daily_panel.company_day_rates, the same pooled rate O1
reads, so both figures agree on what "a company's rate that day" means.

- Delta series: for each (company, engine), the 6 CONSECUTIVE day-pair
  deltas (1,2)...(6,7) -- never all 21 possible pairs. A day missing for
  that company/engine contributes no delta for either pair touching it
  ("a missed day stays missed"); it is never imputed or interpolated.
- Pool ("qualifying companies x 6 change points, per engine pair",
  verbatim): for one engine pair, the paired sample is every (company,
  day-pair) combination where BOTH engines have a computable delta for
  that company on that pair -- i.e. flattened across companies AND change
  points, not one point per company. Six engine pairs
  (itertools.combinations over the 4 engines actually present).
- Statistic: Pearson r (scipy.stats.pearsonr) per pair, and the mean over
  the pairs that were computable.
- MIN_QUALIFYING_FOR_PAIR = 10 (EXPERIMENT.md 3.0's pre-registered
  eligibility-floor number), applied to the INTERSECTION of the two
  engines' qualifying-company sets -- that intersection is the real
  population a pair's r is computed over, not either engine's own count.
  A pair below it is untestable individually, without failing the figure
  (same per-unit-untestable discipline as figures/a3.py/v2.py).
- No middle inconclusive band -- EXPERIMENT.md states only the >=0.3
  "shared movement" reading vs. "near zero, independent" for A2 (unlike
  V1/V2's three-way bands); verdict is a plain >=0.3 check. 2026-09-26
  review fix: the low-r label was "independent", which overclaims -- an
  absence of correlation is not proof of independence, especially given
  V2 shows most units never move at all. Relabeled
  "no_evidence_of_shared_movement".
- 95% CI per pair (Fisher z-transform), reported alongside r -- with an
  explicit caveat that the "n" here (company x day-pair, flattened) is
  NOT an iid sample: the same company contributes up to 6 correlated
  points, and each day-pair is shared across many companies. The CI is a
  standard approximation, likely narrower than the truth (a lower bound
  on the real width), not a clustering-robust estimate.
- Within-engine lag-1 autocorrelation of the delta series (adjacent
  day-pair deltas within one engine): a diagnostic on that engine's OWN
  noise structure. 2026-09-27 review fix -- the prior wording implied
  this "contextualizes... A2's negative cross-engine r values," which
  overclaims: within-engine autocorrelation is a fact about one engine's
  own series and does not, by itself, imply anything about the SIGN of a
  CROSS-engine correlation (that would require a shared mechanism across
  engines, which this diagnostic does not establish). It supports the
  necessary condition "each engine's delta series looks like noise," not
  a causal explanation for why two DIFFERENT engines' r would be
  negative -- the cluster-bootstrap CI below is what actually answers
  that question. A theoretical reference point: pure white noise's first
  difference has a lag-1 autocorrelation of exactly -0.5 (
  `Cov(Δt,Δt+1)=-Var(X)`, `Var(Δt)=2·Var(X)` for iid X), which the
  observed within-engine values (-0.32 to -0.52) straddle almost
  exactly -- consistent with no real per-company drift (V2's own
  finding), not with a specific cross-engine claim.
- 2026-09-27 review fix -- CLUSTER BOOTSTRAP replaces the Fisher-z CI as
  the primary interval. The Fisher-z CI's own caveat already flagged it
  as likely too narrow, since it treats company x day-pair observations
  as iid when the same company contributes up to 6 correlated points.
  `_cluster_bootstrap_ci` resamples COMPANIES (not individual points)
  with replacement, keeping each resampled company's full set of
  day-pair points -- the standard fix for clustered/panel data. The old
  Fisher-z numbers are kept under `fisher_z_ci_by_pair_reference` purely
  so a reader can see how much wider the correct interval is. Any pair
  whose cluster CI still excludes zero is labelled
  `notes["unexplained_pairs"]` -- explicitly not attributed to the
  within-engine autocorrelation artifact, since that mechanism doesn't
  apply cross-engine.
- Engine-level (n=6) parallel statistic, notes-only (never value -- this
  protects the pre-registered confirmatory contract): the mean delta
  across ALL qualifying companies in an engine, one point per day-pair
  (6 points total), correlated between the two engines of a pair. Weaker
  than the primary statistic (n=6), but a more direct test of "did the
  two engines' overall population move together on the same days."
"""
from __future__ import annotations

import itertools
import math
import statistics
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import pearsonr

from analysis.phase_b.daily_panel import (
    CONSECUTIVE_DAY_PAIRS,
    build_daily_panel,
    company_day_rates,
    qualifying_entities_by_engine,
)
from analysis.phase_b.figure_shape import FigureResult
from analysis.phase_b.reading import FrozenInputs

FIGURE_ID = "A2"
MIN_QUALIFYING_FOR_PAIR = 10  # EXPERIMENT.md 3.0's pre-registered eligibility floor
SHARED_MOVEMENT_AT_OR_ABOVE = 0.3
NO_EVIDENCE_OF_SHARED_MOVEMENT = "no_evidence_of_shared_movement"
CI_Z_95 = 1.959963984540054  # scipy.stats.norm.ppf(0.975)
LAG1_PAIRS = tuple(zip(CONSECUTIVE_DAY_PAIRS, CONSECUTIVE_DAY_PAIRS[1:]))  # (Δ(1,2),Δ(2,3)) ... (Δ(5,6),Δ(6,7))
CLUSTER_BOOTSTRAP_SEED = 20260927
N_CLUSTER_BOOTSTRAP = 2_000
WHITE_NOISE_LAG1_REFERENCE = -0.5
CI_DECIMALS = 3
MULTIPLE_PAIRS_NOTE = (
    "6 pairs at 95%: about 0.3 false exclusions of zero are expected by chance alone (6 x 0.05 = 0.3), so a "
    "pair whose CI excludes zero is not by itself evidence of real shared movement."
)
V2_A2_READ_TOGETHER_SENTENCE = (
    "V2 asks whether anything moves. A2 asks whether the movement is shared. V2 can pass while A2 fails, "
    "and that combination means per-company monitoring works but a category-wide alert does not."
)
CI_CAVEAT = (
    "The primary ci_by_pair is a COMPANY-CLUSTER bootstrap (resamples companies with replacement, keeping "
    "each one's full set of day-pair points) -- the standard fix for the non-independence Fisher z ignores. "
    "fisher_z_ci_by_pair_reference keeps the old, narrower Fisher-z numbers for comparison only; never used "
    "for a claim."
)
AUTOCORRELATION_CAVEAT = (
    "A negative lag-1 autocorrelation here is a fact about ONE engine's OWN delta series -- it supports "
    "'this engine's series looks like noise' (consistent with V2's finding that most units never really "
    "move), but it does NOT by itself explain the sign or magnitude of any CROSS-engine correlation, which "
    "would require a shared mechanism across engines that this diagnostic does not establish. Pure white "
    f"noise's first difference has a theoretical lag-1 autocorrelation of exactly {WHITE_NOISE_LAG1_REFERENCE} "
    "-- the observed within-engine values straddle this almost exactly, consistent with no real per-company "
    "drift. Whether any cross-engine r is real is answered by the cluster-bootstrap CI, not by this number."
)
ENGINE_LEVEL_CORRELATION_CAVEAT = (
    "n=6 (one point per consecutive day-pair) -- explicitly exploratory/weak, not a substitute for the "
    "company-day-pair-level statistic in value. This strips out company-level idiosyncratic noise and asks "
    "the more direct question: did the two engines' OVERALL qualifying population move in the same "
    "direction on the same days."
)
POOL_SIZE_EXPLANATION = (
    "n_by_pair is mechanically 6 x n_companies_by_pair (the 6 consecutive day-pairs times however many "
    "companies qualify in BOTH engines of the pair), minus any (company, day-pair) combo where either "
    "engine is missing that day. Pool sizes differ across pairs purely because the INTERSECTION of "
    "qualifying companies differs pair to pair -- not a different measurement window."
)


def _fisher_ci(r: float, n: int) -> dict[str, Any]:
    if n < 4 or abs(r) >= 1.0:
        return {"ci_low": None, "ci_high": None, "excludes_zero": None}
    z = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    ci_low, ci_high = math.tanh(z - CI_Z_95 * se), math.tanh(z + CI_Z_95 * se)
    return {"ci_low": ci_low, "ci_high": ci_high, "excludes_zero": (ci_low > 0.0) or (ci_high < 0.0)}


def _cluster_bootstrap_ci(
    company_points: dict[str, list[tuple[float, float]]], rng: np.random.Generator, n_resamples: int
) -> dict[str, Any]:
    """95% percentile-method bootstrap CI on Pearson r, resampling
    COMPANIES with replacement (each resampled company contributes its
    FULL set of day-pair points, possibly repeated) -- unlike Fisher z,
    this correctly treats a company's up-to-6 points as one non-
    independent cluster rather than 6 iid observations."""
    companies = list(company_points.keys())
    n = len(companies)
    if n < 2:
        return {"ci_low": None, "ci_high": None, "excludes_zero": None, "n_resamples_used": 0}

    rs: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        xs: list[float] = []
        ys: list[float] = []
        for i in idx:
            for dx, dy in company_points[companies[i]]:
                xs.append(dx)
                ys.append(dy)
        if len(xs) < 2:
            continue
        r, _p = pearsonr(xs, ys)
        if not math.isnan(r):
            rs.append(float(r))

    if len(rs) < 2:
        return {"ci_low": None, "ci_high": None, "excludes_zero": None, "n_resamples_used": len(rs)}
    rs.sort()
    ci_low = rs[int(0.025 * (len(rs) - 1))]
    ci_high = rs[int(0.975 * (len(rs) - 1))]
    return {
        "ci_low": ci_low,
        "ci_high": ci_high,
        "excludes_zero": (ci_low > 0.0) or (ci_high < 0.0),
        "n_resamples_used": len(rs),
    }


def _delta_series(day_rate: dict[int, float]) -> dict[tuple[int, int], float]:
    """day_rate: day_index -> rate. One entry per consecutive day pair
    where BOTH days are present for this company/engine -- a pair
    touching a missing day contributes nothing, never interpolated."""
    return {
        pair: day_rate[pair[1]] - day_rate[pair[0]]
        for pair in CONSECUTIVE_DAY_PAIRS
        if pair[0] in day_rate and pair[1] in day_rate
    }


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

    day_rate_by_entity_engine: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for (entity_id, engine, day_index), (n_mentioned, n_total) in rates.items():
        day_rate_by_entity_engine[(entity_id, engine)][day_index] = n_mentioned / n_total

    delta_by_entity_engine: dict[tuple[str, str], dict[tuple[int, int], float]] = {
        key: _delta_series(day_rate) for key, day_rate in day_rate_by_entity_engine.items()
    }

    value: dict[str, float | None] = {}
    n_by_pair: dict[str, int] = {}
    n_companies_by_pair: dict[str, int] = {}
    untestable_pairs: dict[str, str] = {}
    unexplained_pairs: dict[str, str] = {}
    matrix: dict[str, dict[str, float | None]] = {e1: {e2: (1.0 if e1 == e2 else None) for e2 in engines} for e1 in engines}
    computable_rs: list[float] = []
    ci_by_pair: dict[str, dict[str, Any]] = {}
    fisher_z_ci_by_pair_reference: dict[str, dict[str, Any]] = {}
    cluster_rng = np.random.default_rng(CLUSTER_BOOTSTRAP_SEED)

    for e1, e2 in itertools.combinations(engines, 2):
        pair_key = f"{e1}|{e2}"
        companies = qualifying.get(e1, set()) & qualifying.get(e2, set())
        company_points: dict[str, list[tuple[float, float]]] = {}
        for entity_id in sorted(companies):  # sorted: set order varies with PYTHONHASHSEED, the bootstrap must not
            d1 = delta_by_entity_engine.get((entity_id, e1), {})
            d2 = delta_by_entity_engine.get((entity_id, e2), {})
            points = [(d1[pair], d2[pair]) for pair in CONSECUTIVE_DAY_PAIRS if pair in d1 and pair in d2]
            if points:
                company_points[entity_id] = points

        xs = [dx for pts in company_points.values() for dx, _dy in pts]
        ys = [dy for pts in company_points.values() for _dx, dy in pts]

        n_by_pair[pair_key] = len(xs)
        n_companies_by_pair[pair_key] = len(companies)

        if len(companies) < MIN_QUALIFYING_FOR_PAIR:
            value[pair_key] = None
            untestable_pairs[pair_key] = f"only {len(companies)} companies qualify in both engines (need >= {MIN_QUALIFYING_FOR_PAIR})"
            ci_by_pair[pair_key] = {"reason_untestable": untestable_pairs[pair_key]}
            fisher_z_ci_by_pair_reference[pair_key] = {"reason_untestable": untestable_pairs[pair_key]}
            continue
        if len(xs) < 2:
            value[pair_key] = None
            untestable_pairs[pair_key] = "fewer than 2 paired (company, day-pair) observations"
            ci_by_pair[pair_key] = {"reason_untestable": untestable_pairs[pair_key]}
            fisher_z_ci_by_pair_reference[pair_key] = {"reason_untestable": untestable_pairs[pair_key]}
            continue

        r, _p = pearsonr(xs, ys)
        if math.isnan(r):
            value[pair_key] = None
            untestable_pairs[pair_key] = "degenerate correlation (zero variance in at least one engine's deltas)"
            ci_by_pair[pair_key] = {"reason_untestable": untestable_pairs[pair_key]}
            fisher_z_ci_by_pair_reference[pair_key] = {"reason_untestable": untestable_pairs[pair_key]}
            continue

        value[pair_key] = float(r)
        matrix[e1][e2] = float(r)
        matrix[e2][e1] = float(r)
        computable_rs.append(float(r))

        fisher_z_ci_by_pair_reference[pair_key] = {
            "r": float(r), "n": len(xs), "ci_level": 0.95, **_fisher_ci(float(r), len(xs))
        }
        cluster = _cluster_bootstrap_ci(company_points, cluster_rng, N_CLUSTER_BOOTSTRAP)
        ci_by_pair[pair_key] = {"r": float(r), "n": len(xs), "n_companies": len(company_points), "ci_level": 0.95, **cluster}
        if cluster.get("excludes_zero"):
            unexplained_pairs[pair_key] = (
                "exploratory, unexplained -- this cross-engine correlation survives the company-cluster "
                "bootstrap and is NOT attributable to within-engine autocorrelation (that mechanism doesn't "
                "apply across engines; see autocorrelation_caveat)"
            )

    mean_r = statistics.fmean(computable_rs) if computable_rs else None
    value["mean"] = mean_r
    verdict = None if mean_r is None else ("shared_movement" if mean_r >= SHARED_MOVEMENT_AT_OR_ABOVE else NO_EVIDENCE_OF_SHARED_MOVEMENT)

    # Within-engine lag-1 autocorrelation of the delta series -- reuses the
    # same MIN_QUALIFYING_FOR_PAIR floor as the observation-count gate,
    # rather than inventing a new threshold.
    within_engine_autocorr: dict[str, dict[str, Any]] = {}
    for engine in engines:
        xs, ys = [], []
        for entity_id in sorted(qualifying.get(engine, set())):
            d = delta_by_entity_engine.get((entity_id, engine), {})
            for pair_t, pair_t1 in LAG1_PAIRS:
                if pair_t in d and pair_t1 in d:
                    xs.append(d[pair_t])
                    ys.append(d[pair_t1])
        if len(xs) >= MIN_QUALIFYING_FOR_PAIR:
            r, _p = pearsonr(xs, ys)
            within_engine_autocorr[engine] = (
                {"lag1_r": None, "n_pairs": len(xs), "reason_untestable": "degenerate correlation"}
                if math.isnan(r)
                else {"lag1_r": float(r), "n_pairs": len(xs)}
            )
        else:
            within_engine_autocorr[engine] = {
                "lag1_r": None,
                "n_pairs": len(xs),
                "reason_untestable": f"fewer than {MIN_QUALIFYING_FOR_PAIR} paired lag-1 observations",
            }

    # Engine-level (n<=6) parallel statistic -- notes only, never value.
    engine_level_mean_delta: dict[str, dict[str, float | None]] = {}
    for engine in engines:
        per_pair_means: dict[str, float | None] = {}
        for pair in CONSECUTIVE_DAY_PAIRS:
            pair_key2 = f"{pair[0]}-{pair[1]}"
            deltas = [
                delta_by_entity_engine[(entity_id, engine)][pair]
                for entity_id in sorted(qualifying.get(engine, set()))
                if pair in delta_by_entity_engine.get((entity_id, engine), {})
            ]
            per_pair_means[pair_key2] = statistics.fmean(deltas) if deltas else None
        engine_level_mean_delta[engine] = per_pair_means

    engine_level_correlation: dict[str, dict[str, Any]] = {}
    for e1, e2 in itertools.combinations(engines, 2):
        pair_key = f"{e1}|{e2}"
        xs, ys = [], []
        for pair in CONSECUTIVE_DAY_PAIRS:
            k = f"{pair[0]}-{pair[1]}"
            v1, v2 = engine_level_mean_delta[e1].get(k), engine_level_mean_delta[e2].get(k)
            if v1 is not None and v2 is not None:
                xs.append(v1)
                ys.append(v2)
        if len(xs) >= 3:
            r, _p = pearsonr(xs, ys)
            engine_level_correlation[pair_key] = (
                {"r": None, "n": len(xs), "reason_untestable": "degenerate correlation"}
                if math.isnan(r)
                else {"r": float(r), "n": len(xs)}
            )
        else:
            engine_level_correlation[pair_key] = {"r": None, "n": len(xs), "reason_untestable": "fewer than 3 usable day-pairs"}

    # Summary rows are built from the SAME numbers as the table (value /
    # ci_by_pair), rounded once to CI_DECIMALS -- the table, the summary
    # and the ledger can never disagree because there is one source.
    summary_pairs = []
    for pair_key in sorted(k for k in value if k != "mean"):
        ci = ci_by_pair.get(pair_key, {})
        r = value[pair_key]
        summary_pairs.append(
            {
                "pair": pair_key,
                "r": round(r, CI_DECIMALS) if r is not None else None,
                "ci_low": round(ci["ci_low"], CI_DECIMALS) if ci.get("ci_low") is not None else None,
                "ci_high": round(ci["ci_high"], CI_DECIMALS) if ci.get("ci_high") is not None else None,
                "excludes_zero": ci.get("excludes_zero"),
                "label": unexplained_pairs.get(pair_key) and "exploratory, unexplained",
            }
        )
    survivors = [row["pair"] for row in summary_pairs if row["excludes_zero"]]

    return FigureResult(
        figure_id=FIGURE_ID,
        status="ok",
        value=value,
        denominator_before_exclusion=n_companies_by_pair,
        denominator_after_exclusion=n_by_pair,
        n_included=n_by_pair,
        n_excluded={"untestable_pairs": untestable_pairs},
        threshold={"shared_movement_at_or_above": SHARED_MOVEMENT_AT_OR_ABOVE},
        reason_untestable=None,
        notes={
            "engines": engines,
            "r_by_pair": {k: v for k, v in value.items() if k != "mean"},
            "mean_r": mean_r,
            "verdict": verdict,
            "matrix": matrix,
            "n_by_pair": n_by_pair,
            "n_companies_by_pair": n_companies_by_pair,
            "untestable_pairs": untestable_pairs,
            "min_qualifying_for_pair": MIN_QUALIFYING_FOR_PAIR,
            "v2_a2_read_together_sentence": V2_A2_READ_TOGETHER_SENTENCE,
            "ci_by_pair": ci_by_pair,
            "ci_caveat": CI_CAVEAT,
            "fisher_z_ci_by_pair_reference": fisher_z_ci_by_pair_reference,
            "cluster_bootstrap_seed": CLUSTER_BOOTSTRAP_SEED,
            "n_cluster_bootstrap": N_CLUSTER_BOOTSTRAP,
            "unexplained_pairs": unexplained_pairs,
            "ci_decimals": CI_DECIMALS,
            "summary_pairs": summary_pairs,
            "ci_excludes_zero_pairs": survivors,
            "multiple_pairs_note": MULTIPLE_PAIRS_NOTE,
            "within_engine_delta_autocorrelation": within_engine_autocorr,
            "autocorrelation_caveat": AUTOCORRELATION_CAVEAT,
            "white_noise_lag1_reference": WHITE_NOISE_LAG1_REFERENCE,
            "pool_size_explanation": POOL_SIZE_EXPLANATION,
            "engine_level_mean_delta_by_pair": engine_level_mean_delta,
            "engine_level_correlation": engine_level_correlation,
            "engine_level_correlation_caveat": ENGINE_LEVEL_CORRELATION_CAVEAT,
        },
    )
