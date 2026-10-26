"""O2 -- Paraphrase overlap against rep overlap (EXPLORATORY, pinned
EXPERIMENT.md / Notion Part 3.3).

Unit: (engine, intent_id). Reads
analysis.phase_b.daily_panel.build_daily_panel, restricted (as always) to
usable runs, the six unbranded intents, and qualifying companies.

- rep_set(engine, intent, paraphrase, day, rep): the qualifying companies
  mentioned in that ONE run.
- paraphrase_set(engine, intent, paraphrase, day): the qualifying
  companies mentioned in ANY of that paraphrase's usable reps on that
  SAME day (pooled across reps within the day only -- 2026-09-26 fix: an
  earlier version of this figure pooled a paraphrase's mentions across
  all 7 days while rep_set stayed single-day, which mixed cross-day drift
  into only one side of the comparison and made the two numbers not
  apples-to-apples. See the ambiguity ledger,
  o2-same-day-paraphrase-set-redefinition-2026-09-26).
- Rep-overlap, one number per (engine, intent): for each (day, paraphrase)
  with >=2 usable reps present, the mean pairwise Jaccard across that
  day's rep sets; averaged across every (day, paraphrase) combination
  that had a computable value.
- Paraphrase-overlap, one number per (engine, intent): for each day with
  >=2 paraphrases present, the mean pairwise Jaccard across that day's
  paraphrase sets; averaged across every day that had a computable value
  -- the exact same two-stage shape as rep-overlap, one level up, so both
  statistics are measured over the same time window.
- No pre-registered floor (exploratory) -- only a zero-daily-panel-cells
  guard at the whole-figure level.
- Figure-level direction: a 95% bootstrap CI (10,000 resamples, fixed
  seed) on the difference (mean paraphrase-overlap minus mean rep-overlap)
  across every computable (engine, intent) pair -- resampling those pairs
  themselves, not days or reps, since within-week/within-company draws
  are not independent and the pair-level mean is the real unit the
  headline numbers are taken over. "no_meaningful_difference" when the CI
  crosses zero; otherwise "paraphrase_lower" / "rep_lower" by sign. Per
  the spec's own instruction ("report it either way"), and per the
  reviewer's correction (2026-09-26): a near-tie here is read as SUPPORT
  for the design's unit-of-analysis choice (paraphrases behave like
  replicates of the same intent), not as a falsification.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from itertools import combinations
from typing import Any

import numpy as np

from analysis.phase_b.daily_panel import build_daily_panel
from analysis.phase_b.figure_shape import FigureResult
from analysis.phase_b.figures.a1 import INTENT_THEME_NAMES
from analysis.phase_b.reading import FrozenInputs

FIGURE_ID = "O2"
BOOTSTRAP_SEED = 20260926
BOOTSTRAP_N_RESAMPLES = 10_000
BOOTSTRAP_CI_LEVEL = 0.95

NO_MEANINGFUL_DIFFERENCE_SENTENCE = (
    "Paraphrase overlap and rep overlap are statistically indistinguishable (the 95% bootstrap CI on their "
    "difference crosses zero) -- reruns of the exact same string vary about as much week-to-week as different "
    "paraphrases of the same intent do. This SUPPORTS the design's unit-of-analysis choice: intent, not exact "
    "wording, is the right thing to treat as one unit, with paraphrases behaving as replicates rather than a "
    "materially different condition."
)
PARAPHRASE_LOWER_SENTENCE = (
    "Paraphrase overlap is lower than rep overlap, and the 95% CI on the difference excludes zero -- the "
    "design's assumption holds: which exact string was used moves the company set more than which single run did."
)
REP_LOWER_SENTENCE = (
    "Rep overlap is lower than paraphrase overlap, and the 95% CI on the difference excludes zero -- the "
    "OPPOSITE of the design's assumption: which exact run (day) was caught matters more than which paraphrase "
    "was used."
)
UNIT_OF_ANALYSIS_NOTE = (
    "Because paraphrase overlap and rep overlap are statistically tied, individual paraphrases behave like "
    "additional replicates of the same intent rather than a distinct condition -- the effective sample for "
    "'how much does exact wording matter' is closer to the full ~105 (paraphrase x rep x day) draws per "
    "(engine, intent) than to the 5 nominal paraphrases alone."
)
PER_CELL_DIRECTION_CAVEAT = (
    "Per-(engine,intent) direction above is a strict, uncalibrated inequality (no CI) -- only the figure-level "
    "notes['direction'] carries a statistical (bootstrap-CI) claim."
)
# 2026-09-27 review: confirmed this bootstrap already resamples at the
# (engine, intent) CLUSTER level, not individual pairs -- each of the ~24
# resampled units is itself an aggregate (a mean Jaccard over many
# companies and days), not a raw per-company observation, so no further
# company-level nesting is needed inside it (unlike A2's per-company
# clustered delta observations, which do need that).
RESAMPLE_UNIT_NOTE = (
    "Resamples the ~24 (engine, intent) pairs themselves, with replacement -- confirmed cluster-level, not "
    "individual-pair-level. Each (engine, intent) pair is already an aggregate statistic (a mean Jaccard over "
    "many companies and days), not a raw per-company observation, so resampling at that level is the correct "
    "unit for this statistic; no further company-level nesting is needed inside it."
)


def _pairwise_mean_jaccard(sets: list[set[str]]) -> float | None:
    if len(sets) < 2:
        return None
    jaccards = []
    for a, b in combinations(sets, 2):
        union = a | b
        jaccards.append(len(a & b) / len(union) if union else 1.0)
    return statistics.fmean(jaccards)


def _bootstrap_diff_ci(pairs: list[tuple[float, float]], n_resamples: int, seed: int) -> dict[str, float]:
    """95% percentile-method bootstrap CI on mean(paraphrase) - mean(rep),
    resampling the (engine, intent) pairs themselves (with replacement) --
    the real unit the two headline means are taken over."""
    rng = np.random.default_rng(seed)
    arr = np.array(pairs)  # shape (n, 2)
    n = len(arr)
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled = arr[idx]  # shape (n_resamples, n, 2)
    diffs = resampled[:, :, 0].mean(axis=1) - resampled[:, :, 1].mean(axis=1)
    alpha = (1 - BOOTSTRAP_CI_LEVEL) / 2
    return {
        "ci_low": float(np.percentile(diffs, 100 * alpha)),
        "ci_high": float(np.percentile(diffs, 100 * (1 - alpha))),
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

    rep_sets: dict[tuple[str, int, int, int, int], set[str]] = defaultdict(set)
    paraphrase_sets_by_day: dict[tuple[str, int, int, int], set[str]] = defaultdict(set)
    rep_indices_by_day_paraphrase: dict[tuple[str, int, int, int], set[int]] = defaultdict(set)
    paraphrase_ids_by_day: dict[tuple[str, int, int], set[int]] = defaultdict(set)
    paraphrases_by_engine_intent: dict[tuple[str, int], set[int]] = defaultdict(set)

    for cell in cells:
        run_key = (cell.engine, cell.intent_id, cell.paraphrase_id, cell.day_index, cell.rep_index)
        day_paraphrase_key = (cell.engine, cell.intent_id, cell.paraphrase_id, cell.day_index)
        day_key = (cell.engine, cell.intent_id, cell.day_index)
        rep_indices_by_day_paraphrase[day_paraphrase_key].add(cell.rep_index)
        paraphrase_ids_by_day[day_key].add(cell.paraphrase_id)
        paraphrases_by_engine_intent[(cell.engine, cell.intent_id)].add(cell.paraphrase_id)
        if cell.mentioned:
            rep_sets[run_key].add(cell.entity_id)
            paraphrase_sets_by_day[day_paraphrase_key].add(cell.entity_id)

    # Rep-overlap components, per (engine, intent) -- one value per (day,
    # paraphrase) that had >=2 usable reps, averaged afterward.
    rep_overlap_components: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (engine, intent_id, paraphrase_id, day_index), rep_indices in rep_indices_by_day_paraphrase.items():
        rep_sets_present = [rep_sets.get((engine, intent_id, paraphrase_id, day_index, r), set()) for r in sorted(rep_indices)]
        m = _pairwise_mean_jaccard(rep_sets_present)
        if m is not None:
            rep_overlap_components[(engine, intent_id)].append(m)

    # Paraphrase-overlap components, per (engine, intent) -- one value per
    # day that had >=2 paraphrases present, averaged afterward. Same shape
    # as rep-overlap above, one level up, so both are same-day.
    paraphrase_overlap_components: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (engine, intent_id, day_index), paraphrase_ids in paraphrase_ids_by_day.items():
        sets_present = [paraphrase_sets_by_day.get((engine, intent_id, p, day_index), set()) for p in sorted(paraphrase_ids)]
        m = _pairwise_mean_jaccard(sets_present)
        if m is not None:
            paraphrase_overlap_components[(engine, intent_id)].append(m)

    engines = sorted({engine for engine, _intent in paraphrases_by_engine_intent})
    value: dict[str, dict[int, dict[str, Any]]] = {}
    engine_intent_pairs: list[tuple[float, float]] = []  # (paraphrase_jaccard, rep_jaccard), matched

    for engine in engines:
        value[engine] = {}
        intents = sorted(intent_id for e, intent_id in paraphrases_by_engine_intent if e == engine)
        for intent_id in intents:
            paraphrase_ids = sorted(paraphrases_by_engine_intent[(engine, intent_id)])
            paraphrase_components = paraphrase_overlap_components.get((engine, intent_id), [])
            paraphrase_jaccard = statistics.fmean(paraphrase_components) if paraphrase_components else None
            rep_components = rep_overlap_components.get((engine, intent_id), [])
            rep_jaccard = statistics.fmean(rep_components) if rep_components else None

            direction: str | None = None
            if paraphrase_jaccard is not None and rep_jaccard is not None:
                if paraphrase_jaccard < rep_jaccard:
                    direction = "paraphrase_lower"
                elif paraphrase_jaccard > rep_jaccard:
                    direction = "rep_lower"
                else:
                    direction = "tied"
                engine_intent_pairs.append((paraphrase_jaccard, rep_jaccard))

            value[engine][intent_id] = {
                "paraphrase_jaccard": paraphrase_jaccard,
                "rep_jaccard": rep_jaccard,
                "n_paraphrases": len(paraphrase_ids),
                "n_paraphrase_overlap_components": len(paraphrase_components),
                "n_rep_overlap_components": len(rep_components),
                "direction": direction,
                "intent_name": INTENT_THEME_NAMES.get(intent_id, f"intent {intent_id}"),
            }

    overall_paraphrase_mean = statistics.fmean(p for p, _r in engine_intent_pairs) if engine_intent_pairs else None
    overall_rep_mean = statistics.fmean(r for _p, r in engine_intent_pairs) if engine_intent_pairs else None
    overall_diff = None
    bootstrap: dict[str, Any] | None = None

    if overall_paraphrase_mean is None or overall_rep_mean is None:
        overall_direction = None
        direction_sentence = "Not enough data to compare paraphrase overlap against rep overlap."
    else:
        overall_diff = overall_paraphrase_mean - overall_rep_mean
        ci = _bootstrap_diff_ci(engine_intent_pairs, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED)
        crosses_zero = ci["ci_low"] <= 0.0 <= ci["ci_high"]
        bootstrap = {
            "n_resamples": BOOTSTRAP_N_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "resample_unit": "engine_intent_pair",
            "resample_unit_note": RESAMPLE_UNIT_NOTE,
            "ci_level": BOOTSTRAP_CI_LEVEL,
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
            "crosses_zero": crosses_zero,
        }
        if crosses_zero:
            overall_direction = "no_meaningful_difference"
            direction_sentence = NO_MEANINGFUL_DIFFERENCE_SENTENCE
        elif overall_diff < 0:
            overall_direction = "paraphrase_lower"
            direction_sentence = PARAPHRASE_LOWER_SENTENCE
        else:
            overall_direction = "rep_lower"
            direction_sentence = REP_LOWER_SENTENCE

    return FigureResult(
        figure_id=FIGURE_ID,
        status="ok",
        value=value,
        denominator_before_exclusion=sum(len(cells_by_intent) for cells_by_intent in value.values()),
        denominator_after_exclusion=len(engine_intent_pairs),
        n_included=len(engine_intent_pairs),
        n_excluded={
            "engine_intent_pair_without_both_overlaps": sum(len(cells_by_intent) for cells_by_intent in value.values())
            - len(engine_intent_pairs)
        },
        threshold=None,  # exploratory -- no pre-registered band (Notion Part 3.3)
        reason_untestable=None,
        notes={
            "engines": engines,
            "direction": overall_direction,
            "direction_sentence": direction_sentence,
            "overall_paraphrase_mean": overall_paraphrase_mean,
            "overall_rep_mean": overall_rep_mean,
            "overall_diff_paraphrase_minus_rep": overall_diff,
            "bootstrap": bootstrap,
            "unit_of_analysis_note": UNIT_OF_ANALYSIS_NOTE if overall_direction == "no_meaningful_difference" else None,
            "per_cell_direction_caveat": PER_CELL_DIRECTION_CAVEAT,
        },
    )
