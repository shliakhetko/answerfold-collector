"""CLAUDE_PROBATION -- the user's probation rule for the Claude engine,
serialized as a figure-shaped record. Pre-registered (1.1), not
exploratory (user decision 2026-09-29). Every judgment call (population,
metric, tercile tie handling, the meaning of "mention without search",
"most") is fixed in the ledger entry `claude-probation-judgment-calls-
2026-09-29`, written before this figure was ever computed.

The rule, verbatim from the user:
- Population: qualifying companies, unbranded intents only.
- Metric: company-level mention rate per engine; rank by percentile
  within each engine.
- Set A: top tercile in Claude AND bottom tercile in AI Mode. Set B: the
  reverse. Zero in both engines excluded.
- Keep only if (1) each set has >= 3 companies, and (2) most Set A
  Claude mentions come from runs with no search fired or no source
  cited. Otherwise cut, and state which condition failed.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from analysis.phase_b.citation_panel import UNBRANDED_INTENT_IDS
from analysis.phase_b.exclusions import usable_run_ids
from analysis.phase_b.figure_shape import FigureResult
from analysis.phase_b.reading import FrozenInputs
from analysis.phase_b.run_company_panel import build_run_company_panel

FIGURE_ID = "CLAUDE_PROBATION"
CLAUDE = "anthropic_claude"
AI_MODE = "google_ai_mode"
MIN_SET_SIZE = 3
LEDGER_REF = "claude-probation-judgment-calls-2026-09-29"
COND_SET_SIZE = "set_size_below_3"
COND_NO_SEARCH = "set_a_mentions_not_mostly_without_search"


def _mid_rank_percentile(rate: float, all_rates: list[float]) -> float:
    below = sum(1 for r in all_rates if r < rate)
    tied = sum(1 for r in all_rates if r == rate)
    return (below + 0.5 * tied) / len(all_rates)


def _tercile_membership(rates: dict[str, float]) -> tuple[set[str], set[str], float, float]:
    """Value-based cut: top = rate >= 2/3 quantile and > 0; bottom =
    rate <= 1/3 quantile. Ties at a boundary are all included."""
    values = np.array(list(rates.values()), dtype=float)
    q_low = float(np.quantile(values, 1 / 3))
    q_high = float(np.quantile(values, 2 / 3))
    top = {e for e, r in rates.items() if r >= q_high and r > 0}
    bottom = {e for e, r in rates.items() if r <= q_low}
    return top, bottom, q_low, q_high


def compute(inputs: FrozenInputs) -> FigureResult:
    rows, eligibility, _report = build_run_company_panel(inputs)
    usable = usable_run_ids(inputs.raw_runs)
    scoped = [r for r in rows if r.run_id in usable and r.intent_id in UNBRANDED_INTENT_IDS and r.engine in (CLAUDE, AI_MODE)]

    runs_per_engine: dict[str, set[str]] = defaultdict(set)
    mentioned_runs: dict[tuple[str, str], set[str]] = defaultdict(set)  # (engine, entity) -> run ids
    for r in scoped:
        runs_per_engine[r.engine].add(r.run_id)
        if r.mentioned:
            mentioned_runs[(r.engine, r.entity_id)].add(r.run_id)

    if not runs_per_engine.get(CLAUDE) or not runs_per_engine.get(AI_MODE):
        return FigureResult(
            figure_id=FIGURE_ID, status="untestable", value=None, denominator_before_exclusion=None,
            denominator_after_exclusion=None, n_included=None, n_excluded=None, threshold=None,
            reason_untestable="no usable unbranded runs for Claude and/or AI Mode",
        )

    population_all = set(eligibility.qualifying_by_engine.get(CLAUDE, set())) | set(
        eligibility.qualifying_by_engine.get(AI_MODE, set())
    )
    rate = {
        engine: {
            e: len(mentioned_runs.get((engine, e), set())) / len(runs_per_engine[engine]) for e in population_all
        }
        for engine in (CLAUDE, AI_MODE)
    }
    zero_both = {e for e in population_all if rate[CLAUDE][e] == 0 and rate[AI_MODE][e] == 0}
    population = population_all - zero_both

    if len(population) < 3:
        return FigureResult(
            figure_id=FIGURE_ID, status="untestable", value=None, denominator_before_exclusion=None,
            denominator_after_exclusion=None, n_included=None, n_excluded=None, threshold=None,
            reason_untestable=f"only {len(population)} companies in the probation population",
        )

    rates_c = {e: rate[CLAUDE][e] for e in population}
    rates_a = {e: rate[AI_MODE][e] for e in population}
    top_c, bottom_c, qc_low, qc_high = _tercile_membership(rates_c)
    top_a, bottom_a, qa_low, qa_high = _tercile_membership(rates_a)
    set_a = top_c & bottom_a
    set_b = top_a & bottom_c

    # Set A's Claude mentions: (run, company) pairs, mentioned, in usable unbranded Claude runs.
    run_meta = {r["run_id"]: r for r in inputs.raw_runs}
    n_a_mentions = 0
    n_a_without_search = 0
    for e in set_a:
        for run_id in mentioned_runs.get((CLAUDE, e), set()):
            n_a_mentions += 1
            no_search = run_meta[run_id].get("search_fired") is False
            no_source = len(inputs.sources_by_run.get(run_id, [])) == 0
            if no_search or no_source:
                n_a_without_search += 1
    share = (n_a_without_search / n_a_mentions) if n_a_mentions else None

    failed: list[str] = []
    if len(set_a) < MIN_SET_SIZE or len(set_b) < MIN_SET_SIZE:
        failed.append(COND_SET_SIZE)
    if share is None or not share > 0.5:
        failed.append(COND_NO_SEARCH)
    verdict = "keep" if not failed else "cut"

    def describe(members: set[str]) -> list[dict[str, Any]]:
        return [
            {
                "entity_id": e,
                "claude_mention_rate": rates_c[e],
                "claude_percentile": _mid_rank_percentile(rates_c[e], list(rates_c.values())),
                "ai_mode_mention_rate": rates_a[e],
                "ai_mode_percentile": _mid_rank_percentile(rates_a[e], list(rates_a.values())),
            }
            for e in sorted(members)
        ]

    return FigureResult(
        figure_id=FIGURE_ID,
        status="ok",
        value={
            "verdict": verdict,
            "failed_conditions": failed,
            "set_a_size": len(set_a),
            "set_b_size": len(set_b),
            "set_a_claude_mentions": n_a_mentions,
            "set_a_claude_mentions_without_search_or_source": n_a_without_search,
            "share_set_a_mentions_without_search": share,
        },
        denominator_before_exclusion=len(population_all),
        denominator_after_exclusion=len(population),
        n_included=len(population),
        n_excluded={"zero_in_both_engines": len(zero_both)},
        threshold={"min_set_size": MIN_SET_SIZE, "set_a_share_without_search_more_than": 0.5},
        reason_untestable=None,
        supporting_table=[
            {"set": "A", **row} for row in describe(set_a)
        ] + [{"set": "B", **row} for row in describe(set_b)],
        notes={
            "exploratory": False,
            "pre_registered": "1.1",
            "ledger_ref": LEDGER_REF,
            "set_a_definition": "top tercile in Claude AND bottom tercile in AI Mode",
            "set_b_definition": "top tercile in AI Mode AND bottom tercile in Claude",
            "population_definition": "companies qualifying (>=5 unbranded mentions) in Claude or AI Mode; unbranded intents only",
            "tercile_cuts": {
                CLAUDE: {"q_1_3": qc_low, "q_2_3": qc_high},
                AI_MODE: {"q_1_3": qa_low, "q_2_3": qa_high},
            },
            "n_usable_unbranded_runs": {e: len(runs_per_engine[e]) for e in (CLAUDE, AI_MODE)},
            "without_search_definition": "run has search_fired=False OR zero cited sources",
            "set_a": sorted(set_a),
            "set_b": sorted(set_b),
        },
    )
