"""A3 -- Mention rank against recommendation rank (EXPLORATORY, pinned
EXPERIMENT.md / Notion Part 3.3).

Per engine, qualifying companies only (the eligibility floor applies to
every per-company figure, Notion Part 3.0): Spearman rank correlation
between a company's mention rank and its recommendation rank. Mention
rate and recommendation rate are each companies' overall rate across the
six unbranded intents (totals pooled from
analysis.phase_b.rates_panel.build_rates_panel's per-intent RateCell
rows, not recomputed) -- rank 1 is the highest rate in that engine. Ties
use average-rank (scipy.stats.rankdata), since recommendation rate in
particular is frequently tied at or near zero.

No pre-registered threshold or falsifier -- A3 is exploratory (Notion
Part 3.3, distinct from the three confirmatory claims and the
pre-specified tables). The output that matters per the spec is the
teardown: companies whose recommendation rank is at least
TEARDOWN_RANK_GAP places WORSE than their mention rank -- named often,
recommended less; the "named as someone else's competitor" failure mode.

2026-09-25 fix: this used to flag on the unsigned gap
(`abs(mention_rank - recommendation_rank) >= 10`), which the render
script's footer then described in only one direction. On real data that
mislabeled 3 companies (anthropic_claude's call-agent-ai/linda-ai/phone2,
whose recommendation rank is BETTER than their mention rank -- the
opposite failure mode, and not one this figure claims to measure) --
those companies are no longer flagged. Verified separately that
average-rank tie-breaking at the bottom of either axis cannot manufacture
a gap this large on its own on this data (see
notes["bottom_rank_tie_check"]) -- every remaining flagged company has a
genuine cross-axis rate difference behind it.

An engine with fewer than MIN_QUALIFYING_COMPANIES qualifying companies,
or a degenerate rho (e.g. every company tied on recommendation rate), is
marked untestable for A3 specifically without stopping the whole figure.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from scipy.stats import rankdata, spearmanr

from analysis.phase_b.figure_shape import FigureResult
from analysis.phase_b.rates_panel import build_rates_panel
from analysis.phase_b.reading import FrozenInputs

FIGURE_ID = "A3"
TEARDOWN_RANK_GAP = 10
MIN_QUALIFYING_COMPANIES = 3  # below this, a rank correlation is not a meaningful statistic


def _bottom_rank_tie_check(mention_ranks, recommendation_ranks) -> dict[str, Any]:
    """2026-09-25 fix: verifies average-rank tie-breaking at the WORST
    (highest-numbered) rank on either axis can't manufacture a teardown
    gap on its own. A tie block of k companies at the bottom occupies the
    last k integer rank slots and all get the mean of those slots, so the
    furthest any one of them can sit from the single "true worst" slot is
    (k-1)/2 -- that's the max distortion a bottom tie block can contribute
    to a cross-axis gap if the other axis has no tie there at all."""
    n_tied_mention = int((mention_ranks == max(mention_ranks)).sum())
    n_tied_recommendation = int((recommendation_ranks == max(recommendation_ranks)).sum())
    max_distortion = max((n_tied_mention - 1) / 2, (n_tied_recommendation - 1) / 2)
    reaches_threshold = max_distortion >= TEARDOWN_RANK_GAP
    return {
        "n_tied_at_bottom_mention": n_tied_mention,
        "n_tied_at_bottom_recommendation": n_tied_recommendation,
        "max_possible_tie_only_distortion": max_distortion,
        "ties_alone_can_reach_threshold": reaches_threshold,
        "sentence": (
            f"{n_tied_mention} companies tied at the worst mention rank, {n_tied_recommendation} tied at the "
            f"worst recommendation rank -- with average-rank tie-breaking, a tie block that size can distort a "
            f"rank by at most {max_distortion:.1f} places, "
            + ("enough to reach" if reaches_threshold else "short of")
            + f" the {TEARDOWN_RANK_GAP}-place teardown threshold on its own."
        ),
    }


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
    qualifying_by_engine: dict[str, set[str]] = defaultdict(set)
    totals: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"n_runs": 0, "n_mentioned": 0, "n_recommended": 0})
    for c in cells:
        if c.qualifies:
            qualifying_by_engine[c.engine].add(c.entity_id)
        t = totals[(c.entity_id, c.engine)]
        t["n_runs"] += c.n_runs
        t["n_mentioned"] += c.n_mentioned
        t["n_recommended"] += c.n_recommended

    value: dict[str, float | None] = {}
    denominator_before_exclusion: dict[str, int] = {}
    denominator_after_exclusion: dict[str, int] = {}
    n_included: dict[str, int] = {}
    n_excluded: dict[str, dict[str, Any]] = {}
    teardown_by_engine: dict[str, list[dict[str, Any]]] = {}
    scatter_points_by_engine: dict[str, list[dict[str, Any]]] = {}
    bottom_rank_tie_check: dict[str, dict[str, Any]] = {}
    untestable_engines: dict[str, str] = {}

    for engine in engines:
        qualifying = sorted(qualifying_by_engine.get(engine, set()))
        n_qual = len(qualifying)
        denominator_before_exclusion[engine] = n_qual

        if n_qual < MIN_QUALIFYING_COMPANIES:
            value[engine] = None
            n_included[engine] = n_qual
            denominator_after_exclusion[engine] = n_qual
            n_excluded[engine] = {}
            untestable_engines[engine] = f"only {n_qual} qualifying companies (need >= {MIN_QUALIFYING_COMPANIES})"
            continue

        mention_rates = [totals[(e, engine)]["n_mentioned"] / totals[(e, engine)]["n_runs"] for e in qualifying]
        recommendation_rates = [totals[(e, engine)]["n_recommended"] / totals[(e, engine)]["n_runs"] for e in qualifying]

        rho, _p = spearmanr(mention_rates, recommendation_rates)
        # rank 1 == highest rate: rankdata ranks ascending, so rank the negated values
        mention_ranks = rankdata([-r for r in mention_rates], method="average")
        recommendation_ranks = rankdata([-r for r in recommendation_rates], method="average")

        points: list[dict[str, Any]] = []
        teardown: list[dict[str, Any]] = []
        for entity_id, m_rank, r_rank, m_rate, r_rate in zip(
            qualifying, mention_ranks, recommendation_ranks, mention_rates, recommendation_rates
        ):
            # 2026-09-25 fix: one-directional now -- only flags recommendation
            # rank being WORSE (a higher rank number) than mention rank by
            # >= TEARDOWN_RANK_GAP. A company recommended more than it's
            # mentioned (r_rank better than m_rank) is the opposite pattern
            # and is never flagged here (see module docstring).
            gap = r_rank - m_rank
            points.append(
                {
                    "entity_id": entity_id, "mention_rank": m_rank, "recommendation_rank": r_rank,
                    "mention_rate": m_rate, "recommendation_rate": r_rate,
                }
            )
            if gap >= TEARDOWN_RANK_GAP:
                teardown.append(
                    {"entity_id": entity_id, "mention_rank": m_rank, "recommendation_rank": r_rank, "gap": gap}
                )
        scatter_points_by_engine[engine] = points
        teardown_by_engine[engine] = sorted(teardown, key=lambda t: -t["gap"])
        bottom_rank_tie_check[engine] = _bottom_rank_tie_check(mention_ranks, recommendation_ranks)

        denominator_after_exclusion[engine] = n_qual
        n_included[engine] = n_qual
        n_excluded[engine] = {}

        if math.isnan(rho):
            value[engine] = None
            untestable_engines[engine] = "degenerate correlation (a rate is constant across all qualifying companies)"
            continue

        value[engine] = float(rho)

    return FigureResult(
        figure_id=FIGURE_ID,
        status="ok",
        value=value,
        denominator_before_exclusion=denominator_before_exclusion,
        denominator_after_exclusion=denominator_after_exclusion,
        n_included=n_included,
        n_excluded=n_excluded,
        threshold=None,  # exploratory -- no pre-registered band (Notion Part 3.3)
        reason_untestable=None,
        notes={
            "teardown_by_engine": teardown_by_engine,
            "scatter_points_by_engine": scatter_points_by_engine,
            "untestable_engines": untestable_engines,
            "teardown_rank_gap": TEARDOWN_RANK_GAP,
            "bottom_rank_tie_check": bottom_rank_tie_check,
        },
    )
