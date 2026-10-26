"""Phase B shared intermediate: the company x intent x engine rates panel
every new Block-A figure (A1, V1, K1, A3) reads.

Built on run_company_panel.build_run_company_panel(inputs), not on
analysis.coding.b3_aggregates.build_cell_rates directly -- that function's
own denominators come from B1's raw run x entity cross product, never
filtered through analysis.phase_b.exclusions.usable_run_ids(), so it would
silently keep the 1 recorded collection error (and any future manual-UI
row) in n_runs. This panel re-derives n_runs/mention_rate/
recommendation_rate itself, over the same (usable-run-filtered,
ownership-corrected) rows S2 already reads, rather than trusting
b3_aggregates' unfiltered grouping.

Restricted to the six unbranded intents
(analysis.phase_b.citation_panel.UNBRANDED_INTENT_IDS, asserted against
prompts.csv by analysis.phase_b.intent_map at the top of every real run --
see that module's docstring) -- per EXPERIMENT.md 3.0, the primary metric
is mention rate on unbranded intents only.

qualifies carries the eligibility floor (>=5 mentions across unbranded
intents, within an engine) from the same EligibilityReport
build_run_company_panel already computes -- not recomputed here. Every
in-frame entity's cells are emitted regardless of qualifies, so a consumer
figure decides for itself whether to filter (V1's primary vs sensitivity
qualifying sets need this to differ per consumer, not be pre-filtered away
here).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from analysis.phase_b.citation_panel import UNBRANDED_INTENT_IDS
from analysis.phase_b.exclusions import usable_run_ids
from analysis.phase_b.reading import FrozenInputs
from analysis.phase_b.run_company_panel import RunCompanyRow, build_run_company_panel


@dataclass(frozen=True)
class RateCell:
    entity_id: str
    intent_id: int
    engine: str
    n_runs: int  # real usable-run denominator for this cell -- never the nominal 105
    n_mentioned: int
    n_recommended: int
    mention_rate: float
    recommendation_rate: float
    qualifies: bool  # this entity qualifies for this engine (>=5 mentions across unbranded intents)

    def to_json(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "intent_id": self.intent_id,
            "engine": self.engine,
            "n_runs": self.n_runs,
            "n_mentioned": self.n_mentioned,
            "n_recommended": self.n_recommended,
            "mention_rate": self.mention_rate,
            "recommendation_rate": self.recommendation_rate,
            "qualifies": self.qualifies,
        }


@dataclass(frozen=True)
class RatesPanelBuildReport:
    """No headline number lives here beyond qualifying_by_engine -- a build
    report, not a figure record, same convention as domain_ownership.
    BuildReport / citation_panel.CitationPanelBuildReport.

    qualifying_by_engine is the "qualifying company count per engine,
    printed first, before any other number" the task requires -- computed
    once here, printed by scripts/results/run_phase_b.py immediately after the
    intent map, before any other intermediate or figure. Every engine that
    produced at least one usable, unbranded-intent cell gets an entry, 0
    included explicitly rather than the engine being silently absent."""

    cells_produced: int
    engines_seen: list[str]
    qualifying_by_engine: dict[str, int]

    def to_json(self) -> dict[str, Any]:
        return {
            "cells_produced": self.cells_produced,
            "engines_seen": self.engines_seen,
            "qualifying_by_engine": self.qualifying_by_engine,
        }


def build_rates_panel(inputs: FrozenInputs) -> tuple[list[RateCell], RatesPanelBuildReport]:
    rows, eligibility, _panel_report = build_run_company_panel(inputs)
    usable_ids = usable_run_ids(inputs.raw_runs)

    scoped_rows: list[RunCompanyRow] = [
        r for r in rows if r.run_id in usable_ids and r.intent_id in UNBRANDED_INTENT_IDS
    ]

    groups: dict[tuple[str, int, str], list[RunCompanyRow]] = defaultdict(list)
    engines_seen: set[str] = set()
    for r in scoped_rows:
        groups[(r.entity_id, r.intent_id, r.engine)].append(r)
        engines_seen.add(r.engine)

    cells: list[RateCell] = []
    for (entity_id, intent_id, engine), group_rows in groups.items():
        n = len(group_rows)
        n_mentioned = sum(1 for r in group_rows if r.mentioned)
        n_recommended = sum(1 for r in group_rows if r.recommended)
        cells.append(
            RateCell(
                entity_id=entity_id,
                intent_id=intent_id,
                engine=engine,
                n_runs=n,
                n_mentioned=n_mentioned,
                n_recommended=n_recommended,
                mention_rate=n_mentioned / n,
                recommendation_rate=n_recommended / n,
                qualifies=entity_id in eligibility.qualifying_by_engine.get(engine, set()),
            )
        )

    report = RatesPanelBuildReport(
        cells_produced=len(cells),
        engines_seen=sorted(engines_seen),
        qualifying_by_engine={engine: eligibility.qualifying_count(engine) for engine in sorted(engines_seen)},
    )
    return cells, report
