"""Phase B shared intermediate: the daily panel V2, A2, and O1 all read.

One row per (entity_id, engine, intent_id, paraphrase_id, day_index,
rep_index) -- mentioned 0/1 -- restricted to usable runs
(analysis.phase_b.exclusions.usable_run_ids), the six unbranded intents
(analysis.phase_b.citation_panel.UNBRANDED_INTENT_IDS), and qualifying
entities only (the eligibility floor, from the same EligibilityReport
analysis.phase_b.run_company_panel.build_run_company_panel already
computes -- not recomputed here).

A missing or excluded cell is simply absent from the returned list --
never backfilled, never imputed. "A missed day stays missed" (the task's
own words) is enforced by omission: every consumer counts however many
reps/days it actually found for a given key, it never assumes 3 reps or 7
days.

Also carries check_daily_completeness(), the Frog block's literal "first
action": for every (engine, intent_id, paraphrase_id, day_index) cell in
the full raw grid (all 7 intents, not just the six unbranded -- this is a
raw-collection-integrity check, not a per-figure eligibility one), count
usable reps and compare to the expected 3. A gap already known and
recorded (KNOWN_INCOMPLETE_CELLS, cross-referenced to
reports/collection_audit.md) is reported as such, not as a fresh gate
failure; anything else is.

Full raw grid = 4 engines x 7 intents (1-6 unbranded + 7 branded
"alternatives to X") x 5 paraphrases x 7 days = 980 cells -- NOT
4 x 8 x 5 x 7 = 1,120. The study has exactly 7 intents total
(EXPERIMENT.md's intent table; config/prompts/prompts.csv has exactly 35 rows =
7 x 5), so a reader arriving at 1,120 has assumed an 8th intent that does
not exist (2026-09-26 review, see the ambiguity ledger).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from analysis.phase_b.citation_panel import UNBRANDED_INTENT_IDS
from analysis.phase_b.exclusions import usable_run_ids
from analysis.phase_b.reading import FrozenInputs
from analysis.phase_b.run_company_panel import build_run_company_panel
from collector import paths

EXPECTED_REPS_PER_CELL = 3
EXPECTED_DAYS = tuple(range(1, 8))
# The 6 consecutive day pairs V2/A2/O1 all read: (1,2), (2,3) ... (6,7) --
# never all 21 possible pairs.
CONSECUTIVE_DAY_PAIRS = tuple(zip(EXPECTED_DAYS, EXPECTED_DAYS[1:]))

# Known, already-documented gaps in raw collection -- each entry names the
# exact run_id(s) responsible and where the gap is already on record, so
# check_daily_completeness can tell "a fact we already know about" from "a
# new problem" without silently swallowing either. Whitelisting a cell here
# does not change how it's used downstream: it is still simply absent from
# build_daily_panel's rows (never imputed) -- it is just not treated as a
# fresh gate failure by scripts/publish/gate_check.py.
KNOWN_INCOMPLETE_CELLS: dict[tuple[str, int, int, int], dict[str, Any]] = {
    ("google_gemini", 6, 4, 7): {
        "usable_reps": 2,
        "run_ids": ["google_gemini-i6-p4-d7-r3"],
        "reason": "500 INTERNAL, not retried",
        "ledger_ref": paths.rel(paths.COLLECTION_AUDIT_MD).as_posix(),
    },
}


@dataclass(frozen=True)
class DailyPanelCell:
    entity_id: str
    engine: str
    intent_id: int
    paraphrase_id: int
    day_index: int
    rep_index: int
    run_id: str
    mentioned: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "engine": self.engine,
            "intent_id": self.intent_id,
            "paraphrase_id": self.paraphrase_id,
            "day_index": self.day_index,
            "rep_index": self.rep_index,
            "run_id": self.run_id,
            "mentioned": self.mentioned,
        }


@dataclass(frozen=True)
class DailyCompletenessReport:
    """Fine-grain (engine, intent_id, paraphrase_id, day_index) usable-rep
    counts, over the FULL raw grid (all 7 intents -- a raw-collection-
    integrity check, independent of which intents/entities later qualify
    for any figure). A cell with usable_reps != EXPECTED_REPS_PER_CELL is
    either an already-known gap (known_incomplete) or a fresh one
    (unwhitelisted_missing) -- only the latter fails the gate."""

    total_cells_expected: int
    known_incomplete: list[dict[str, Any]]
    unwhitelisted_missing: list[dict[str, Any]]

    @property
    def is_complete(self) -> bool:
        """"Complete" means no FRESH gap -- a known, whitelisted gap does
        not flip this to False. See KNOWN_INCOMPLETE_CELLS' docstring."""
        return not self.unwhitelisted_missing

    def to_json(self) -> dict[str, Any]:
        return {
            "total_cells_expected": self.total_cells_expected,
            "known_incomplete": self.known_incomplete,
            "unwhitelisted_missing": self.unwhitelisted_missing,
            "is_complete": self.is_complete,
        }


def check_daily_completeness(raw_runs: list[dict[str, Any]]) -> DailyCompletenessReport:
    usable_ids = usable_run_ids(raw_runs)
    reps_by_cell: dict[tuple[str, int, int, int], set[int]] = defaultdict(set)
    all_cells: set[tuple[str, int, int, int]] = set()
    for run in raw_runs:
        cell = (run["engine"], run["intent_id"], run["paraphrase_id"], run["day_index"])
        all_cells.add(cell)
        if run["run_id"] in usable_ids:
            reps_by_cell[cell].add(run["rep_index"])

    known_incomplete: list[dict[str, Any]] = []
    unwhitelisted_missing: list[dict[str, Any]] = []
    for cell in sorted(all_cells):
        engine, intent_id, paraphrase_id, day_index = cell
        n_usable = len(reps_by_cell.get(cell, set()))
        if n_usable == EXPECTED_REPS_PER_CELL:
            continue
        record = {
            "engine": engine,
            "intent_id": intent_id,
            "paraphrase_id": paraphrase_id,
            "day_index": day_index,
            "usable_reps": n_usable,
            "expected_reps": EXPECTED_REPS_PER_CELL,
        }
        known = KNOWN_INCOMPLETE_CELLS.get(cell)
        if known is not None and known["usable_reps"] == n_usable:
            record.update(known)
            known_incomplete.append(record)
        else:
            unwhitelisted_missing.append(record)

    return DailyCompletenessReport(
        total_cells_expected=len(all_cells),
        known_incomplete=known_incomplete,
        unwhitelisted_missing=unwhitelisted_missing,
    )


@dataclass(frozen=True)
class DailyPanelBuildReport:
    cells_produced: int
    qualifying_by_engine: dict[str, int]
    completeness: DailyCompletenessReport

    def to_json(self) -> dict[str, Any]:
        return {
            "cells_produced": self.cells_produced,
            "qualifying_by_engine": self.qualifying_by_engine,
            "completeness": self.completeness.to_json(),
        }


def build_daily_panel(inputs: FrozenInputs) -> tuple[list[DailyPanelCell], DailyPanelBuildReport]:
    rows, eligibility, _panel_report = build_run_company_panel(inputs)
    usable_ids = usable_run_ids(inputs.raw_runs)
    run_meta_by_id = {
        r["run_id"]: {
            "paraphrase_id": r["paraphrase_id"],
            "day_index": r["day_index"],
            "rep_index": r["rep_index"],
        }
        for r in inputs.raw_runs
    }

    cells: list[DailyPanelCell] = []
    engines_seen: set[str] = set()
    for row in rows:
        if row.run_id not in usable_ids:
            continue
        if row.intent_id not in UNBRANDED_INTENT_IDS:
            continue
        if row.entity_id not in eligibility.qualifying_by_engine.get(row.engine, set()):
            continue
        meta = run_meta_by_id.get(row.run_id)
        if meta is None:
            continue
        engines_seen.add(row.engine)
        cells.append(
            DailyPanelCell(
                entity_id=row.entity_id,
                engine=row.engine,
                intent_id=row.intent_id,
                paraphrase_id=meta["paraphrase_id"],
                day_index=meta["day_index"],
                rep_index=meta["rep_index"],
                run_id=row.run_id,
                mentioned=row.mentioned,
            )
        )

    # 2026-09-26 review: confirmed (not just readable) that every consumer
    # of this panel (V2/A2/O1/O2) only ever sees unbranded-intent cells.
    assert all(c.intent_id in UNBRANDED_INTENT_IDS for c in cells), "build_daily_panel leaked a branded-intent cell"

    # 2026-09-29 freeze: a fixed order. Upstream (B1) builds rows by
    # iterating sets of strings, whose order changes with PYTHONHASHSEED;
    # every seeded resample downstream (V2/A2/O1 nulls and bootstraps)
    # consumes its random stream in this order, so an unsorted panel made
    # the figures differ between otherwise identical builds.
    cells.sort(key=lambda c: (c.engine, c.entity_id, c.intent_id, c.paraphrase_id, c.day_index, c.rep_index, c.run_id))

    report = DailyPanelBuildReport(
        cells_produced=len(cells),
        qualifying_by_engine={engine: eligibility.qualifying_count(engine) for engine in sorted(engines_seen)},
        completeness=check_daily_completeness(inputs.raw_runs),
    )
    return cells, report


def qualifying_entities_by_engine(cells: list[DailyPanelCell]) -> dict[str, set[str]]:
    """The set of entity_ids actually present in the daily panel for each
    engine. Since build_daily_panel already restricts to qualifying
    entities, this IS that engine's qualifying set as it manifests in the
    panel -- shared by A2 and O1, which both need the per-engine company
    population itself, not just its count (rates_panel.RatesPanelBuildReport.
    qualifying_by_engine only carries the count)."""
    result: dict[str, set[str]] = defaultdict(set)
    for cell in cells:
        result[cell.engine].add(cell.entity_id)
    return dict(result)


def company_day_rates(cells: list[DailyPanelCell]) -> dict[tuple[str, str, int], tuple[int, int]]:
    """(entity_id, engine, day_index) -> (n_mentioned, n_total), pooled
    across every unbranded intent, paraphrase, and usable rep for that
    company/engine/day -- the single company-level day-rate A2 and O1 both
    read (EXPERIMENT.md A2/O1, "company level, pooled across intents
    1-6"). Shared here rather than duplicated in both figure modules."""
    counts: dict[tuple[str, str, int], list[int]] = defaultdict(lambda: [0, 0])
    for cell in cells:
        key = (cell.entity_id, cell.engine, cell.day_index)
        counts[key][1] += 1
        if cell.mentioned:
            counts[key][0] += 1
    return {key: (v[0], v[1]) for key, v in counts.items()}
