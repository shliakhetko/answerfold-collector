"""B4: figures (Post-Collection Coding Pipeline Spec). No model.

"Formulas as written in the frozen doc, not restated here" -- that frozen
doc doesn't exist in this repo (config/figure_formulas.md is a stub).
compute_figure is wired to every figure ID and its confirmatory/table/
exploratory tier, but raises NotImplementedError until a real formula
replaces the stub for that figure -- this stage stays a skeleton on
purpose, not a guess at what the real study would report.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analysis.coding.b1_per_run_company import PerRunCompanyRow
from analysis.coding.b2_per_run_rollups import PerRunRollupRow
from analysis.coding.b3_aggregates import CellRates, EligibilityReport

from collector import paths
DEFAULT_FORMULAS_PATH = paths.FIGURE_FORMULAS_MD

# Note: figure ID "A1" here is unrelated to pipeline stage A1 (answer
# coding) -- same label from the frozen doc's own numbering, different
# thing. Same note appears in b3b_serp_derivation.py.
CONFIRMATORY_FIGURE_IDS = ("S1", "A1", "V1")
TABLE_FIGURE_IDS = ("S2", "S3", "K1", "K2", "O1", "V2", "A2")
EXPLORATORY_FIGURE_IDS = ("A3", "O2", "X1")
FIGURE_IDS = CONFIRMATORY_FIGURE_IDS + TABLE_FIGURE_IDS + EXPLORATORY_FIGURE_IDS


def figure_tier(figure_id: str) -> str:
    if figure_id in CONFIRMATORY_FIGURE_IDS:
        return "confirmatory"
    if figure_id in TABLE_FIGURE_IDS:
        return "table"
    if figure_id in EXPLORATORY_FIGURE_IDS:
        return "exploratory"
    raise KeyError(f"{figure_id!r} is not one of the spec's figure IDs: {FIGURE_IDS}")


@dataclass(frozen=True)
class AggregatesBundle:
    """Everything B4 could read to compute a figure, once formulas exist."""

    per_run_company_rows: list[PerRunCompanyRow]
    per_run_rollup_rows: list[PerRunRollupRow]
    cell_rates: list[CellRates]
    eligibility: EligibilityReport


def compute_figure(figure_id: str, bundle: AggregatesBundle, formulas_path: Path = DEFAULT_FORMULAS_PATH) -> Any:
    tier = figure_tier(figure_id)  # raises KeyError on an unknown figure_id
    raise NotImplementedError(
        f"{figure_id} ({tier}): formula not yet frozen -- see {formulas_path}. "
        "compute_figure is wired but deliberately unimplemented until that file "
        "has real content (spec: 'Do not compute any figure inside Phase A' and "
        "'Formulas as written in the frozen doc, not restated here')."
    )
