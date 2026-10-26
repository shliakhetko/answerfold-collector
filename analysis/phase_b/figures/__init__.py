"""Registered-figure lookup. One unit of compute per figure_id, addressable
by that identifier; figures never call each other -- any relationship
between figures (e.g. "S2 needs S3 read first to make sense") belongs to
the display layer, not here.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable

from analysis.phase_b.figure_shape import FigureResult
from analysis.phase_b.figures import a1, a2, a3, claude_probation, k1, k2, o1, o2, s1, s2, s3, s4, ui_calibration, v1, v2, x1
from analysis.phase_b.reading import FrozenInputs

# Which of a figure's own notes/value keys IS its verdict ("verdict or null"
# on the envelope). A figure without a pre-registered verdict is absent here
# and carries an explicit None. Exploratory labels (O1, O2, probation) are
# included because they are the figure's own stated reading.
VERDICT_SOURCES: dict[str, tuple[str, tuple[str, ...]]] = {
    "S1": ("notes", ("overall_primary_pass", "primary_test_pass_by_engine", "secondary_verdict_by_engine")),
    "A1": ("notes", ("verdict_by_engine",)),
    "V1": ("notes", ("verdict",)),
    "V2": ("notes", ("verdict_by_engine", "noise_comparison_by_engine")),
    "A2": ("notes", ("verdict", "ci_excludes_zero_pairs")),
    "O1": ("notes", ("interpretation_band",)),
    "O2": ("notes", ("direction",)),
    "CLAUDE_PROBATION": ("value", ("verdict", "failed_conditions")),
}


def _with_verdict(figure_id: str, compute: Callable[[FrozenInputs], FigureResult]) -> Callable[[FrozenInputs], FigureResult]:
    def wrapped(inputs: FrozenInputs) -> FigureResult:
        result = compute(inputs)
        source = VERDICT_SOURCES.get(figure_id)
        if source is None or result.status != "ok":
            return result
        where, keys = source
        container = result.notes if where == "notes" else (result.value or {})
        verdict = {k: container.get(k) for k in keys}
        return replace(result, verdict=verdict)

    wrapped.__name__ = getattr(compute, "__name__", "compute")
    return wrapped


_RAW_FIGURES: dict[str, Callable[[FrozenInputs], FigureResult]] = {
    s1.FIGURE_ID: s1.compute,
    s2.FIGURE_ID: s2.compute,
    k2.FIGURE_ID: k2.compute,
    s3.FIGURE_ID: s3.compute,
    s4.FIGURE_ID: s4.compute,
    x1.FIGURE_ID: x1.compute,
    a1.FIGURE_ID: a1.compute,
    v1.FIGURE_ID: v1.compute,
    k1.FIGURE_ID: k1.compute,
    a3.FIGURE_ID: a3.compute,
    v2.FIGURE_ID: v2.compute,
    a2.FIGURE_ID: a2.compute,
    o1.FIGURE_ID: o1.compute,
    o2.FIGURE_ID: o2.compute,
    claude_probation.FIGURE_ID: claude_probation.compute,
    ui_calibration.FIGURE_ID: ui_calibration.compute,
}

FIGURES: dict[str, Callable[[FrozenInputs], FigureResult]] = {
    figure_id: _with_verdict(figure_id, compute) for figure_id, compute in _RAW_FIGURES.items()
}
