"""The one result shape every registered figure returns.

A bare number is not a figure: every figure carries its denominator, how
many units entered it, how many were excluded and why, and the
pre-registered threshold or falsifier it is judged against (explicit
`None` where none exists -- never an omitted key). This module has no
opinion about any particular figure's formula; it only defines the
envelope every figure_id's compute function must fill in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FigureResult:
    """value/denominator_before_exclusion/denominator_after_exclusion/
    n_included/n_excluded/threshold are deliberately `Any`: a figure
    computed "per engine" carries a dict keyed by engine for each of them
    (see figures/s3.py); a pooled figure would carry a plain number.
    Whatever the shape, the same fields travel together -- that pairing
    is the whole point of the figure envelope.

    Convention: `denominator_after_exclusion` is always the true divisor
    `value` was computed over -- same number as `n_included`, by
    construction, for every figure that excludes units before computing
    (S3: errored/manual/no-text runs excluded). `denominator_before_exclusion`
    is the full pool before that exclusion was applied (usable +
    excluded) -- carried explicitly, not left for a reader to reconstruct
    by summing `n_excluded`'s reasons themselves. A figure must never
    report a `denominator_after_exclusion` a reader could not reproduce
    `value` from. `n_excluded` records what was dropped and why for the
    audit trail. The author display surface checks
    `n_included == denominator_after_exclusion`: the one invariant that
    should always hold and that would catch a figure that quietly
    violates this convention.

    status: "ok" or "untestable". When "untestable", value/both
    denominators/n_included/n_excluded/threshold are None and
    reason_untestable names why -- never a silently-omitted key, never a
    figure computed on a degraded denominator without saying so.
    """

    figure_id: str
    status: str  # "ok" | "untestable"
    value: Any
    denominator_before_exclusion: Any
    denominator_after_exclusion: Any
    n_included: Any
    n_excluded: Any
    threshold: Any
    reason_untestable: str | None
    supporting_table: list[dict[str, Any]] = field(default_factory=list)
    # Computed facts specific to this figure that don't fit the core
    # numeric slots above (e.g. "these two columns turned out to be the
    # same set, per engine") -- stated here as a fact on the record, not
    # left for a reader to infer by comparing columns by eye. Optional,
    # empty for figures with nothing to add.
    notes: dict[str, Any] = field(default_factory=dict)
    # The figure's verdict against its pre-registered threshold/falsifier,
    # or an explicit None when the figure has none (tables, exploratory
    # figures). Filled from the figure's own notes/value by
    # analysis.phase_b.figures.__init__ (one registry of which keys are the
    # verdict), so "verdict or null" is a key every serialized figure
    # carries -- never an omitted one.
    verdict: Any = None

    def __post_init__(self) -> None:
        if self.status not in ("ok", "untestable"):
            raise ValueError(f"FigureResult.status must be 'ok' or 'untestable', got {self.status!r}")
        if self.status == "untestable" and not self.reason_untestable:
            raise ValueError("an untestable FigureResult must carry a reason_untestable")

    def to_json(self) -> dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "status": self.status,
            "value": self.value,
            "denominator_before_exclusion": self.denominator_before_exclusion,
            "denominator_after_exclusion": self.denominator_after_exclusion,
            "n_included": self.n_included,
            "n_excluded": self.n_excluded,
            "threshold": self.threshold,
            "reason_untestable": self.reason_untestable,
            "supporting_table": self.supporting_table,
            "notes": self.notes,
            "verdict": self.verdict,
        }
