"""The seam: the one serialized file computing hands to displaying.
Display reads this file and nothing else -- no live DB, no frozen
artifact, no raw row.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analysis.phase_b.figure_shape import FigureResult


def write_seam_file(
    path: Path,
    *,
    inputs_manifest: dict[str, Any],
    figure_results: dict[str, FigureResult],
    intermediates: dict[str, Any] | None = None,
) -> None:
    """Byte-deterministic (2026-09-29 freeze): the whole file is a pure
    function of the frozen inputs (sorted keys, no wall-clock or random
    value anywhere). The one non-deterministic fact, `generated_at`, was
    moved out to a sidecar `<name>.meta.json` next to the seam file so
    the sha256 of the seam file itself is the reproducibility claim
    (ledger: freeze-two-builds-identical-hashes-2026-09-29).

    `intermediates`: summaries of shared intermediates the compute job
    already built (e.g. row counts, null counts) -- never the raw
    intermediate rows themselves. This is the only way the author display
    surface can show anything about an intermediate, since it reads
    nothing but this file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seam = {
        "generator": "scripts/results/run_phase_b.py",
        "inputs": inputs_manifest,
        "figures": {figure_id: result.to_json() for figure_id, result in sorted(figure_results.items())},
        "intermediates": intermediates or {},
    }
    path.write_text(json.dumps(seam, indent=2, sort_keys=True), encoding="utf-8")
    path.with_suffix(".meta.json").write_text(
        json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(), "figures_file": path.name}, indent=2),
        encoding="utf-8",
    )
