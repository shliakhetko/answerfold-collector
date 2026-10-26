"""Every registered figure (S4, the Claude probation record and the UI
calibration included) serializes the full envelope: value, denominator
before and after, n excluded, threshold or null, verdict or null. Enforced
both on the empty-input (untestable) path and on the real figures.json
when it is present."""
import json
from pathlib import Path

from collector import paths

import pytest

from analysis.phase_b.figures import FIGURES
from analysis.phase_b.reading import FrozenInputs

REQUIRED_KEYS = (
    "value", "denominator_before_exclusion", "denominator_after_exclusion", "n_excluded",
    "n_included", "threshold", "verdict", "status", "reason_untestable",
)


def _empty():
    return FrozenInputs(
        entities=[], alt_domains_rows=[], entity_verification_rows=[], mentions_raw=[], raw_runs=[], sources_by_run={},
    )


def test_probation_s4_and_ui_calibration_are_registered():
    for fid in ("S4", "CLAUDE_PROBATION", "UI_CALIBRATION"):
        assert fid in FIGURES


@pytest.mark.parametrize("figure_id", sorted(FIGURES))
def test_every_registered_figure_serializes_the_full_envelope_on_empty_input(figure_id):
    record = FIGURES[figure_id](_empty()).to_json()
    for key in REQUIRED_KEYS:
        assert key in record, f"{figure_id} omits {key}"
    if record["status"] == "untestable":
        assert record["reason_untestable"]


def test_real_figures_file_has_full_envelope_and_reasons_for_exclusions():
    path = paths.FIGURES_JSON
    if not path.exists():
        pytest.skip("no figures.json built in this checkout")
    figures = json.loads(path.read_text())["figures"]
    assert set(figures) == set(FIGURES)
    for fid, record in figures.items():
        for key in REQUIRED_KEYS:
            assert key in record, f"{fid} omits {key}"
        if record["status"] == "ok":
            assert isinstance(record["n_excluded"], (dict, int)), f"{fid}: n_excluded must state what was excluded and why"
