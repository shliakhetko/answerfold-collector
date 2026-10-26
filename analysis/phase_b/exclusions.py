"""The one place "usable response" is decided. Any figure or intermediate
that needs to know whether a raw run counts as usable calls
exclusion_reason() -- never its own copy of this logic. Reused by
figures/s3.py and domain_ownership.py so "how many responses do we
actually have usable data for" means the same number everywhere;
inventing a second version of this check anywhere else is exactly the
"join written twice" this module exists to prevent.
"""
from __future__ import annotations

from typing import Any

EXCLUDE_MANUAL = "manual_ui_calibration"
EXCLUDE_ERROR = "collection_error"
EXCLUDE_NO_TEXT = "no_response_text"


def exclusion_reason(run: dict[str, Any]) -> str | None:
    """None means usable. A failed call is not a real zero (pinned
    EXPERIMENT.md, Notion Part 1.7) -- errored and manual-UI-calibration
    runs are excluded before anything downstream treats them as data."""
    if run.get("is_manual"):
        return EXCLUDE_MANUAL
    if run.get("error") is not None:
        return EXCLUDE_ERROR
    if not run.get("raw_response_text"):
        return EXCLUDE_NO_TEXT
    return None


def usable_run_ids(raw_runs: list[dict[str, Any]]) -> set[str]:
    """The canonical "usable responses" set -- the one number
    total_responses_available and every per-engine denominator are meant
    to agree on."""
    return {run["run_id"] for run in raw_runs if exclusion_reason(run) is None}
