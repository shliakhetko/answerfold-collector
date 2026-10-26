"""The standard post-window auxiliary record envelope.

Every JSONL row and every `.meta.json` sidecar this batch writes carries
these fields, so a Phase-A reader can tell auxiliary data apart from window
data without consulting a manifest, and knows exactly how stale it is:

    collected_at              ISO-8601, UTC
    collected_at_kyiv         ISO-8601, Europe/Kyiv (the study's operating tz)
    post_window               always True for this batch
    days_after_window_close   integer, (collected date - 2026-08-26)
    collector_version         "geo-collector/<pkg version>+<git short sha>"
    collector                 the collector name (e.g. "collect_cited_urls")
    raw                       the full, unmodified response body / payload

Analysis-layer code (INSTRUMENT.md section 12): imports from collector/ only.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

from collector.runner.execute import InstrumentCommitError, get_instrument_commit

# The seven-day window's last collection day (day_index 7). Auxiliary data
# collected after this reflects site state on the collection date, not during
# the window -- see reports/ambiguity_ledger.md.
WINDOW_CLOSE_DATE = date(2026, 8, 26)

# Europe/Kyiv is UTC+3 (EEST) for the whole study period (Aug-Sep 2026);
# stdlib has no tz database guarantee on every platform, so the fixed offset
# is stated once here rather than depending on zoneinfo.
KYIV_TZ = timezone(timedelta(hours=3), name="Europe/Kyiv")

# Keep in sync with pyproject.toml [project].version. This repo sets
# `package = false`, so there is no installed distribution to read a version
# from via importlib.metadata.
PACKAGE_VERSION = "0.1.0"


def collector_version() -> str:
    try:
        sha = get_instrument_commit()[:12]
    except InstrumentCommitError:
        sha = "unknown"
    return f"geo-collector/{PACKAGE_VERSION}+{sha}"


def days_after_window_close(collected: datetime | None = None) -> int:
    """Computed, not hard-coded, so a re-run on a later date stays correct.
    It is 3 for a run on 2026-08-29."""
    when = (collected or datetime.now(timezone.utc)).astimezone(timezone.utc).date()
    return (when - WINDOW_CLOSE_DATE).days


def make_envelope(collector: str, *, now: datetime | None = None) -> dict[str, Any]:
    """The envelope fields for one record, minus `raw` and the collector's
    own task-specific fields, which the caller merges in."""
    when = now or datetime.now(timezone.utc)
    when_utc = when.astimezone(timezone.utc)
    return {
        "collected_at": when_utc.isoformat(),
        "collected_at_kyiv": when_utc.astimezone(KYIV_TZ).isoformat(),
        "post_window": True,
        "days_after_window_close": days_after_window_close(when_utc),
        "collector_version": collector_version(),
        "collector": collector,
    }


def wrap(collector: str, fields: dict[str, Any], raw: Any, *, now: datetime | None = None) -> dict[str, Any]:
    """One complete record: envelope + task-specific `fields` + verbatim `raw`."""
    record = make_envelope(collector, now=now)
    record.update(fields)
    record["raw"] = raw
    return record
