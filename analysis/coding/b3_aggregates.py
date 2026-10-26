"""B3: aggregates, per company x intent x engine (Post-Collection Coding
Pipeline Spec). No model.

apply_eligibility_floor and the qualifying-count report are spec-precise:
"5 mentions minimum across the six unbranded intents within that engine.
Report qualifying count per engine before any other number." Everything
past that (mention_rate/recommendation_rate/variance/spread/top-quartile/
top-ten-domains) is a first-pass definition -- the real formulas live in
the frozen doc this repo doesn't have yet (config/figure_formulas.md is a
stub); B4 is what actually reports study figures, and only once that file
has real content. Treat every function below as provisional plumbing, not
a frozen statistical definition.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from analysis.coding.b1_per_run_company import PerRunCompanyRow, _domain_of

ELIGIBILITY_FLOOR = 5  # mentions, across unbranded intents, within an engine (spec)


@dataclass(frozen=True)
class EligibilityReport:
    qualifying_by_engine: dict[str, set[str]] = field(default_factory=dict)

    def qualifying_count(self, engine: str) -> int:
        return len(self.qualifying_by_engine.get(engine, set()))


def apply_eligibility_floor(
    per_run_company_rows: list[PerRunCompanyRow],
    run_meta_by_id: dict[str, dict[str, Any]],
    unbranded_intent_ids: set[int],
    *,
    floor: int = ELIGIBILITY_FLOOR,
) -> EligibilityReport:
    """run_meta_by_id: run_id -> {"engine": str, "intent_id": int, ...}."""
    counts: dict[tuple[str, str], int] = {}
    for row in per_run_company_rows:
        if not row.mentioned:
            continue
        meta = run_meta_by_id.get(row.run_id)
        if meta is None or meta["intent_id"] not in unbranded_intent_ids:
            continue
        key = (meta["engine"], row.entity_id)
        counts[key] = counts.get(key, 0) + 1

    qualifying_by_engine: dict[str, set[str]] = {}
    for (engine, entity_id), count in counts.items():
        if count >= floor:
            qualifying_by_engine.setdefault(engine, set()).add(entity_id)
    return EligibilityReport(qualifying_by_engine=qualifying_by_engine)


@dataclass(frozen=True)
class CellRates:
    entity_id: str
    intent_id: int
    engine: str
    n_runs: int
    mention_rate: float
    recommendation_rate: float


def build_cell_rates(
    per_run_company_rows: list[PerRunCompanyRow],
    run_meta_by_id: dict[str, dict[str, Any]],
    eligibility: EligibilityReport,
    *,
    restrict_to_eligible: bool = True,
) -> list[CellRates]:
    """One row per (entity, intent, engine) cell. By default restricted to
    entities that cleared apply_eligibility_floor for that engine (spec's
    floor gates *entry* into per-entity stats, not just display).

    restrict_to_eligible=False keeps every in-frame entity's cells --
    including zero-mention ones (per_run_company_rows is already the full
    run x in-frame-entity cross product, so mention_rate is well defined
    even at 0). Use it to inspect the full grid before committing to the
    spec's floor."""
    groups: dict[tuple[str, int, str], list[PerRunCompanyRow]] = {}
    for row in per_run_company_rows:
        meta = run_meta_by_id.get(row.run_id)
        if meta is None:
            continue
        engine = meta["engine"]
        if restrict_to_eligible and row.entity_id not in eligibility.qualifying_by_engine.get(engine, set()):
            continue
        key = (row.entity_id, meta["intent_id"], engine)
        groups.setdefault(key, []).append(row)

    cells: list[CellRates] = []
    for (entity_id, intent_id, engine), rows in groups.items():
        n = len(rows)
        cells.append(
            CellRates(
                entity_id=entity_id,
                intent_id=intent_id,
                engine=engine,
                n_runs=n,
                mention_rate=sum(1 for r in rows if r.mentioned) / n,
                recommendation_rate=sum(1 for r in rows if r.recommended) / n,
            )
        )
    return cells


def within_day_variance(
    per_run_company_rows: list[PerRunCompanyRow],
    run_meta_by_id: dict[str, dict[str, Any]],
    entity_id: str,
    intent_id: int,
    engine: str,
) -> float | None:
    """Mean, across days, of the population variance of the mentioned
    indicator (0/1) across that day's reps -- "noise within a fixed
    day/hour" (INSTRUMENT.md section 4's reps exist to measure exactly
    this). None if there's fewer than 2 reps on every day (variance
    undefined)."""
    by_day: dict[int, list[int]] = {}
    for row in per_run_company_rows:
        if row.entity_id != entity_id:
            continue
        meta = run_meta_by_id.get(row.run_id)
        if meta is None or meta["intent_id"] != intent_id or meta["engine"] != engine:
            continue
        by_day.setdefault(meta["day_index"], []).append(1 if row.mentioned else 0)

    day_variances = [statistics.pvariance(values) for values in by_day.values() if len(values) >= 2]
    return statistics.fmean(day_variances) if day_variances else None


def day_to_day_variance(
    per_run_company_rows: list[PerRunCompanyRow],
    run_meta_by_id: dict[str, dict[str, Any]],
    entity_id: str,
    intent_id: int,
    engine: str,
) -> float | None:
    """Population variance, across days, of each day's mention rate (mean
    mentioned indicator for that day) -- "noise day-to-day" as opposed to
    within_day_variance's "noise within a day". None with fewer than 2
    days observed."""
    by_day: dict[int, list[int]] = {}
    for row in per_run_company_rows:
        if row.entity_id != entity_id:
            continue
        meta = run_meta_by_id.get(row.run_id)
        if meta is None or meta["intent_id"] != intent_id or meta["engine"] != engine:
            continue
        by_day.setdefault(meta["day_index"], []).append(1 if row.mentioned else 0)

    daily_rates = [statistics.fmean(values) for values in by_day.values() if values]
    return statistics.pvariance(daily_rates) if len(daily_rates) >= 2 else None


@dataclass(frozen=True)
class SpreadReport:
    engine_spread: float | None  # max - min of mention_rate across engines, for this entity
    intent_spread: float | None  # max - min of mention_rate across intents, for this entity


def vertical_vs_engine_spread(cells: list[CellRates], entity_id: str) -> SpreadReport:
    entity_cells = [c for c in cells if c.entity_id == entity_id]
    by_engine: dict[str, list[float]] = {}
    by_intent: dict[int, list[float]] = {}
    for c in entity_cells:
        by_engine.setdefault(c.engine, []).append(c.mention_rate)
        by_intent.setdefault(c.intent_id, []).append(c.mention_rate)

    engine_means = [statistics.fmean(v) for v in by_engine.values()]
    intent_means = [statistics.fmean(v) for v in by_intent.values()]
    return SpreadReport(
        engine_spread=(max(engine_means) - min(engine_means)) if len(engine_means) >= 2 else None,
        intent_spread=(max(intent_means) - min(intent_means)) if len(intent_means) >= 2 else None,
    )


def top_quartile_membership_per_day(
    per_run_company_rows: list[PerRunCompanyRow], run_meta_by_id: dict[str, dict[str, Any]]
) -> dict[tuple[int, str], bool]:
    """(day_index, entity_id) -> whether that entity's mention count that day
    was in the top quartile of all entities mentioned that day. A day with
    fewer than 4 distinct mentioned entities has no well-defined quartile
    cutoff and every entity present that day is reported False."""
    counts_by_day: dict[int, dict[str, int]] = {}
    for row in per_run_company_rows:
        if not row.mentioned:
            continue
        meta = run_meta_by_id.get(row.run_id)
        if meta is None:
            continue
        day_counts = counts_by_day.setdefault(meta["day_index"], {})
        day_counts[row.entity_id] = day_counts.get(row.entity_id, 0) + 1

    membership: dict[tuple[int, str], bool] = {}
    for day_index, day_counts in counts_by_day.items():
        values = sorted(day_counts.values())
        if len(values) < 4:
            for entity_id in day_counts:
                membership[(day_index, entity_id)] = False
            continue
        cutoff = statistics.quantiles(values, n=4)[-1]  # 75th percentile
        for entity_id, count in day_counts.items():
            membership[(day_index, entity_id)] = count >= cutoff
    return membership


def top_ten_domains_by_citation_share(sources_by_run: dict[str, list[dict[str, Any]]]) -> list[tuple[str, float]]:
    """Across every source in sources_by_run, the 10 most-cited normalized
    domains and each one's share of total citations. Empty list if there
    are no sources at all."""
    counts: dict[str, int] = {}
    total = 0
    for sources in sources_by_run.values():
        for source in sources:
            raw = source.get("domain") or source.get("url_raw", "")
            if not raw:
                continue
            domain = _domain_of(raw)
            counts[domain] = counts.get(domain, 0) + 1
            total += 1
    if total == 0:
        return []
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    return [(domain, count / total) for domain, count in ranked]
