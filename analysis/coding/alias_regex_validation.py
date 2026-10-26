"""Parallel: alias regex validation (Post-Collection Coding Pipeline Spec).
Runs the frozen alias list (config/alias_list.csv, currently a stub) as
regex over the same runs, independently of Phase A -- "the answer to the
obvious critique that a model is grading model output."
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from collector import paths
DEFAULT_ALIAS_LIST_PATH = paths.ALIAS_LIST_CSV


def load_alias_list(alias_list_csv: Path = DEFAULT_ALIAS_LIST_PATH) -> list[tuple[str, re.Pattern[str]]]:
    with alias_list_csv.open(encoding="utf-8") as f:
        return [
            (row["canonical_name"], re.compile(row["alias_regex"]))
            for row in csv.DictReader(filter(lambda line: not line.startswith("#"), f))
        ]


def run_alias_regex(
    run_rows: list[dict[str, Any]], alias_list_csv: Path = DEFAULT_ALIAS_LIST_PATH
) -> list[dict[str, str]]:
    """One hit per (run_id, canonical_name) where that alias's regex matches
    the run's raw_response_text -- real regex logic, no model."""
    aliases = load_alias_list(alias_list_csv)
    hits: list[dict[str, str]] = []
    for run in run_rows:
        text = run.get("raw_response_text") or ""
        if not text:
            continue
        for canonical_name, pattern in aliases:
            if pattern.search(text):
                hits.append({"run_id": run["run_id"], "canonical_name": canonical_name})
    return hits


@dataclass(frozen=True)
class ComparisonReport:
    ensemble_recall_on_seed_list: float | None
    regex_found_ensemble_missed: list[tuple[str, str]] = field(default_factory=list)
    ensemble_found_regex_missed: list[tuple[str, str]] = field(default_factory=list)


def _surface_form_to_canonical_name(entity_registry_rows: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in entity_registry_rows:
        for surface_form in (row.get("surface_forms") or "").split(";"):
            if surface_form:
                mapping[surface_form] = row["canonical_name"]
    return mapping


def compare_to_ensemble(
    alias_hits: list[dict[str, str]],
    mentions_raw_rows: list[dict[str, Any]],
    entity_registry_rows: list[dict[str, Any]],
    seed_list: set[str],
) -> ComparisonReport:
    """seed_list: lowercased canonical_name/homepage_domain values, e.g. from
    analysis.coding.a4_entity_classification.load_seed_list. "The seed list
    is a recall yardstick now, not the population" (spec) -- regex hits on
    seed-list companies stand in for a recall denominator since the seed
    list is the one part of this pipeline with an independent ground truth."""
    regex_set = {(h["run_id"], h["canonical_name"]) for h in alias_hits}

    sf_to_canonical = _surface_form_to_canonical_name(entity_registry_rows)
    ensemble_set = {
        (m["run_id"], sf_to_canonical[m["surface_form"]])
        for m in mentions_raw_rows
        if m["surface_form"] in sf_to_canonical
    }

    seed_pairs = {(run_id, name) for run_id, name in regex_set if name.lower() in seed_list}
    recall = (len(seed_pairs & ensemble_set) / len(seed_pairs)) if seed_pairs else None

    return ComparisonReport(
        ensemble_recall_on_seed_list=recall,
        regex_found_ensemble_missed=sorted(regex_set - ensemble_set),
        ensemble_found_regex_missed=sorted(ensemble_set - regex_set),
    )
