"""The intent branded/unbranded map, read from config/prompts/prompts.csv.

analysis.phase_b.citation_panel.UNBRANDED_INTENT_IDS is a hardcoded literal
that S1/S2/K2/X1 already ship on -- correct today, but a bare literal with
no link back to the prompt set it's supposed to describe. This module reads
the real source of truth (prompts.csv's own `branded` column) and asserts
it agrees with that literal, so a future prompt-set edit that changes an
intent's branded status gets caught loudly at the top of a Phase B run,
never silently ignored by five different hardcoded copies of the same
number.

scripts/results/run_phase_b.py calls read_intent_map() + assert_intent_map() once,
first, before building the rates panel or any figure, and prints the
resolved map -- the "print the intent map at the top of the run" step. New
Block-A modules (rates_panel, A1, V1, K1, A3) still import
UNBRANDED_INTENT_IDS from citation_panel (consistent with the rest of the
pipeline), but only ever run downstream of that assertion in the real
entry point -- this module is what makes that constant a verified fact
about the current prompt set rather than an unverified guess.
"""
from __future__ import annotations

import csv
from pathlib import Path

from collector import paths

PROMPTS_CSV_PATH = paths.PROMPTS_CSV


class IntentMapError(Exception):
    """prompts.csv is missing, malformed, has inconsistent branded flags
    within one intent, or its resolved unbranded/branded split disagrees
    with the pipeline's existing UNBRANDED_INTENT_IDS constant."""


def read_intent_map(repo_root: Path) -> dict[int, bool]:
    """intent_id -> branded (bool), one entry per distinct intent_id in
    prompts.csv. Raises IntentMapError if a single intent mixes branded
    and unbranded rows across its paraphrases, or if the file is missing."""
    path = Path(repo_root) / PROMPTS_CSV_PATH
    if not path.exists():
        raise IntentMapError(f"prompts.csv not found at {path}")

    branded_values_by_intent: dict[int, set[bool]] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            intent_id = int(row["intent_id"])
            branded = bool(int(row["branded"]))
            branded_values_by_intent.setdefault(intent_id, set()).add(branded)

    intent_map: dict[int, bool] = {}
    for intent_id, branded_values in sorted(branded_values_by_intent.items()):
        if len(branded_values) != 1:
            raise IntentMapError(
                f"intent {intent_id} has inconsistent branded flags across its paraphrases in "
                f"prompts.csv: {sorted(branded_values)}"
            )
        intent_map[intent_id] = branded_values.pop()
    return intent_map


def unbranded_intent_ids(intent_map: dict[int, bool]) -> frozenset[int]:
    return frozenset(intent_id for intent_id, branded in intent_map.items() if not branded)


def branded_intent_ids(intent_map: dict[int, bool]) -> frozenset[int]:
    return frozenset(intent_id for intent_id, branded in intent_map.items() if branded)


def assert_intent_map(intent_map: dict[int, bool]) -> None:
    """Cross-checks the CSV-derived map against
    analysis.phase_b.citation_panel.UNBRANDED_INTENT_IDS -- the literal
    every already-shipped figure (S1/S2/K2/X1) is built on. If prompts.csv
    and that literal ever disagree, every figure in this pipeline would
    silently be scoped to different intents depending which module you
    read; this fails loudly instead, before any figure computes anything."""
    from analysis.phase_b.citation_panel import UNBRANDED_INTENT_IDS

    unbranded = unbranded_intent_ids(intent_map)
    if unbranded != UNBRANDED_INTENT_IDS:
        raise IntentMapError(
            f"prompts.csv unbranded intents {sorted(unbranded)} do not match the pipeline's existing "
            f"UNBRANDED_INTENT_IDS {sorted(UNBRANDED_INTENT_IDS)} (analysis/phase_b/citation_panel.py)"
        )
