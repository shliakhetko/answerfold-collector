"""Loads a prompts CSV as plain data -- there is no prompts DB table.
prompt_text is denormalized directly onto each run row instead (INSTRUMENT.md
section 8.1: "joining to a prompt table would make historical rows depend on
a table that might later be edited").

CSV columns:
  intent_id
  paraphrase_id
  prompt_text
  branded            "1"/"true"/"yes" if this intent names a vendor
                      (section 5: branded intents are reported separately
                      and must never enter a pooled unbranded metric);
                      anything else, including a blank cell, means unbranded
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

_TRUE_VALUES = {"1", "true", "yes"}


@dataclass(frozen=True)
class Prompt:
    intent_id: int
    paraphrase_id: int
    prompt_text: str
    branded: bool


def load_prompts_csv(path: str | Path) -> list[Prompt]:
    prompts: list[Prompt] = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            prompts.append(
                Prompt(
                    intent_id=int(row["intent_id"]),
                    paraphrase_id=int(row["paraphrase_id"]),
                    prompt_text=row["prompt_text"],
                    branded=(row.get("branded") or "").strip().lower() in _TRUE_VALUES,
                )
            )
    return prompts
