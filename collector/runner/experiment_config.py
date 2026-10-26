"""Loads the experiment-declared engine/model/schedule config (EXPERIMENT.md's
conformance checklist, INSTRUMENT.md section 14) from a small YAML file.

This is deliberately NOT env vars: which model is pinned for which engine
cell is an experiment fact declared before collection begins, not a
deployment tunable (contrast with collector/config.py's OpenAIConfig, which
holds operational settings like timeout/retries).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class EngineCell:
    engine: str
    model_requested: str


@dataclass(frozen=True)
class ExperimentConfig:
    engines: list[EngineCell]
    reps_per_day: int
    days: int


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    data = yaml.safe_load(Path(path).read_text())
    engines = [
        EngineCell(engine=cell["engine"], model_requested=cell["model_requested"])
        for cell in data["engines"]
    ]
    return ExperimentConfig(
        engines=engines,
        reps_per_day=int(data["reps_per_day"]),
        days=int(data["days"]),
    )
