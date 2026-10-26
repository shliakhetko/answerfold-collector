#!/usr/bin/env python3
"""CLI entry point: turn a coding-pipeline LLM cache directory into a dollar
cost report. Pure local file reads -- no API calls, safe to run any time.

Every A1/A3/A4/A5 call already writes its token usage into its cache entry
(analysis/coding/cache.py::LLMCache) via each stage's tokens_input/
tokens_output field, but nothing converts that into a dollar figure. This
walks every cached *.json file under --cache-dir, groups by stage (the
subdirectory name, e.g. "a1_extractors") and by model_returned, and prices
each entry via analysis.aggregate.pricing.compute_cost_usd -- the same
pricing table/logic used for the collector's own runs, extended (see that
module) to cover the coding pipeline's cheap extractor/judge models.

Usage:
    uv run python -m scripts.coding.report_coding_pipeline_cost \\
        --cache-dir data/_trial/coded/llm_cache \\
        --output data/_trial/coding_trial_baseline_cost.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from analysis.aggregate.pricing import compute_cost_usd
from collector import paths

DEFAULT_CACHE_DIR = paths.LLM_CACHE_DIR


@dataclass
class ModelTotals:
    model: str
    n_calls: int = 0
    n_priced: int = 0  # calls where a rate was found (model in PRICING)
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    cost_usd_if_batch: float = 0.0


@dataclass
class StageTotals:
    stage: str
    n_calls: int = 0
    n_errors: int = 0
    by_model: dict[str, ModelTotals] = field(default_factory=dict)

    def cost_usd(self) -> float:
        return sum(m.cost_usd for m in self.by_model.values())

    def cost_usd_if_batch(self) -> float:
        return sum(m.cost_usd_if_batch for m in self.by_model.values())


@dataclass
class CostReport:
    cache_dir: str
    by_stage: dict[str, StageTotals] = field(default_factory=dict)

    def total_cost_usd(self) -> float:
        return sum(s.cost_usd() for s in self.by_stage.values())

    def total_cost_usd_if_batch(self) -> float:
        return sum(s.cost_usd_if_batch() for s in self.by_stage.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_dir": self.cache_dir,
            "total_cost_usd": round(self.total_cost_usd(), 4),
            "total_cost_usd_if_batch": round(self.total_cost_usd_if_batch(), 4),
            "by_stage": {
                stage: {
                    "n_calls": totals.n_calls,
                    "n_errors": totals.n_errors,
                    "cost_usd": round(totals.cost_usd(), 4),
                    "cost_usd_if_batch": round(totals.cost_usd_if_batch(), 4),
                    "by_model": {
                        model: {
                            "n_calls": mt.n_calls,
                            "n_priced": mt.n_priced,
                            "tokens_input": mt.tokens_input,
                            "tokens_output": mt.tokens_output,
                            "cost_usd": round(mt.cost_usd, 4),
                            "cost_usd_if_batch": round(mt.cost_usd_if_batch, 4),
                        }
                        for model, mt in sorted(totals.by_model.items())
                    },
                }
                for stage, totals in sorted(self.by_stage.items())
            },
        }


def build_report(cache_dir: Path) -> CostReport:
    """Every cache subdirectory directly under cache_dir is one stage (e.g.
    a1_extractors/, a3_entity_resolution/, a1_arbiter/) -- every *.json file
    under it (searched recursively, so nested layouts still work) is one
    cached LLM call. A stage directory that doesn't exist yet (e.g.
    a1_arbiter/ before Step 3 lands) is simply absent from the report, not
    an error."""
    report = CostReport(cache_dir=str(cache_dir))
    if not cache_dir.is_dir():
        return report

    for stage_dir in sorted(p for p in cache_dir.iterdir() if p.is_dir()):
        stage = StageTotals(stage=stage_dir.name)
        for path in sorted(stage_dir.rglob("*.json")):
            entry = json.loads(path.read_text(encoding="utf-8"))
            stage.n_calls += 1
            if entry.get("error"):
                stage.n_errors += 1

            model = entry.get("model_returned") or "unknown"
            mt = stage.by_model.setdefault(model, ModelTotals(model=model))
            mt.n_calls += 1

            tokens_input = entry.get("tokens_input")
            tokens_output = entry.get("tokens_output")
            if tokens_input is not None:
                mt.tokens_input += tokens_input
            if tokens_output is not None:
                mt.tokens_output += tokens_output

            cost = compute_cost_usd(model, tokens_input, tokens_output)
            cost_batch = compute_cost_usd(model, tokens_input, tokens_output, batch=True)
            if cost is not None:
                mt.n_priced += 1
                mt.cost_usd += cost
                mt.cost_usd_if_batch += cost_batch  # cost is not None => cost_batch isn't either

        report.by_stage[stage.stage] = stage
    return report


def _print_summary(report: CostReport) -> None:
    print(f"cache dir: {report.cache_dir}")
    for stage, totals in sorted(report.by_stage.items()):
        print(f"  {stage}: {totals.n_calls} calls ({totals.n_errors} errored), ${totals.cost_usd():.4f}")
        for model, mt in sorted(totals.by_model.items()):
            unpriced = f", {mt.n_calls - mt.n_priced} unpriced" if mt.n_priced < mt.n_calls else ""
            print(
                f"    {model}: {mt.n_calls} calls, "
                f"{mt.tokens_input} in / {mt.tokens_output} out tokens, "
                f"${mt.cost_usd:.4f} (batch would be ${mt.cost_usd_if_batch:.4f}){unpriced}"
            )
    print(f"TOTAL: ${report.total_cost_usd():.4f} (batch would be ${report.total_cost_usd_if_batch():.4f})")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=None, help="write the full JSON report here, if given")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(args.cache_dir)
    _print_summary(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
