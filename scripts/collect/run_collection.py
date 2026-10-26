#!/usr/bin/env python3
"""CLI entry point: run one day's collection pass.

For every engine cell in the experiment config, every prompt in the prompt
set, and every repetition (INSTRUMENT.md section 4's
intents > paraphrases > repetitions > days structure, one day at a time),
calls that engine's adapter and stores the result via collector.runner.execute.

Loop shape: `for rep (pass): for prompt:`, not `for prompt: for rep:` --
repetitions of a prompt land ~1 pass apart (minutes) instead of seconds
apart, so a rep isn't just re-asking the same question back-to-back inside
one narrow window. Within a pass, prompt order is shuffled by a seeded RNG
(--prompt-order-seed), computed once and shared by every engine, so all
engines ask the same prompt at (nearly) the same time.

Concurrency: one worker thread per engine cell, synchronized by a single
`threading.Barrier` sized to the number of cells. All engines fire the same
(prompt, rep) concurrently, then every thread blocks at the barrier until
all engines have written a row for that (prompt, rep) -- successful, errored,
or timed out -- before any of them moves on to the next one. This bounds
wall time to roughly the slowest engine per (prompt, rep) instead of the sum
of every engine's total runtime, while keeping every engine's reps aligned
to the same pass.

Usage:
    uv run python -m scripts.collect.run_collection --day-index 1 \\
        --prompts config/prompts/prompts.csv \\
        --experiment-config config/experiment.yaml \\
        --prompt-order-seed 20260820

Runs are appended to <--runs-dir>/YYYY-MM-DD.jsonl (collector.storage.jsonl),
not written to SQLite.
"""
from __future__ import annotations

import argparse
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from collector.engines.registry import ADAPTERS
from collector import paths
from collector.runner.execute import execute_run
from collector.runner.experiment_config import EngineCell, ExperimentConfig, load_experiment_config
from collector.runner.load_prompts import Prompt, load_prompts_csv

# Hard ceiling on a single adapter call, enforced by execute_run regardless
# of the adapter's own SDK-level timeout/retry settings (collector/config.py).
# Measured worst case in practice is ~25s; 90s is a generous backstop so one
# stuck engine can never block the others past a bounded wait (see
# collector.runner.execute.execute_run's timeout_seconds docstring).
PER_REQUEST_TIMEOUT_SECONDS = 90.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--day-index", type=int, required=True, help="1-based day index within the collection window")
    parser.add_argument("--prompts", type=Path, default=paths.PROMPTS_CSV)
    parser.add_argument("--experiment-config", type=Path, default=paths.EXPERIMENT_YAML)
    parser.add_argument("--runs-dir", type=Path, default=None, help="defaults to data/01_raw/runs/ (collector.storage.jsonl.DEFAULT_RUNS_DIR)")
    parser.add_argument(
        "--prompt-order-seed",
        type=int,
        required=True,
        help="seeds the per-pass prompt shuffle shared by every engine; logged on every row so the order is fully reconstructable",
    )
    return parser.parse_args(argv)


def _build_passes(prompts: list[Prompt], *, reps_per_day: int, seed: int) -> list[list[Prompt]]:
    """One shuffled prompt order per pass (rep), computed once up front from
    a single seeded RNG so every engine thread reads the exact same
    sequence of passes -- the barrier only guarantees engines stay in lock
    step if what they're stepping through is identical."""
    rng = random.Random(seed)
    return [rng.sample(prompts, len(prompts)) for _ in range(reps_per_day)]


def _run_engine(
    *,
    cell: EngineCell,
    passes: list[list[Prompt]],
    total_per_engine: int,
    day_index: int,
    seed: int,
    runs_dir: Path | None,
    barrier: threading.Barrier,
) -> None:
    adapter = ADAPTERS[cell.engine]
    prompts_per_pass = len(passes[0]) if passes else 0

    for rep_index, order in enumerate(passes, start=1):
        for position_in_pass, prompt in enumerate(order):
            overall = (rep_index - 1) * prompts_per_pass + position_in_pass + 1
            print(
                f"[{cell.engine}] pass {rep_index}/{len(passes)} "
                f"pos {position_in_pass}/{prompts_per_pass - 1} "
                f"(run {overall}/{total_per_engine}) "
                f"intent={prompt.intent_id} paraphrase={prompt.paraphrase_id} seed={seed}"
            )
            try:
                run_id = execute_run(
                    adapter=adapter,
                    engine=cell.engine,
                    model_requested=cell.model_requested,
                    intent_id=prompt.intent_id,
                    paraphrase_id=prompt.paraphrase_id,
                    prompt_text=prompt.prompt_text,
                    rep_index=rep_index,
                    day_index=day_index,
                    prompt_order_seed=seed,
                    position_in_pass=position_in_pass,
                    runs_dir=runs_dir,
                    timeout_seconds=PER_REQUEST_TIMEOUT_SECONDS,
                )
                print(f"stored {run_id}")
            finally:
                # execute_run's own adapter-call timeout/error handling
                # already turns every per-request failure into a written
                # error row rather than an exception (INSTRUMENT.md section
                # 7's adapter contract, extended to the timeout wrapper) --
                # this finally is a backstop for anything that still escapes
                # (e.g. a write failure), so the barrier is always released
                # exactly once per (prompt, rep) and no engine ever stalls
                # the others waiting on one that crashed instead of erroring.
                barrier.wait()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    config: ExperimentConfig = load_experiment_config(args.experiment_config)
    if not (1 <= args.day_index <= config.days):
        raise SystemExit(
            f"--day-index {args.day_index} is outside the declared window of {config.days} days"
        )

    prompts = load_prompts_csv(args.prompts)
    seed = args.prompt_order_seed
    print(f"prompt_order_seed={seed}")

    passes = _build_passes(prompts, reps_per_day=config.reps_per_day, seed=seed)
    total_per_engine = len(prompts) * config.reps_per_day

    # One party per engine cell (INSTRUMENT.md section 6/7: every cell is
    # its own access path). Every real experiment config declares one cell
    # per distinct engine; a config with two cells for the same engine would
    # fire both concurrently against that vendor, since the barrier has no
    # notion of "vendor" -- not handled here, since no config does this.
    barrier = threading.Barrier(len(config.engines))

    with ThreadPoolExecutor(max_workers=len(config.engines)) as executor:
        futures = {
            executor.submit(
                _run_engine,
                cell=cell,
                passes=passes,
                total_per_engine=total_per_engine,
                day_index=args.day_index,
                seed=seed,
                runs_dir=args.runs_dir,
                barrier=barrier,
            ): cell.engine
            for cell in config.engines
        }
        # Collect every engine's outcome via as_completed rather than
        # letting the first exception propagate -- a setup failure (e.g. a
        # missing API key) in one engine must not cancel or truncate the
        # others, which keep running in their own threads regardless.
        errors: dict[str, BaseException] = {}
        for future in as_completed(futures):
            engine = futures[future]
            try:
                future.result()
            except BaseException as exc:
                errors[engine] = exc
                barrier.abort()

    if errors:
        details = "; ".join(f"{engine}: {exc!r}" for engine, exc in errors.items())
        raise SystemExit(f"collection failed for engine(s): {details}")


if __name__ == "__main__":
    main()
