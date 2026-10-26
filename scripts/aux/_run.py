"""Shared helpers for the auxiliary collector CLIs: the pre-run estimate /
confirmation gate (CLAUDE.md: ask again before APIs / long jobs) and a small
progress printer.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]


def rel(path: Path | str) -> str:
    """Repo-relative string for display, tolerant of paths outside the repo
    (custom --aux-root, tests running under tmp_path)."""
    p = Path(path)
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def show_estimate_and_gate(title: str, lines: Iterable[str], *, confirmed: bool, dry_run: bool) -> bool:
    """Print the estimate. Return True if the collector should proceed.

    - dry_run: print and stop (returns False).
    - confirmed (--confirm): proceed without prompting.
    - otherwise: prompt on a TTY; refuse on a non-TTY (must pass --confirm).
    """
    print(f"\n=== {title} ===")
    for line in lines:
        print(f"  {line}")
    print("=" * (len(title) + 8))

    if dry_run:
        print("[dry-run] no network calls made.")
        return False
    if confirmed:
        print("[confirmed] proceeding.")
        return True
    if not sys.stdin.isatty():
        print("[blocked] re-run with --confirm (or --dry-run) -- refusing to spend/crawl unattended.")
        return False
    answer = input("Proceed? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


class Progress:
    def __init__(self, total: int, *, every: int = 25, label: str = "done") -> None:
        self.total = total
        self.every = max(1, every)
        self.label = label
        self.n = 0

    def tick(self, note: str = "") -> None:
        self.n += 1
        if self.n % self.every == 0 or self.n == self.total:
            suffix = f" — {note}" if note else ""
            print(f"  [{self.n}/{self.total}] {self.label}{suffix}", flush=True)
