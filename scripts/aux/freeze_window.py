#!/usr/bin/env python3
"""Task 0 -- freeze the raw collection window.

  1. Tag the head of exp/ai-voice-assistants as collection-close-2026-08-26
     (annotated). That commit == main and carries no committed config/experiment.yaml
     / prompts.csv / raw data, so the tag pins the instrument code only; the
     prompt set and raw data are pinned by SHA256 in the manifest below. This
     limitation is logged in the ambiguity ledger (Task 8).
  2. Copy every data/01_raw/runs/*.jsonl to data/01_raw/raw_frozen/ read-only (chmod 444).
  3. Write data/01_raw/MANIFEST.json: per raw file the relative path, SHA256,
     byte size, line count, unique run_id count, and the first + last ts_utc
     inside the file.

Read-only against the working tree except for the frozen copy and the
manifest. Does not commit -- the caller commits (CLAUDE.md).

Usage:
    uv run python -m scripts.aux.freeze_window
    uv run python -m scripts.aux.freeze_window --no-tag        # skip the git tag
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from analysis.aux.record import collector_version
from scripts.aux._run import rel

from collector import paths
REPO_ROOT = paths.REPO_ROOT
DEFAULT_RUNS_DIR = paths.RUNS_DIR
FROZEN_DIR = paths.RAW_FROZEN_DIR
MANIFEST_PATH = paths.RAW_MANIFEST

TAG_NAME = "collection-close-2026-08-26"
TAG_TARGET = "exp/ai-voice-assistants"
WINDOW_START = "2026-08-20"
WINDOW_END = "2026-08-26"


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)


def ensure_tag(dry_run: bool = False) -> str:
    existing = _git("tag", "--list", TAG_NAME).stdout.strip()
    if existing:
        target = _git("rev-list", "-n", "1", TAG_NAME).stdout.strip()
        return f"tag {TAG_NAME} already exists at {target[:12]} -- left as is"
    rev = _git("rev-parse", "--verify", TAG_TARGET).stdout.strip()
    if not rev:
        return f"SKIPPED: branch {TAG_TARGET} not found; create the tag manually"
    if dry_run:
        return f"would tag {TAG_NAME} -> {TAG_TARGET} ({rev[:12]})"
    result = _git(
        "tag", "-a", TAG_NAME, TAG_TARGET,
        "-m",
        "Collection window close for the AI voice-receptionist study (day 7 = "
        f"{WINDOW_END}). Instrument code only; config/prompts, config/experiment.yaml, raw runs "
        "are pinned by SHA256 in data/01_raw/MANIFEST.json.",
    )
    if result.returncode != 0:
        return f"git tag failed: {result.stderr.strip()}"
    return f"tagged {TAG_NAME} -> {TAG_TARGET} ({rev[:12]})"


def _file_manifest(path: Path) -> dict:
    sha = hashlib.sha256()
    line_count = 0
    first_ts: str | None = None
    last_ts: str | None = None
    run_ids: set[str] = set()
    with path.open("rb") as fb:
        for raw_line in fb:
            sha.update(raw_line)
            text = raw_line.strip()
            if not text:
                continue
            line_count += 1
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                continue
            ts = record.get("ts_utc")
            if ts:
                first_ts = ts if first_ts is None or ts < first_ts else first_ts
                last_ts = ts if last_ts is None or ts > last_ts else last_ts
            rid = record.get("run_id")
            if rid:
                run_ids.add(rid)
    return {
        "path": rel(path),
        "sha256": sha.hexdigest(),
        "bytes": path.stat().st_size,
        "lines": line_count,
        "unique_run_ids": len(run_ids),
        "first_ts_utc": first_ts,
        "last_ts_utc": last_ts,
    }


def build_manifest(runs_dir: Path) -> dict:
    files = sorted(runs_dir.glob("*.jsonl"))
    entries = [_file_manifest(p) for p in files]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts.aux.freeze_window",
        "collector_version": collector_version(),
        "window": {"start": WINDOW_START, "end": WINDOW_END, "days": 7},
        "tag": TAG_NAME,
        "raw_files": entries,
        "totals": {
            "files": len(entries),
            "lines": sum(e["lines"] for e in entries),
            "unique_run_ids": sum(e["unique_run_ids"] for e in entries),
            "bytes": sum(e["bytes"] for e in entries),
        },
    }


def freeze_copy(runs_dir: Path, dry_run: bool = False) -> list[str]:
    notes: list[str] = []
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    for src in sorted(runs_dir.glob("*.jsonl")):
        dst = FROZEN_DIR / src.name
        if dry_run:
            notes.append(f"would copy {src.name} -> {rel(dst)}")
            continue
        if dst.exists():
            os.chmod(dst, stat.S_IWUSR | stat.S_IRUSR)
        shutil.copy2(src, dst)
        os.chmod(dst, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # 0444
        notes.append(f"froze {src.name} ({dst.stat().st_size} bytes, read-only)")
    return notes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    p.add_argument("--no-tag", action="store_true", help="do not create the git tag")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    runs_dir = args.runs_dir
    if not runs_dir.is_dir():
        raise SystemExit(f"runs dir not found: {runs_dir}")

    print(f"[freeze] runs dir: {runs_dir}  ({len(list(runs_dir.glob('*.jsonl')))} files)")

    if not args.no_tag:
        print(f"[freeze] {ensure_tag(dry_run=args.dry_run)}")

    for note in freeze_copy(runs_dir, dry_run=args.dry_run):
        print(f"[freeze] {note}")

    manifest = build_manifest(runs_dir)
    if MANIFEST_PATH.exists():
        try:
            prior = json.loads(MANIFEST_PATH.read_text())
            if "aux" in prior:  # keep the Task 9 section if it ran first
                manifest["aux"] = prior["aux"]
        except json.JSONDecodeError:
            pass
    if args.dry_run:
        print(f"[freeze] would write {rel(MANIFEST_PATH)} ({manifest['totals']})")
    else:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[freeze] wrote {rel(MANIFEST_PATH)}")
    print(f"[freeze] totals: {manifest['totals']}")


if __name__ == "__main__":
    main()
