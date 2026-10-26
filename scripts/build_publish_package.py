#!/usr/bin/env python3
"""The one command for a reader of the publish package.

    uv run python scripts/build_publish_package.py

What it does, in order:

1. Rebuilds every figure from the shipped tables (data/) with the same code
   that built the study's figures, with no model call.
2. Compares the rebuilt figures file, byte for byte, with the shipped frozen
   copy (figures/figures.json).
3. Checks that no Google Gemini or Google AI Mode answer text ships (those
   rows are stripped; see analysis/publication/withheld.py).

It needs no API key and reads nothing outside this folder. Exit code 0 only
if every step passed.

Owner-only flag: --write-frozen PATH writes the rebuilt figures file to PATH
(used once, by the owner's build, to create the shipped frozen copy). The
normal run never writes into the package.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from analysis.publication import bundle, withheld  # noqa: E402  (after the path fix)
from collector import paths  # noqa: E402


STUDY_DATA_ENV = "ANSWERFOLD_STUDY_DATA"


def find_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    if os.environ.get(STUDY_DATA_ENV):
        return Path(os.environ[STUDY_DATA_ENV]).resolve()
    if (PACKAGE_ROOT / bundle.COMPANY_TABLE).exists():
        return PACKAGE_ROOT
    if (PACKAGE_ROOT / paths.rel(paths.PUBLISH_DIR) / bundle.COMPANY_TABLE).exists():
        return PACKAGE_ROOT / paths.rel(paths.PUBLISH_DIR)
    return PACKAGE_ROOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild every figure from the publish package and verify it")
    parser.add_argument("--root", default=None, help="package root (default: the folder holding scripts/)")
    parser.add_argument("--output-dir", default=None, help="where to write the rebuilt figures (default: a temporary folder)")
    parser.add_argument("--write-frozen", default=None, help="owner only: also write the rebuilt figures file to this path")
    args = parser.parse_args(argv)

    root = find_root(args.root)
    if not (root / bundle.COMPANY_TABLE).exists():
        print(f"SKIPPED: no study data at {root}. Download the study package from "
              f"https://doi.org/10.5281/zenodo.23559036, unpack it, and run this command inside it, "
              f"or set {STUDY_DATA_ENV} to the unpacked folder.")
        return 0
    print(f"package root: {root}")

    out_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="publish_rebuild_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    rebuilt = out_dir / "figures.rebuilt.json"
    try:
        bundle.build_seam(root, rebuilt)
    except bundle.BundleError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1
    print(f"rebuilt figures written to {rebuilt}")

    if args.write_frozen:
        target = Path(args.write_frozen)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(rebuilt.read_bytes())
        print(f"frozen copy written to {target} (owner step)")
        return 0

    ok = True
    shipped = root / bundle.FIGURES_COPY
    if not shipped.exists():
        print(f"STOP: shipped frozen copy {bundle.FIGURES_COPY} is missing", file=sys.stderr)
        return 1
    same, detail = bundle.compare_to_shipped(rebuilt, shipped)
    print(f"rebuilt figures vs shipped frozen copy: {detail}")
    if not same:
        print("FAIL: the rebuilt figures differ from the shipped copy. That is a bug, not noise.", file=sys.stderr)
        ok = False
    else:
        print("OK: every figure rebuilt from the shipped tables matches the shipped frozen copy.")

    audit = withheld.audit_package(root)
    print(f"withheld-text check: {audit['summary']}")
    for problem in audit["problems"][:20]:
        print(f"  {problem}", file=sys.stderr)
    if audit["problems"]:
        print("FAIL: Gemini or AI Mode text found in the shipped data.", file=sys.stderr)
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
