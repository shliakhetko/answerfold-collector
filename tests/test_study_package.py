"""The one command against the AI receptionists study package (Zenodo).

The study data does not ship with the tool. The full rebuild runs only when
ANSWERFOLD_STUDY_DATA points to an unpacked copy of the package from
https://doi.org/10.5281/zenodo.23559036.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "build_publish_package.py"
STUDY_DATA = os.environ.get("ANSWERFOLD_STUDY_DATA")


def test_one_command_skips_without_study_data(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "ANSWERFOLD_STUDY_DATA"}
    result = subprocess.run([sys.executable, str(SCRIPT), "--root", str(tmp_path)],
                            capture_output=True, text=True, env=env, cwd=REPO)
    assert result.returncode == 0
    assert "SKIPPED: no study data" in result.stdout


@pytest.mark.skipif(not STUDY_DATA, reason="set ANSWERFOLD_STUDY_DATA to an unpacked study package")
def test_one_command_rebuilds_the_study_figures(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), "--output-dir", str(tmp_path)],
                            capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "matches the shipped frozen copy" in result.stdout
