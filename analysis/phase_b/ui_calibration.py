"""UI calibration reading layer: the 30 manual web-UI captures under
<data_root>/ui_test/ (see ledger entry `ui-calibration-design-2026-09-29`).

Answer-container-only extraction: the answer is cut out of the captured
page by ONE fixed selector per UI, never by scanning the whole page:

- Gemini web UI:  `model-response message-content div.markdown`
- ChatGPT web UI: `div[data-assistant-markdown]`

Exactly one match is required. Zero or several matches, an unreadable
file, or empty text is recorded as an extraction miss with its reason,
never guessed at. No AI call anywhere.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from collector import paths

UI_DIR = paths.data_rel(paths.UI_TEST_DIR)  # relative to the data root
EYEBALL_FILE = "eyeball_check.json"
CONTAINER_SELECTOR = {
    "gemini": "model-response message-content div.markdown",
    "openai": "div[data-assistant-markdown]",
}
API_ENGINE_FOR_UI = {"gemini": "google_gemini", "openai": "openai_chatgpt"}
RUN_NAME = re.compile(r"^(?P<ui>gemini|openai)_I(?P<intent>\d+)_P(?P<para>\d+)_R(?P<rep>\d+)\.html$")


def extract_answer_text(ui: str, html: str) -> tuple[str | None, str | None]:
    """(text, None) on success; (None, reason) on an extraction miss."""
    soup = BeautifulSoup(html, "html.parser")
    matches = soup.select(CONTAINER_SELECTOR[ui])
    if len(matches) != 1:
        return None, f"container_matches_{len(matches)}_expected_1"
    text = matches[0].get_text(separator="\n").strip()
    if not text:
        return None, "container_empty"
    return text, None


def read_ui_runs(data_root: Path) -> list[dict[str, Any]]:
    ui_dir = Path(data_root) / UI_DIR
    runs: list[dict[str, Any]] = []
    if not ui_dir.is_dir():
        return runs
    for html_path in sorted(ui_dir.glob("*/*.html")):
        m = RUN_NAME.match(html_path.name)
        if not m:
            runs.append({"run_key": html_path.stem, "extraction_miss": "unrecognized_file_name", "text": None})
            continue
        ui = m["ui"]
        run = {
            "run_key": html_path.stem,
            "ui": ui,
            "engine": API_ENGINE_FOR_UI[ui],
            "intent_id": int(m["intent"]),
            "paraphrase_id": int(m["para"]),
            "rep": int(m["rep"]),
            "png": str(html_path.with_suffix(".png").relative_to(ui_dir)),
            "text": None,
            "extraction_miss": None,
        }
        try:
            text, miss = extract_answer_text(ui, html_path.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:  # unreadable/unparseable capture: listed, never fatal
            text, miss = None, f"unparseable:{type(exc).__name__}"
        run["text"], run["extraction_miss"] = text, miss
        runs.append(run)
    return runs


def read_eyeball(data_root: Path) -> dict[str, Any]:
    path = Path(data_root) / UI_DIR / EYEBALL_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
