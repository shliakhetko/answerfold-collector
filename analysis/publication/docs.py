"""Renders the package documents from templates in config/publish_docs/.

Every number in a document is a tag read from the shipped frozen figures
file, the shipped manifests or the build's own counts; nothing is retyped.

Tags:
    {{figures.S3.denominator_before_exclusion.openai_chatgpt|int}}   a value in the figures file
    {{intermediates.daily_panel.completeness.total_cells_expected|int}}
    {{inputs.instrument_commit}}
    {{package.window_rows_withheld|int}}                            a count made by the build
    {{manifest.raw.window.start}}                                   a shipped manifest (phase_a, raw, aux, ui_test)
    {{gap:name}}   a value the study never stored: printed as a visible gap
    {{todo:text}}  an open item the owner must fill: printed as a visible TODO

Filters: int, f1, f2, f3, pct, pct0, label, text, list, engines, keys, yesno.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from collector import paths

ENGINE_NAMES = {
    "anthropic_claude": "Anthropic Claude",
    "google_ai_mode": "Google AI Mode",
    "google_gemini": "Google Gemini",
    "openai_chatgpt": "OpenAI ChatGPT",
}
FILTERS = {"int", "f1", "f2", "f3", "pct", "pct0", "label", "text", "list", "engines", "keys", "yesno"}
TAG = re.compile(r"\{\{(.+?)\}\}")
TEMPLATE_DIR = paths.CONFIG_DIR / "publish_docs"
DOC_TEMPLATES = ("README.md", "METHODS.md", "LIMITATIONS.md", "NAMING.md")
PLAIN_COPIES = ("LICENSE-DATA",)  # copied as is, not rendered


class DocError(Exception):
    pass


def lookup(root: Any, dotted: str) -> Any:
    node = root
    for token in dotted.split("."):
        if isinstance(node, dict) and token in node:
            node = node[token]
        elif isinstance(node, list) and token.isdigit():
            node = node[int(token)]
        else:
            raise DocError(f"key {dotted!r} not found (stopped at {token!r})")
    return node


def apply_filter(name: str, value: Any) -> str:
    if value is None:
        return "null"
    if name == "int":
        return f"{value:,}"
    if name == "f1":
        return f"{value:.1f}"
    if name == "f2":
        return f"{value:.2f}"
    if name == "f3":
        return f"{value:.3f}"
    if name == "pct":
        return f"{value:.1%}"
    if name == "pct0":
        return f"{value:.0%}"
    if name == "label":
        return str(value).replace("_", " ")
    if name == "list":
        return ", ".join(str(v) for v in value)
    if name == "keys":
        keys = [str(k) for k in value]
        return ", ".join(keys[:-1]) + (" and " if len(keys) > 1 else "") + keys[-1] if keys else "none"
    if name == "yesno":
        return "yes" if value else "no"
    if name == "engines":
        names = [ENGINE_NAMES.get(v, v) for v in value]
        return ", ".join(names[:-1]) + (" and " if len(names) > 1 else "") + names[-1] if names else "none"
    return str(value)


def render(template: str, context: dict[str, Any]) -> str:
    def sub(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if inner.startswith("gap:"):
            return f"[NOT STORED: {inner[4:].replace('_', ' ')}]"
        if inner.startswith("todo:"):
            return f"[TODO: {inner[5:].strip()}]"
        path, sep, last = inner.rpartition("|")
        if not sep or last not in FILTERS:
            path, last = inner, "text"
        return apply_filter(last, lookup(context, path))

    return TAG.sub(sub, template)


def build_context(publish: Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    figures = json.loads((publish / "figures" / "figures.json").read_text(encoding="utf-8"))
    manifests = {
        key: json.loads((publish / "manifests" / f"{key}_MANIFEST.json").read_text(encoding="utf-8"))
        for key in ("phase_a", "raw", "aux", "ui_test")
    }
    context = {
        "figures": figures["figures"],
        "intermediates": figures["intermediates"],
        "inputs": figures["inputs"],
        "manifest": manifests,
    }
    if extra:
        context.update(extra)
    return context


def render_all(repo: Path, publish: Path, *, package: dict[str, Any]) -> None:
    """Render every document template into publish/. `package` holds the
    build's own counts (rows withheld, robots files emptied, chart failures),
    read by the templates as {{package....}}."""
    context = build_context(publish, {"package": package})
    template_dir = Path(repo) / TEMPLATE_DIR
    for name in DOC_TEMPLATES:
        source = template_dir / name
        if not source.exists():
            raise DocError(f"template {source} is missing")
        (publish / name).write_text(render(source.read_text(encoding="utf-8"), context), encoding="utf-8")
    for name in PLAIN_COPIES:
        shutil.copyfile(template_dir / name, publish / name)
    zen = template_dir / "zenodo.json"
    if zen.exists():
        text = render(zen.read_text(encoding="utf-8"), context)
        json.loads(text)  # must stay valid JSON
        (publish / ".zenodo.json").write_text(text, encoding="utf-8")
