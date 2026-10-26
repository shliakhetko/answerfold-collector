"""Google Gemini and Google AI Mode answers do not ship (Google's terms do not
allow sharing them). This module is the one place that decides what is
removed from those rows and how the public reader reads what is left.

What is removed, in place, same layout (the key stays, the value becomes
null or an empty list):

* window rows: the answer text (Gemini candidate text parts and the opaque
  thought signature; AI Mode text_blocks and reconstructed_markdown), Gemini
  grounding supports (they quote answer segments), the Gemini search entry
  point (rendered HTML holding the search queries), the Gemini search query
  text, every AI Mode reference field except the link and its index
  (title, snippet, source name, thumbnail, icon), AI Mode images, shopping
  results and any other extra block, and the SerpApi saved-page links and
  search id in AI Mode search_metadata;
* app-check (web interface) rows: the answer text;
* coded mentions: the quoted sentence (evidence_span); where it started in the
  answer ships as evidence_span_position.

What stays: run ids, request (the prompt sent), timestamps, model ids,
finish reason, usage, cited URLs in order, Gemini source titles (for Gemini
they are the cited site's domain, never a page title, and the Gemini title
fallback for wrapped citation URLs reads them), and the AI Mode search
metadata.

What is added, per row, under the key ``withheld``, computed from the
original before removal: whether the answer had text, and for Gemini the
number of searches and the number of site-limited searches (a query holding
``site:``). Those are the only things any figure needs from the removed
text, apart from which companies the text names: that comes from the
shipped table ``data/coded/name_matches.jsonl`` (one row per withheld text,
computed privately with the study's own name matcher).

``restore_for_parse`` turns a stripped row back into something the shared
extractors (analysis.aggregate) read the same way as the original: the
answer text becomes a marker that names the run, and each Gemini search
becomes a placeholder query. Nothing a figure counts changes.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from analysis.aggregate.response_fields import extract_raw_response_text
from analysis.coding.text_clean import strip_markdown_links

WITHHELD_ENGINES = ("google_gemini", "google_ai_mode")
WITHHELD_KEY = "withheld"
REASON = "Google's terms do not allow sharing Gemini or AI Mode answers; the text was removed before shipping"
MARKER_PREFIX = "[answer text withheld: "
MARKER_SUFFIX = "]"
QUERY_PLACEHOLDER = "[search query withheld]"
SITE_OPERATOR = "site:"
AI_MODE_REFERENCE_KEEP = ("link", "index")
# AI Mode response keys that keep their (stripped) content; every other top-level key (inline_images,
# shopping_results, ...) is emptied. search_metadata keeps only its status and timings: the SerpApi saved-page
# links (json_endpoint, markdown_endpoint, raw_html_file, prettify_html_file), the Google URL and the SerpApi
# search id are removed. status is what the finish-reason extractor reads.
AI_MODE_TOP_LEVEL = ("search_metadata", "search_parameters", "text_blocks", "reconstructed_markdown", "references")
AI_MODE_METADATA_KEEP = ("status", "created_at", "processed_at", "total_time_taken")


def is_withheld_engine(engine: str | None) -> bool:
    return engine in WITHHELD_ENGINES


def marker(text_id: str) -> str:
    """The stand-in for a withheld answer text. It names the run (or the
    app-check capture), so two withheld texts never share a marker."""
    return f"{MARKER_PREFIX}{text_id}{MARKER_SUFFIX}"


def marker_id(text: str | None) -> str | None:
    """The run id inside a marker, or None when the text is not a marker."""
    if isinstance(text, str) and text.startswith(MARKER_PREFIX) and text.endswith(MARKER_SUFFIX):
        return text[len(MARKER_PREFIX):-len(MARKER_SUFFIX)]
    return None


def gemini_queries(response: dict[str, Any] | None) -> list[str]:
    """The search queries of a Gemini response, the same rule the search
    figures use (string items of the first candidate's web_search_queries)."""
    candidate = ((response or {}).get("candidates") or [{}])[0] or {}
    metadata = candidate.get("grounding_metadata") or {}
    return [q for q in (metadata.get("web_search_queries") or []) if isinstance(q, str)]


def _strip_gemini(response: dict[str, Any]) -> None:
    # HTTP response headers of the API call: no figure reads them.
    http = response.get("sdk_http_response")
    if isinstance(http, dict) and "headers" in http:
        http["headers"] = None
    for candidate in response.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if isinstance(content, dict) and "parts" in content:
            content["parts"] = []
        metadata = candidate.get("grounding_metadata")
        if isinstance(metadata, dict):
            for key in ("grounding_supports", "search_entry_point", "web_search_queries", "retrieval_queries"):
                if key in metadata:
                    metadata[key] = None


def _strip_ai_mode(response: dict[str, Any]) -> None:
    metadata = response.get("search_metadata")
    if isinstance(metadata, dict):
        for key in list(metadata):
            if key not in AI_MODE_METADATA_KEEP:
                metadata[key] = None
    for key in list(response):
        if key not in AI_MODE_TOP_LEVEL:  # images, shopping results and anything else SerpApi adds
            response[key] = [] if isinstance(response[key], list) else None
    if "text_blocks" in response:
        response["text_blocks"] = []
    if "reconstructed_markdown" in response:
        response["reconstructed_markdown"] = None
    refs = response.get("references")
    if isinstance(refs, list):
        response["references"] = [
            {k: item[k] for k in AI_MODE_REFERENCE_KEEP if k in item} if isinstance(item, dict) else item for item in refs
        ]


def strip_window_record(record: dict[str, Any]) -> dict[str, Any]:
    """A window row as it ships. ChatGPT and Claude rows come back unchanged
    (the same object); Gemini and AI Mode rows come back as a stripped copy
    with the ``withheld`` block. A row with no response (a failed call) has
    nothing to strip and only gets the block."""
    engine = record.get("engine")
    if not is_withheld_engine(engine):
        return record
    out = copy.deepcopy(record)
    response = out.get("response")
    text = extract_raw_response_text(engine, response) if isinstance(response, dict) else None
    block: dict[str, Any] = {"reason": REASON, "answer_text_present": bool(text)}
    if engine == "google_gemini":
        queries = gemini_queries(response) if isinstance(response, dict) else []
        block["n_search_queries"] = len(queries)
        block["n_site_limited_search_queries"] = sum(1 for q in queries if SITE_OPERATOR in q.lower())
    if isinstance(response, dict):
        if engine == "google_gemini":
            _strip_gemini(response)
        else:
            _strip_ai_mode(response)
    out[WITHHELD_KEY] = block
    return out


def restore_for_parse(record: dict[str, Any]) -> dict[str, Any]:
    """Inverse view of strip_window_record for the shared extractors: the
    answer text becomes the run's marker (only where the original had text)
    and each Gemini search a placeholder query. Rows without a ``withheld``
    block come back unchanged."""
    block = record.get(WITHHELD_KEY)
    if not block:
        return record
    out = copy.deepcopy(record)
    response = out.get("response")
    if not isinstance(response, dict):
        return out
    text = marker(out["run_id"]) if block.get("answer_text_present") else None
    if out.get("engine") == "google_gemini":
        candidates = response.get("candidates") or []
        if candidates and isinstance(candidates[0], dict):
            first = candidates[0]
            if text is not None:
                content = first.get("content") if isinstance(first.get("content"), dict) else {}
                content["parts"] = [{"text": text}]
                first["content"] = content
            n = int(block.get("n_search_queries") or 0)
            if n:
                metadata = first.get("grounding_metadata") if isinstance(first.get("grounding_metadata"), dict) else {}
                metadata["web_search_queries"] = [QUERY_PLACEHOLDER] * n
                first["grounding_metadata"] = metadata
    elif text is not None:
        response["reconstructed_markdown"] = text
    return out


def strip_ui_run(run: dict[str, Any]) -> dict[str, Any]:
    """An app-check (web interface) row as it ships: Gemini text removed."""
    if not is_withheld_engine(run.get("engine")):
        return run
    out = dict(run)
    out[WITHHELD_KEY] = {"reason": REASON, "answer_text_present": bool(run.get("text"))}
    out["text"] = None
    return out


def restore_ui_run(run: dict[str, Any]) -> dict[str, Any]:
    block = run.get(WITHHELD_KEY)
    if not block:
        return run
    out = {k: v for k, v in run.items() if k != WITHHELD_KEY}
    out["text"] = marker(run["run_key"]) if block.get("answer_text_present") else None
    return out


def mention_engine(mention: dict[str, Any]) -> str:
    """Engine of a coded mention, from its run id (``<engine>-i..-p..-d..-r..``)."""
    return str(mention.get("run_id") or "").split("-", 1)[0]


def strip_mention(mention: dict[str, Any], answer_text: str | None = None) -> dict[str, Any]:
    """A coded mention as it ships: the quoted sentence of a Gemini or AI Mode
    answer is removed (the company it names stays). Where the quote started in
    the link-stripped answer text (-1 if not found) ships as
    evidence_span_position, the one thing a figure reads from the quote
    (analysis.coding.b1_per_run_company, first mention position)."""
    if not is_withheld_engine(mention_engine(mention)):
        return mention
    out = dict(mention)
    span = mention.get("evidence_span")
    out["evidence_span_position"] = strip_markdown_links(answer_text or "").find(span) if span is not None else -1
    out["evidence_span"] = None
    return out


# --------------------------------------------------------------------------
# audit: anyone can check that no withheld text ships
# --------------------------------------------------------------------------

AI_MODE_TEXT_KEYS = ("text_blocks", "reconstructed_markdown")
GEMINI_TEXT_METADATA_KEYS = ("grounding_supports", "search_entry_point", "web_search_queries", "retrieval_queries")


def _window_row_problems(record: dict[str, Any]) -> list[str]:
    run_id = record.get("run_id")
    response = record.get("response")
    problems: list[str] = []
    if not isinstance(response, dict):
        return problems
    if extract_raw_response_text(record["engine"], response):
        problems.append(f"{run_id}: answer text present")
    if record["engine"] == "google_gemini":
        for candidate in response.get("candidates") or []:
            content = (candidate or {}).get("content") or {}
            if content.get("parts"):
                problems.append(f"{run_id}: candidate parts present")
            metadata = (candidate or {}).get("grounding_metadata") or {}
            for key in GEMINI_TEXT_METADATA_KEYS:
                if metadata.get(key):
                    problems.append(f"{run_id}: grounding_metadata.{key} present")
    else:
        for key in AI_MODE_TEXT_KEYS:
            if response.get(key):
                problems.append(f"{run_id}: {key} present")
        for key, value in (response.get("search_metadata") or {}).items():
            if key not in AI_MODE_METADATA_KEEP and value is not None:
                problems.append(f"{run_id}: search_metadata.{key} present")
        for key, value in response.items():
            if key not in AI_MODE_TOP_LEVEL and value:
                problems.append(f"{run_id}: {key} present")
        for item in response.get("references") or []:
            extra = set(item or {}) - set(AI_MODE_REFERENCE_KEEP)
            if extra:
                problems.append(f"{run_id}: reference fields {sorted(extra)} present")
                break
    return problems


def audit_package(root) -> dict[str, Any]:
    """Scan the shipped data: no Gemini or AI Mode answer text, text part,
    rebuilt markdown, text block, snippet, reference title, search query or
    quote; ChatGPT and Claude answer text, quotes and search queries present.
    Returns {"problems": [...], "summary": str, "counts": {...}}."""
    root = Path(root)
    problems: list[str] = []
    counts = {"withheld_rows": 0, "withheld_rows_with_block": 0, "kept_rows_with_text": 0, "kept_rows_with_queries": 0,
              "ui_withheld": 0, "ui_kept_with_text": 0, "mentions_withheld_quote_present": 0, "mentions_kept_with_quote": 0}
    for path in sorted((root / "data" / "raw").glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                engine = record.get("engine")
                if is_withheld_engine(engine):
                    counts["withheld_rows"] += 1
                    counts["withheld_rows_with_block"] += WITHHELD_KEY in record
                    if WITHHELD_KEY not in record:
                        problems.append(f"{record.get('run_id')}: no withheld block")
                    problems += _window_row_problems(record)
                else:
                    counts["kept_rows_with_text"] += bool(extract_raw_response_text(engine, record.get("response")))
                    envelope = json.dumps(record.get("response") or {})
                    counts["kept_rows_with_queries"] += ('"web_search_call"' in envelope or '"server_tool_use"' in envelope)
    ui_path = root / "data" / "ui" / "ui_runs.jsonl"
    if ui_path.exists():
        for line in ui_path.read_text(encoding="utf-8").splitlines():
            run = json.loads(line)
            if is_withheld_engine(run.get("engine")):
                counts["ui_withheld"] += 1
                if run.get("text"):
                    problems.append(f"app check {run.get('run_key')}: text present")
            else:
                counts["ui_kept_with_text"] += bool(run.get("text"))
    mentions_path = root / "data" / "coding" / "mentions_coded.jsonl"
    if mentions_path.exists():
        with mentions_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                mention = json.loads(line)
                if is_withheld_engine(mention_engine(mention)):
                    if mention.get("evidence_span"):
                        counts["mentions_withheld_quote_present"] += 1
                        problems.append(f"mention in {mention.get('run_id')}: quote present")
                else:
                    counts["mentions_kept_with_quote"] += bool(mention.get("evidence_span"))
    for key in ("kept_rows_with_text", "kept_rows_with_queries", "ui_kept_with_text", "mentions_kept_with_quote"):
        if counts[key] == 0:
            problems.append(f"no ChatGPT or Claude data found for {key}")
    summary = (
        f"{len(problems)} problem(s); {counts['withheld_rows']} Gemini/AI Mode window rows checked, "
        f"{counts['ui_withheld']} Gemini app-check rows checked; ChatGPT/Claude rows with text: {counts['kept_rows_with_text']}, "
        f"with search calls: {counts['kept_rows_with_queries']}, app-check texts: {counts['ui_kept_with_text']}, "
        f"quotes: {counts['mentions_kept_with_quote']}"
    )
    return {"problems": problems, "summary": summary, "counts": counts}
