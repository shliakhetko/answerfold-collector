"""UI_CALIBRATION -- 30 manual web-UI runs against the API runs for the
SAME prompts in the collection window, both counted with the ONE
deterministic matcher (analysis.phase_b.name_matcher). Calibration only,
never a finding. Design, selectors and caveats: ledger entry
`ui-calibration-design-2026-09-29` (written before any UI mention was
counted).

Raw counts only ("2 of 3 UI runs", "18 of 21 API runs"); no rates.

Caveats stated on the record: 30 runs; 2 engines (Gemini, ChatGPT) x 5
prompts (intents 1-5, paraphrase 1) x 3 repeats each (even, unlike the
planned uneven design); no Claude / AI Mode UI runs; UI captured
2026-08-29, three days after the window (the API side is the window,
2026-08-20..26, 7 days x 3 reps = 21 runs per prompt).

value[engine]["i{intent}-p{para}"] = {
  ui_runs, ui_runs_parsed, api_runs,
  companies_named_in_ui (union over the parsed UI runs),
  of_which_also_named_in_api (any API run of the 21),
  ui_only_companies, per_company: [{entity_id, ui_runs_with, api_runs_with}]
}
The by-eye check (data/01_raw/ui_test/eyeball_check.json) is summarized in
notes["eyeball"], per run and in category counts.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from analysis.phase_b.exclusions import exclusion_reason
from analysis.phase_b.figure_shape import FigureResult
from analysis.phase_b.name_matcher import build_matcher, match_entities
from analysis.phase_b.reading import FrozenInputs

FIGURE_ID = "UI_CALIBRATION"
LEDGER_REF = "ui-calibration-design-2026-09-29"
CAVEATS = (
    "30 runs only. 2 engines (Gemini, ChatGPT) x 5 prompts (intents 1-5, paraphrase 1) x 3 repeats each; "
    "no Claude or AI Mode UI runs; intents 6 and 7 not covered. UI runs were captured 2026-08-29, three days "
    "after the collection window; the API side is the window (7 days x 3 reps = 21 runs per prompt). Both "
    "sides are counted with the same deterministic name matcher, not the study's model-based mention "
    "extraction. Raw counts, not rates. Not a finding either way."
)
EYEBALL_CATEGORIES = (
    "generic_phrase_false_positive", "source_chip_only", "ambiguous_or_duplicate_entity", "visible_vendor_not_matched",
)


def _eyeball_summary(eyeball: dict[str, Any], ui_runs: list[dict[str, Any]]) -> dict[str, Any]:
    runs = eyeball.get("runs", {})
    keys = [r["run_key"] for r in ui_runs]
    checked = [k for k in keys if runs.get(k, {}).get("png_checked")]
    per_run = {}
    counts = {c: 0 for c in EYEBALL_CATEGORIES}
    runs_with = {c: 0 for c in EYEBALL_CATEGORIES}
    n_strict_agree = 0
    n_vendor_level_agree = 0
    for k in keys:
        rec = runs.get(k)
        if not rec:
            continue
        for c in EYEBALL_CATEGORIES:
            counts[c] += len(rec.get(c, []))
            if rec.get(c):
                runs_with[c] += 1
        strict = all(not rec.get(c) for c in EYEBALL_CATEGORIES)
        # vendor-level: ignore generic-phrase false positives only
        vendor_level = all(not rec.get(c) for c in EYEBALL_CATEGORIES if c != "generic_phrase_false_positive")
        n_strict_agree += strict
        n_vendor_level_agree += vendor_level
        per_run[k] = {
            "container_extraction": rec.get("container_extraction"),
            "strict_agreement": strict,
            "agreement_ignoring_generic_phrase_hits": vendor_level,
            **{c: rec.get(c, []) for c in EYEBALL_CATEGORIES},
            "png_capture_note": rec.get("png_capture_note"),
        }
    return {
        "n_ui_runs": len(keys),
        "n_checked_by_eye_against_png": len(checked),
        "n_runs_strict_agreement": n_strict_agree,
        "n_runs_agreement_ignoring_generic_phrase_hits": n_vendor_level_agree,
        "n_hits_by_category": counts,
        "n_runs_with_category": runs_with,
        "categories": {
            "generic_phrase_false_positive": "matcher hit on an ordinary word/phrase, not a vendor mention",
            "source_chip_only": "name appears only inside a source chip in the container, not in the answer body",
            "ambiguous_or_duplicate_entity": "one visible name maps to two roster entities (or a substring of another vendor's name)",
            "visible_vendor_not_matched": "vendor visibly named in the answer but not matched (not in the in-frame roster)",
        },
        "per_run": per_run,
    }


def compute(inputs: FrozenInputs) -> FigureResult:
    ui_runs = inputs.ui_test_runs
    if not ui_runs:
        return FigureResult(
            figure_id=FIGURE_ID, status="untestable", value=None, denominator_before_exclusion=None,
            denominator_after_exclusion=None, n_included=None, n_excluded=None, threshold=None,
            reason_untestable="no ui_test runs under the data root",
        )
    matcher = build_matcher(inputs.entities)

    api_matches: dict[tuple[str, int, int], list[set[str]]] = defaultdict(list)
    for run in inputs.raw_runs:
        if exclusion_reason(run) is not None:
            continue
        api_matches[(run["engine"], run["intent_id"], run["paraphrase_id"])].append(
            match_entities(run.get("raw_response_text") or "", matcher)
        )

    parsed = [r for r in ui_runs if r.get("text")]
    misses = [{"run_key": r["run_key"], "reason": r.get("extraction_miss")} for r in ui_runs if not r.get("text")]

    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for r in ui_runs:
        grouped[(r["engine"], r["intent_id"], r["paraphrase_id"])].append(r)

    value: dict[str, dict[str, Any]] = defaultdict(dict)
    supporting: list[dict[str, Any]] = []
    for (engine, intent, para), group in sorted(grouped.items()):
        ui_sets = [match_entities(r["text"], matcher) for r in group if r.get("text")]
        api_sets = api_matches.get((engine, intent, para), [])
        ui_union = set().union(*ui_sets) if ui_sets else set()
        api_union = set().union(*api_sets) if api_sets else set()
        per_company = [
            {
                "entity_id": e,
                "ui_runs_with": f"{sum(1 for s in ui_sets if e in s)} of {len(ui_sets)}",
                "api_runs_with": f"{sum(1 for s in api_sets if e in s)} of {len(api_sets)}",
            }
            for e in sorted(ui_union)
        ]
        cell = {
            "ui_runs": len(group),
            "ui_runs_parsed": len(ui_sets),
            "api_runs": len(api_sets),
            "companies_named_in_ui": len(ui_union),
            "of_which_also_named_in_api": len(ui_union & api_union),
            "ui_only_companies": sorted(ui_union - api_union),
            "per_company": per_company,
        }
        value[engine][f"i{intent}-p{para}"] = cell
        supporting.append({"engine": engine, "intent_id": intent, "paraphrase_id": para, **{k: v for k, v in cell.items() if k != "per_company"}})

    return FigureResult(
        figure_id=FIGURE_ID,
        status="ok",
        value=dict(value),
        denominator_before_exclusion=len(ui_runs),
        denominator_after_exclusion=len(parsed),
        n_included=len(parsed),
        n_excluded={"container_extraction_miss": len(misses)},
        threshold=None,
        reason_untestable=None,
        supporting_table=supporting,
        notes={
            "calibration_only_not_a_finding": True,
            "ledger_ref": LEDGER_REF,
            "caveats": CAVEATS,
            "extraction_misses": misses,
            "container_selectors": {
                "gemini": "model-response message-content div.markdown",
                "openai": "div[data-assistant-markdown]",
            },
            "ui_engine_to_api_engine": {"gemini": "google_gemini", "openai": "openai_chatgpt"},
            "eyeball": _eyeball_summary(inputs.ui_eyeball, ui_runs),
            "planned_design": "3 engines, uneven repeats per engine and prompt",
            "real_design": "2 engines x 5 prompts x 3 repeats (even), 30 runs",
        },
    )
