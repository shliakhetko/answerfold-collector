"""Business figures for every remaining dev section: S3, S2, K1, O1, K2, V2, A2, A3, O2, X1, S4,
the change-detection outcome and the study/method counts the user page needs.

Exploratory and derived, built only from the frozen data/04_results/figures.json plus the study constants
the existing runners already read (config/experiment.yaml, the raw MANIFEST window, the inclusion-frame
date). Nothing is recomputed from raw data and nothing here feeds or changes figures.json.

Aggregate only: counts, shares, correlations and flags per engine, never a company name, a company id or a
domain. Per-engine values are copied from figures.json; the rest is min/max with the engine that holds
it (extremes.py), ratios, counts and boolean claim flags, one flag per comparative sentence a reader could
write. Claim flags live under derived.<section>.flags; a flag turns false when the data stops saying so.
"""
from __future__ import annotations

import datetime as dt
import re
import statistics
from typing import Any

from analysis.coding.b3_aggregates import ELIGIBILITY_FLOOR
from analysis.phase_b.business.extremes import ENGINE_ORDER, extreme, ordered
from analysis.phase_b.figures.o1 import CONSISTENT_WITH_NOISE
from analysis.phase_b.figures.v2 import REAL_LABEL

OPENAI = "openai_chatgpt"
AI_MODE = "google_ai_mode"
GEMINI = "google_gemini"
CLAUDE = "anthropic_claude"
POOLED = "pooled"
PAIR_JOINER = "__"  # engine pair id in this file; "|" is the filter separator of the page tags
K2_TREATMENTS_HEADLINE = "included"
K2_POPULATION = "excluded_population"
K2_BELOW, K2_BETWEEN, K2_ABOVE = "below", "between", "above"
# Plain-language helpers for the S2 sentence (2026-10-12 final edit pass). "About half" means within HALF_BAND of
# 0.5; a share is "about" a simple fraction when it is within FRACTION_TOLERANCE of it. Stored as numbers, the
# wording is the template's job.
HALF_BAND = 0.05
FRACTION_TOLERANCE = 0.03
SIMPLE_FRACTIONS = ((1, 6), (1, 5), (1, 4), (1, 3), (1, 2), (2, 3), (3, 4), (5, 6))
VENDOR_CLASS_PREFIX = "vendor"
SECTIONS = ("s3", "s2", "k1", "o1", "k2", "v2", "a2", "a3", "o2", "x1", "s4", "ui_calibration")
UI_PROMPT_ID = re.compile(r"^i(\d+)-p1$")
UI_CAPTURED = re.compile(r"captured (\d{4}-\d{2}-\d{2})")

NOTE = (
    "Exploratory and derived; separate from figures.json. Every per-engine value is copied from figures.json; "
    "ratios, shares of stored counts, min/max with engine and the claim flags are derived from those values. "
    "Aggregate only: no company names."
)


class RestBusinessError(Exception):
    """A stored figure no longer has the shape this file reads."""


# --- small helpers ------------------------------------------------------------------------------

def _words(label: str | None) -> str | None:
    return None if label is None else str(label).replace("_", " ")


def _share(count: int | float, total: int | float) -> float | None:
    return count / total if total else None


def _rename(found: dict[str, Any], key: str) -> dict[str, Any]:
    return {side: (None if v is None else {key: v["engine"], "value": v["value"]}) for side, v in found.items()}


def _extreme_keyed(values: dict[str, Any], key: str) -> dict[str, Any]:
    """extreme() for values keyed by something other than an engine (an intent id, an engine pair)."""
    return _rename(extreme(values), key)


def _all(engines, test) -> bool:
    return all(test(e) for e in engines)


def _pair_id(raw: str) -> str:
    return PAIR_JOINER.join(raw.split("|"))


def _stored_bands(note: str) -> dict[str, float]:
    """The K2 decision bands, read from the stored band note ("At or above 60% ... Below 40% ...")."""
    high = re.search(r"At or above (\d+(?:\.\d+)?)%", note)
    low = re.search(r"Below (\d+(?:\.\d+)?)%", note)
    if high is None or low is None:
        raise RestBusinessError("K2 pre_reg_bands_note no longer states the upper and lower band")
    return {"high": float(high.group(1)) / 100, "low": float(low.group(1)) / 100}


def _fig(figures: dict[str, Any], name: str) -> dict[str, Any]:
    return figures["figures"][name]


# --- raw blocks, one per dev section -----------------------------------------------------------------

def _s3(figures: dict[str, Any]) -> dict[str, Any]:
    f = _fig(figures, "S3")
    identity = f["notes"]["search_fired_zero_sources_identity"]
    engines = {}
    for e in ordered(f["value"]):
        engines[e] = {
            "share": f["value"][e],
            "n_citing_nothing": identity[e]["n_zero_sources"],
            "n_answers": f["denominator_after_exclusion"][e],
            "n_planned": f["denominator_before_exclusion"][e],
            "n_not_usable": f["denominator_before_exclusion"][e] - f["denominator_after_exclusion"][e],
        }
    return {"engines": engines}


def _vs_half(value: float) -> str:
    if value > 0.5 + HALF_BAND:
        return "above"
    if value < 0.5 - HALF_BAND:
        return "below"
    return "about"


def _plain_fraction(value: float) -> dict[str, Any]:
    """Nearest simple fraction of SIMPLE_FRACTIONS ("2 in 3"), with the distance and whether it is within
    FRACTION_TOLERANCE (otherwise `label` is None: no plain fraction fits)."""
    n, d = min(SIMPLE_FRACTIONS, key=lambda nd: (abs(value - nd[0] / nd[1]), nd[1]))
    error = abs(value - n / d)
    return {"numerator": n, "denominator": d, "label": f"{n} in {d}" if error <= FRACTION_TOLERANCE else None,
            "distance": error}


def _s2(figures: dict[str, Any]) -> dict[str, Any]:
    f = _fig(figures, "S2")
    teardown = f["notes"]["teardown_by_engine"]
    engines = {}
    for e in ordered(f["value"]):
        under, qualifying = teardown[e]["entities_under_10pct_own_citation"], teardown[e]["qualifying_entities"]
        engines[e] = {
            "restricted": f["value"][e]["restricted"],
            "unrestricted": f["value"][e]["unrestricted"],
            "n_restricted": f["denominator_after_exclusion"][e]["restricted"],
            "n_unrestricted": f["denominator_after_exclusion"][e]["unrestricted"],
            "n_dropped_no_sources": f["n_excluded"][e]["restricted"]["no_sources"],
            "teardown_under": under,
            "teardown_qualifying": qualifying,
            "teardown_share": _share(under, qualifying),
        }
    return {"teardown_line": f["threshold"]["teardown_own_citation_below"], "engines": engines}


def _k1(figures: dict[str, Any]) -> dict[str, Any]:
    f = _fig(figures, "K1")
    notes = f["notes"]
    engines = {}
    for e in ordered(f["value"]):
        distinct = notes["distinct_companies_mentioned_by_engine"][e]
        engines[e] = {
            "top_share": f["value"][e],
            "top_n": len(notes["top_5_ranked"][e]),
            "count_to_half": notes["count_to_50pct"][e],
            "distinct_named": distinct,
            "mentions": f["denominator_after_exclusion"][e],
            "flat_share": notes["flat_baseline_recomputed_by_engine"][e],
            "out_of_frame": notes["out_of_frame_named_by_engine"][e]["total_out_of_frame_entities"],
        }
    return {
        "roster_size": notes["flat_baseline_fixed"]["n"],
        "flat_fixed_share": notes["flat_baseline_fixed"]["share"],
        "engines": engines,
    }


def _o1(figures: dict[str, Any]) -> dict[str, Any]:
    f = _fig(figures, "O1")
    notes = f["notes"]
    engines = {}
    for e in ordered(f["value"]):
        labels = notes["noise_labels_by_engine"][e]
        engines[e] = {
            "head_overlap": f["value"][e]["head_mean_jaccard"],
            "tail_overlap": f["value"][e]["tail_mean_jaccard"],
            "head_minus_tail": notes["raw_head_vs_tail_descriptive"][e]["head_minus_tail"],
            "head_label": labels["head"], "head_label_words": _words(labels["head"]),
            "tail_label": labels["tail"], "tail_label_words": _words(labels["tail"]),
            "stable_head_churning_tail_met": notes["prereg_band_met_by_engine"][e],
            "n_qualifying": notes["n_qualifying_by_engine"][e],
            "n_tail": notes["n_tail_by_engine"][e],
        }
    return {
        "head_size": notes["head_size"],
        "head_stable_at_or_above": notes["interpretation_thresholds"]["head_stable_at_or_above"],
        "tail_churning_at_or_below": notes["interpretation_thresholds"]["tail_churning_at_or_below"],
        "engines": engines,
    }


def _k2(figures: dict[str, Any]) -> dict[str, Any]:
    f = _fig(figures, "K2")
    notes = f["notes"]
    scopes = ordered(e for e in f["value"] if e != POOLED)
    treatments = list(f["value"][scopes[0]])

    def cell(scope: str, treatment: str) -> dict[str, Any]:
        return {
            "share": f["value"][scope][treatment],
            "n_citations": f["denominator_after_exclusion"][scope][treatment],
            "distinct_domains": notes["unique_domains"][scope][treatment],
            "domains_to_half": notes["count_to_50pct"][scope][treatment],
            "listed_n": len(notes["top_10_ranked"][scope][treatment]),
        }

    return {
        "treatments": treatments,
        "bands": _stored_bands(notes["pre_reg_bands_note"]),
        "engines": {e: {t: cell(e, t) for t in treatments} for e in scopes},
        "pooled": {t: cell(POOLED, t) for t in treatments},
    }


def _ui_calibration(figures: dict[str, Any]) -> dict[str, Any]:
    f = _fig(figures, "UI_CALIBRATION")
    captured = UI_CAPTURED.search(f["notes"]["caveats"])
    if captured is None:
        raise RestBusinessError("UI_CALIBRATION caveats no longer state the date the UI runs were captured")
    engines = {}
    for e in ordered(f["value"]):
        prompts = {}
        for pid, cell in f["value"][e].items():
            match = UI_PROMPT_ID.match(pid)
            if match is None:
                raise RestBusinessError(f"UI_CALIBRATION prompt id {pid!r} is not i<n>-p1")
            prompts[match.group(1)] = {
                "ui_runs": cell["ui_runs"],
                "ui_runs_parsed": cell["ui_runs_parsed"],
                "api_runs": cell["api_runs"],
                "named_in_ui": cell["companies_named_in_ui"],
                "also_named_in_api": cell["of_which_also_named_in_api"],
                "ui_only": len(cell["ui_only_companies"]),
            }
        engines[e] = {"prompts": dict(sorted(prompts.items(), key=lambda kv: int(kv[0])))}
    return {
        "calibration_only": bool(f["notes"]["calibration_only_not_a_finding"]),
        "ui_captured": captured.group(1),
        "engines": engines,
    }


def _v2(figures: dict[str, Any]) -> dict[str, Any]:
    f = _fig(figures, "V2")
    notes, verdict = f["notes"], f["verdict"]
    engines = {}
    for e in ordered(f["value"]):
        noise = verdict["noise_comparison_by_engine"][e]
        engines[e] = {
            "median_excess_drift": f["value"][e],
            "n_units": f["n_included"][e],
            "n_alert_worthy": notes["n_alert_worthy_by_engine"][e],
            "alert_worthy_share": notes["share_alert_worthy_by_engine"][e],
            "null_share": noise["null_share_alert_worthy"],
            "null_p95_share": noise["null_p95_share"],
            "ratio_to_null": noise["ratio_observed_to_null"],
            "observed_percentile_in_null": noise["observed_percentile_in_null"],
            "label": noise["label"], "label_words": _words(noise["label"]),
            "stored_verdict": verdict["verdict_by_engine"][e],
            "stored_verdict_words": _words(verdict["verdict_by_engine"][e]),
            "n_all_zero": notes["n_all_zero_by_engine"][e],
            "share_all_zero": notes["share_all_zero_by_engine"][e],
        }
    return {
        "alert_worthy_at_or_above": f["threshold"]["alert_worthy_at_or_above"],
        "no_product_at_or_below": f["threshold"]["no_product_at_or_below"],
        "engines": engines,
    }


def _a2(figures: dict[str, Any]) -> dict[str, Any]:
    f = _fig(figures, "A2")
    ci = f["notes"]["ci_by_pair"]
    pairs = {}
    for raw in ci:
        a, b = raw.split("|")
        pairs[_pair_id(raw)] = {
            "engine_a": a, "engine_b": b,
            "correlation": f["value"][raw],
            "ci_low": ci[raw]["ci_low"], "ci_high": ci[raw]["ci_high"],
            "excludes_zero": ci[raw]["excludes_zero"],
            "n_points": ci[raw]["n"], "n_companies": ci[raw]["n_companies"],
        }
    levels = {c["ci_level"] for c in ci.values()}
    if len(levels) != 1:
        raise RestBusinessError("A2 pair intervals no longer share one level")
    return {
        "shared_movement_at_or_above": f["threshold"]["shared_movement_at_or_above"],
        "mean_correlation": f["value"]["mean"],
        "ci_level": levels.pop(),
        "stored_verdict": f["verdict"]["verdict"], "stored_verdict_words": _words(f["verdict"]["verdict"]),
        "pairs": pairs,
    }


def _a3(figures: dict[str, Any]) -> dict[str, Any]:
    f = _fig(figures, "A3")
    notes = f["notes"]
    engines = {}
    for e in ordered(f["value"]):
        flagged = len(notes["teardown_by_engine"][e])
        n = f["denominator_after_exclusion"][e]
        engines[e] = {
            "rank_correlation": f["value"][e],
            "n_companies": n,
            "teardown_flagged": flagged,
            "teardown_share": _share(flagged, n),
            "tie_only_distortion_max": notes["bottom_rank_tie_check"][e]["max_possible_tie_only_distortion"],
            "ties_alone_can_reach_gap": notes["bottom_rank_tie_check"][e]["ties_alone_can_reach_threshold"],
        }
    return {"teardown_rank_gap": notes["teardown_rank_gap"], "engines": engines}


def _o2(figures: dict[str, Any]) -> dict[str, Any]:
    f = _fig(figures, "O2")
    notes = f["notes"]
    engines = {}
    paraphrases = set()
    for e in ordered(f["value"]):
        cells = f["value"][e].values()
        paraphrases |= {c["n_paraphrases"] for c in cells}
        para = statistics.fmean(c["paraphrase_jaccard"] for c in cells)
        rep = statistics.fmean(c["rep_jaccard"] for c in cells)
        engines[e] = {
            "paraphrase_mean": para, "rep_mean": rep, "difference": para - rep,
            "n_cells": len(f["value"][e]),
            "n_paraphrase_lower": sum(c["direction"] == "paraphrase_lower" for c in cells),
        }
    boot = notes["bootstrap"]
    return {
        "paraphrase_mean": notes["overall_paraphrase_mean"],
        "rep_mean": notes["overall_rep_mean"],
        "difference": notes["overall_diff_paraphrase_minus_rep"],
        "ratio": _share(notes["overall_paraphrase_mean"], notes["overall_rep_mean"]),
        "ci_level": boot["ci_level"], "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
        "interval_crosses_zero": boot["crosses_zero"],
        "n_pairs": f["n_included"],
        "paraphrases_per_intent": sorted(paraphrases),
        "engines": engines,
    }


def _x1(figures: dict[str, Any]) -> dict[str, Any]:
    f = _fig(figures, "X1")
    notes = f["notes"]
    engines = {}
    for e in ordered(f["value"]):
        modal = notes["modal_class_by_engine"][e]
        engines[e] = {
            "modal_class": modal,
            "modal_share": f["value"][e][modal],
            "modal_margin": notes["modal_margin_by_engine"][e],
            "weak_modal": notes["weak_modal_by_engine"][e],
            "raw_vendor_share": notes["raw_vendor_share_by_engine"][e],
            "review_site_share": notes["review_site_share_by_engine"][e],
            "n_classified": f["denominator_after_exclusion"][e],
            "n_unclassified": f["n_excluded"][e].get("unclassified", 0),
        }
    return {"identical_stored": notes["modal_class_identical_across_engines"], "engines": engines}


def _s4(figures: dict[str, Any], intent_map: dict[str, bool]) -> dict[str, Any]:
    f = _fig(figures, "S4")
    identity = _fig(figures, "S3")["notes"]["search_fired_zero_sources_identity"]
    engines = {}
    for e in ordered(f["value"]):
        n = f["notes"]["n_search_not_fired_by_engine"][e]
        usable = f["denominator_after_exclusion"][e]
        engines[e] = {
            "n_not_fired": n, "n_usable": usable, "share": _share(n, usable),
            "n_zero_sources": identity[e]["n_zero_sources"],
            "no_search_equals_zero_sources_when_search_ran": identity[e]["n_both"] == identity[e]["n_no_search"],
            "same_runs_as_zero_sources": identity[e]["identical_sets"],
        }
    cells = [r for r in f["supporting_table"] if r["engine"] == GEMINI]
    by_paraphrase: dict[int, list[dict[str, Any]]] = {}
    for r in cells:
        if not intent_map[str(r["intent_id"])]:
            by_paraphrase.setdefault(r["paraphrase_id"], []).append(r)
    mean_rate = {p: statistics.fmean(r["share"] for r in rows) for p, rows in by_paraphrase.items()}
    top = max(sorted(mean_rate), key=mean_rate.get)
    intents = {}
    for r in sorted((r for r in cells if r["paraphrase_id"] == top), key=lambda r: r["intent_id"]):
        others = [o["share"] for o in cells if o["intent_id"] == r["intent_id"] and o["paraphrase_id"] != top]
        intents[str(r["intent_id"])] = {
            "rate": r["share"], "n_not_fired": r["n_search_not_fired"], "n_usable": r["n_usable_runs"],
            "branded": intent_map[str(r["intent_id"])],
            "best_other_paraphrase_rate": max(others),
        }
    return {
        "engines": engines,
        "gemini_top_paraphrase": {
            "paraphrase_id": top,
            "mean_rate_unbranded": mean_rate[top],
            "best_other_mean_rate_unbranded": max(v for p, v in mean_rate.items() if p != top),
            "intents": intents,
        },
    }


def _method(figures: dict[str, Any], study: dict[str, Any], o2: dict[str, Any], s3: dict[str, Any]) -> dict[str, Any]:
    inter = figures["intermediates"]
    intent_map = inter["intent_map"]
    own = inter["source_ownership"]
    engines = ordered(s3["engines"])
    planned = {e: s3["engines"][e]["n_planned"] for e in engines}
    paraphrases = o2["paraphrases_per_intent"]
    return {
        "engines": engines,
        "intent_ids": sorted(intent_map, key=int),
        "n_intents": len(intent_map),
        "n_branded_intents": sum(1 for b in intent_map.values() if b),
        "n_unbranded_intents": sum(1 for b in intent_map.values() if not b),
        "paraphrases_per_intent": paraphrases[0] if len(paraphrases) == 1 else None,
        "planned_runs": {"engines": planned, "total": sum(planned.values())},
        "usable_runs_total": inter["citation_panel"]["usable_runs_total"],
        "coded_runs": {
            "with_in_frame_mention": own["distinct_responses_covered"],
            "all_mentions_out_of_frame": own["usable_runs_with_mentions_all_out_of_frame"],
            "total": own["distinct_responses_covered"] + own["usable_runs_with_mentions_all_out_of_frame"],
            "no_extracted_mention": own["usable_runs_with_zero_extracted_mentions"],
        },
        "roster_size": _fig(figures, "K1")["notes"]["flat_baseline_fixed"]["n"],
        "qualifying_companies": dict(inter["daily_panel"]["qualifying_by_engine"]),
        "eligibility_floor": ELIGIBILITY_FLOOR,
        "grid_cells_expected": inter["daily_panel"]["completeness"]["total_cells_expected"],
        "runs_per_day": study["runs_per_day"],
        "days": study["days"],
        "window": dict(study["window"]),
        "inclusion_frame_frozen": study["frame_frozen_date"],
        "inclusion_frame_days_after_window": study["frame_days_after_window"],
    }


# --- derived layer ------------------------------------------------------------------------------------

def _d_s3(s3: dict[str, Any]) -> dict[str, Any]:
    names = ordered(s3["engines"])
    share = {e: s3["engines"][e]["share"] for e in names}
    found = extreme(share)
    top, rest = found["max"]["engine"], {e: v for e, v in share.items() if e != found["max"]["engine"]}
    return {
        "share": found,
        "n_citing_nothing": extreme({e: s3["engines"][e]["n_citing_nothing"] for e in names}),
        "n_answers": extreme({e: s3["engines"][e]["n_answers"] for e in names}),
        "counts": {"engines": len(names), "citing_nothing_total": sum(s3["engines"][e]["n_citing_nothing"] for e in names)},
        "flags": {
            "highest_share_engine_is_google_gemini": top == GEMINI,
            "highest_share_ahead_of_every_other_engine": share[top] > max(rest.values()),
            "every_engine_cites_something_in_most_answers": _all(names, lambda e: share[e] < 0.5),
            "every_engine_has_some_answers_citing_nothing": _all(names, lambda e: s3["engines"][e]["n_citing_nothing"] > 0),
        },
    }


def _d_s2(s2: dict[str, Any], s3: dict[str, Any]) -> dict[str, Any]:
    names = ordered(s2["engines"])
    eng = s2["engines"]
    restricted = {e: eng[e]["restricted"] for e in names}
    unrestricted = {e: eng[e]["unrestricted"] for e in names}
    gap = {e: unrestricted[e] - restricted[e] for e in names}
    teardown = {e: eng[e]["teardown_share"] for e in names}
    under = {e: eng[e]["teardown_under"] for e in names}
    low, high = extreme(restricted)["min"]["engine"], extreme(restricted)["max"]["engine"]
    others = [e for e in names if e != low]
    s3_top = extreme({e: s3["engines"][e]["share"] for e in names})["max"]["engine"]
    vs_half = {e: _vs_half(restricted[e]) for e in names}
    plain = {e: _plain_fraction(restricted[e]) for e in names}
    ranked = sorted(names, key=lambda e: (-restricted[e], names.index(e)))
    near = lambda e, target: abs(restricted[e] - target) <= FRACTION_TOLERANCE  # noqa: E731
    return {
        "restricted_by_engine": restricted,
        "restricted_vs_half": vs_half,
        "engines_above_half": [e for e in names if vs_half[e] == "above"],
        "engines_about_half": [e for e in names if vs_half[e] == "about"],
        "engines_below_half": [e for e in names if vs_half[e] == "below"],
        "restricted_plain_fraction": plain,
        "restricted_ranked_engines": ranked,
        "restricted_half_band": HALF_BAND,
        "restricted_fraction_tolerance": FRACTION_TOLERANCE,
        "restricted": extreme(restricted),
        "unrestricted": extreme(unrestricted),
        "unrestricted_minus_restricted": extreme(gap),
        "n_restricted": extreme({e: eng[e]["n_restricted"] for e in names}),
        "teardown_under": extreme(under),
        "teardown_share": extreme(teardown),
        "counts": {"engines": len(names), "teardown_under_total": sum(under.values()),
                   "teardown_qualifying_total": sum(eng[e]["teardown_qualifying"] for e in names)},
        "flags": {
            "lowest_restricted_engine_is_openai_chatgpt": low == OPENAI,
            "openai_chatgpt_restricted_below_every_other_engine": restricted[OPENAI] < min(v for e, v in restricted.items() if e != OPENAI),
            "every_other_engine_restricted_above_half": _all(others, lambda e: restricted[e] > 0.5),
            "highest_restricted_engine_is_anthropic_claude": high == CLAUDE,
            # What the data says after the 2026-10-12 relabel (the two flags above evaluate false).
            "highest_restricted_engine_is_google_gemini": high == GEMINI,
            "highest_two_restricted_engines_are_google_gemini_and_anthropic_claude": set(ranked[:2]) == {GEMINI, CLAUDE},
            "google_gemini_and_anthropic_claude_clearly_above_half": _all([GEMINI, CLAUDE], lambda e: vs_half[e] == "above"),
            "google_gemini_and_anthropic_claude_about_two_thirds": _all([GEMINI, CLAUDE], lambda e: near(e, 2 / 3)),
            "google_ai_mode_about_half": vs_half[AI_MODE] == "about",
            "google_ai_mode_below_half": restricted[AI_MODE] < 0.5,
            "openai_chatgpt_about_one_in_six": near(OPENAI, 1 / 6),
            "exactly_two_engines_clearly_above_half": sum(v == "above" for v in vs_half.values()) == 2,
            "most_teardown_companies_engine_is_google_gemini": extreme(under)["max"]["engine"] == GEMINI,
            "highest_teardown_share_engine_is_google_gemini": extreme(teardown)["max"]["engine"] == GEMINI,
            "lowest_teardown_share_engine_is_openai_chatgpt": extreme(teardown)["min"]["engine"] == OPENAI,
            "lowest_restricted_engine_has_lowest_teardown_share": low == extreme(teardown)["min"]["engine"],
            "unrestricted_at_or_above_restricted_in_every_engine": _all(names, lambda e: gap[e] >= 0),
            "largest_unrestricted_gap_engine_is_highest_citing_nothing_engine": extreme(gap)["max"]["engine"] == s3_top,
        },
    }


def _d_k1(k1: dict[str, Any]) -> dict[str, Any]:
    names = ordered(k1["engines"])
    eng = k1["engines"]
    top = {e: eng[e]["top_share"] for e in names}
    half = {e: eng[e]["count_to_half"] for e in names}
    distinct = {e: eng[e]["distinct_named"] for e in names}
    flat = {e: eng[e]["flat_share"] for e in names}
    lead = extreme(top)["max"]["engine"]
    return {
        "top_share": extreme(top),
        "count_to_half": extreme(half),
        "distinct_named": extreme(distinct),
        "flat_share": extreme(flat),
        "out_of_frame": extreme({e: eng[e]["out_of_frame"] for e in names}),
        "top_share_over_flat": {e: _share(top[e], flat[e]) for e in names},
        "top_share_over_fixed_flat": {e: _share(top[e], k1["flat_fixed_share"]) for e in names},
        "top_share_over_flat_range": _extreme_keyed({e: _share(top[e], flat[e]) for e in names}, "engine"),
        "half_share_of_named": {e: _share(half[e], distinct[e]) for e in names},
        "half_share_of_named_range": _extreme_keyed({e: _share(half[e], distinct[e]) for e in names}, "engine"),
        "counts": {"engines": len(names), "top_n": max(eng[e]["top_n"] for e in names),
                   "roster": k1["roster_size"]},
        "flags": {
            "highest_top_share_engine_is_openai_chatgpt": lead == OPENAI,
            "highest_top_share_ahead_of_every_other_engine": top[lead] > max(v for e, v in top.items() if e != lead),
            "fewest_companies_to_half_engine_is_openai_chatgpt": extreme(half)["min"]["engine"] == OPENAI,
            "openai_chatgpt_needs_fewer_companies_to_half_than_every_other_engine": half[OPENAI] < min(v for e, v in half.items() if e != OPENAI),
            "most_distinct_named_engine_is_google_gemini": extreme(distinct)["max"]["engine"] == GEMINI,
            "top_share_above_even_spread_in_every_engine": _all(names, lambda e: top[e] > flat[e]),
            "top_share_above_fixed_roster_even_spread_in_every_engine": _all(names, lambda e: top[e] > k1["flat_fixed_share"]),
            "fewer_than_half_of_named_companies_reach_half_of_mentions_in_every_engine": _all(names, lambda e: half[e] * 2 < distinct[e]),
            "out_of_frame_companies_named_in_every_engine": _all(names, lambda e: eng[e]["out_of_frame"] > 0),
            "most_out_of_frame_engine_is_google_gemini": extreme({e: eng[e]["out_of_frame"] for e in names})["max"]["engine"] == GEMINI,
        },
    }


def _d_o1(o1: dict[str, Any]) -> dict[str, Any]:
    names = ordered(o1["engines"])
    eng = o1["engines"]
    head = {e: eng[e]["head_overlap"] for e in names}
    tail = {e: eng[e]["tail_overlap"] for e in names}
    diff = {e: eng[e]["head_minus_tail"] for e in names}
    stable, churn = o1["head_stable_at_or_above"], o1["tail_churning_at_or_below"]
    head_over_tail = {e: head[e] > tail[e] for e in names}
    labels = [eng[e][k] for e in names for k in ("head_label", "tail_label")]
    at_line = [e for e in names if head[e] >= stable]
    top_head = extreme(head)["max"]["engine"]
    return {
        "head_overlap_by_engine": head,
        "tail_overlap_by_engine": tail,
        "head_gap_to_stable_line": {e: head[e] - stable for e in names},
        "engines_head_at_or_above_stable_line": at_line,
        "head_overlap": extreme(head),
        "tail_overlap": extreme(tail),
        "head_minus_tail": extreme(diff),
        "head_above_tail": head_over_tail,
        "head_at_or_above_stable_line": {e: head[e] >= stable for e in names},
        "tail_at_or_below_churning_line": {e: tail[e] <= churn for e in names},
        "beyond_noise": {e: {"head": eng[e]["head_label"] != CONSISTENT_WITH_NOISE,
                             "tail": eng[e]["tail_label"] != CONSISTENT_WITH_NOISE} for e in names},
        "counts": {
            "engines": len(names), "head_size": o1["head_size"],
            "engines_head_above_tail": sum(head_over_tail.values()),
            "engines_stable_head_churning_tail_met": sum(eng[e]["stable_head_churning_tail_met"] for e in names),
            "engines_head_at_or_above_stable_line": sum(head[e] >= stable for e in names),
            "engines_tail_at_or_below_churning_line": sum(tail[e] <= churn for e in names),
            "labels_beyond_noise": sum(label != CONSISTENT_WITH_NOISE for label in labels),
        },
        "flags": {
            "stable_head_churning_tail_met_in_no_engine": not any(eng[e]["stable_head_churning_tail_met"] for e in names),
            "stable_head_churning_tail_met_in_every_engine": all(eng[e]["stable_head_churning_tail_met"] for e in names),
            "no_head_overlap_at_or_above_stable_line": not any(head[e] >= stable for e in names),
            "highest_head_overlap_engine_is_google_gemini": extreme(head)["max"]["engine"] == GEMINI,
            "lowest_head_overlap_engine_is_google_ai_mode": extreme(head)["min"]["engine"] == AI_MODE,
            "head_above_tail_in_every_engine": all(head_over_tail.values()),
            "head_above_tail_in_some_engine": any(head_over_tail.values()),
            "every_label_consistent_with_noise": all(label == CONSISTENT_WITH_NOISE for label in labels),
            # Prose sweep (2026-10-12): no engine's tail overlap is as low as the churning line.
            "no_tail_overlap_at_or_below_churning_line": not any(tail[e] <= churn for e in names),
            # What the data says after the 2026-10-12 relabel (no_head_overlap_at_or_above_stable_line is false).
            "head_overlap_at_or_above_stable_line_in_some_engine": bool(at_line),
            "head_overlap_at_or_above_stable_line_in_exactly_one_engine": len(at_line) == 1,
            "only_google_gemini_head_overlap_at_or_above_stable_line": at_line == [GEMINI],
            "highest_head_overlap_engine_is_at_or_above_stable_line": head[top_head] >= stable,
            "stable_head_churning_tail_met_in_some_engine": any(eng[e]["stable_head_churning_tail_met"] for e in names),
            "stable_head_without_churning_tail_in_some_engine": any(
                head[e] >= stable and tail[e] > churn for e in names),
            **{f"head_overlap_at_or_above_stable_line_in_{e}": head[e] >= stable for e in names},
        },
    }


def _reading(share: float, bands: dict[str, float]) -> str:
    if share >= bands["high"]:
        return K2_ABOVE
    if share < bands["low"]:
        return K2_BELOW
    return K2_BETWEEN


def _d_k2(k2: dict[str, Any]) -> dict[str, Any]:
    names = ordered(k2["engines"])
    eng, bands = k2["engines"], k2["bands"]
    treatments = k2["treatments"]
    share = {t: {e: eng[e][t]["share"] for e in names} for t in treatments}
    reading = {t: {e: _reading(share[t][e], bands) for e in names} for t in treatments}
    head = K2_TREATMENTS_HEADLINE
    rising = {e: share[K2_POPULATION][e] - share[head][e] for e in names}
    in_head = reading[head]
    raised = [e for e in names if rising[e] > 0]
    lowered = [e for e in names if rising[e] < 0]
    return {
        "share_by_engine": {t: share[t] for t in treatments},
        "engines_exclusion_raises_share": raised,
        "engines_exclusion_lowers_share": lowered,
        "bands": dict(bands),
        "share": {t: extreme(share[t]) for t in treatments},
        "distinct_domains": {t: extreme({e: eng[e][t]["distinct_domains"] for e in names}) for t in treatments},
        "domains_to_half": {t: extreme({e: eng[e][t]["domains_to_half"] for e in names}) for t in treatments},
        "population_minus_included": rising,
        "reading": reading,
        "reading_counts": {t: {word: sum(r == word for r in reading[t].values()) for word in (K2_BELOW, K2_BETWEEN, K2_ABOVE)}
                           for t in treatments},
        "counts": {"engines": len(names), "treatments": len(treatments), "listed_n": max(eng[e][head]["listed_n"] for e in names)},
        "flags": {
            "no_engine_at_or_above_high_band_with_every_citation_counted": not any(r == K2_ABOVE for r in in_head.values()),
            "no_engine_at_or_above_high_band_in_any_treatment": not any(r == K2_ABOVE for t in treatments for r in reading[t].values()),
            "google_ai_mode_below_low_band_with_every_citation_counted": in_head[AI_MODE] == K2_BELOW,
            "google_ai_mode_only_engine_below_low_band_with_every_citation_counted": [e for e in names if in_head[e] == K2_BELOW] == [AI_MODE],
            "every_other_engine_between_bands_with_every_citation_counted": _all([e for e in names if e != AI_MODE], lambda e: in_head[e] == K2_BETWEEN),
            # Prose sweep (2026-10-12): the main-view short list never carries a majority of an engine's citations.
            "top_sites_hold_under_half_of_citations_with_every_citation_counted": all(v < 0.5 for v in share[head].values()),
            "highest_share_engine_is_openai_chatgpt_with_every_citation_counted": extreme(share[head])["max"]["engine"] == OPENAI,
            "dropping_population_domains_raises_share_in_every_engine": _all(names, lambda e: rising[e] > 0),
            # What the data says after the 2026-10-12 relabel (the raise-in-every-engine flag is false).
            "dropping_population_domains_raises_share_in_some_engine": bool(raised),
            "dropping_population_domains_raises_share_in_every_engine_but_one": len(raised) == len(names) - 1,
            "dropping_population_domains_lowers_share_only_in_anthropic_claude": lowered == [CLAUDE],
            "dropping_population_domains_raises_share_in_every_engine_except_anthropic_claude": raised == [e for e in names if e != CLAUDE],
            "pooled_needs_more_domains_to_half_than_any_engine": k2["pooled"][head]["domains_to_half"] > max(eng[e][head]["domains_to_half"] for e in names),
        },
    }


def _d_ui_calibration(ui: dict[str, Any], window_end: str) -> dict[str, Any]:
    names = ordered(ui["engines"])
    cells = {e: list(ui["engines"][e]["prompts"].values()) for e in names}
    every = [c for e in names for c in cells[e]]

    def total(e: str, key: str) -> int:
        return sum(c[key] for c in cells[e])

    totals = {e: {k: total(e, k) for k in ("ui_runs", "named_in_ui", "also_named_in_api", "ui_only")} for e in names}
    overall = {k: sum(totals[e][k] for e in names) for k in totals[names[0]]}
    gap = (dt.date.fromisoformat(ui["ui_captured"]) - dt.date.fromisoformat(str(window_end))).days
    return {
        "totals": totals,
        "overall": overall,
        "counts": {
            "engines": len(names),
            "prompts": len(cells[names[0]]),
            "ui_runs_per_prompt": max(c["ui_runs"] for c in every),
            "api_runs_per_prompt": max(c["api_runs"] for c in every),
            "ui_days_after_window": gap,
        },
        "named_in_ui_per_prompt": {
            "min": min(c["named_in_ui"] for c in every),
            "max": max(c["named_in_ui"] for c in every),
        },
        "flags": {
            "calibration_only_not_a_finding": ui["calibration_only"],
            "every_ui_run_parsed": all(c["ui_runs_parsed"] == c["ui_runs"] for c in every),
            "same_ui_runs_for_every_question": len({c["ui_runs"] for c in every}) == 1,
            "same_questions_in_every_engine": len({tuple(ui["engines"][e]["prompts"]) for e in names}) == 1,
            "ui_captured_after_window_closed": gap > 0,
            "more_api_runs_than_ui_runs_for_every_question": all(c["api_runs"] > c["ui_runs"] for c in every),
            "most_ui_named_companies_also_named_in_api_in_every_engine": _all(names, lambda e: 2 * totals[e]["also_named_in_api"] > totals[e]["named_in_ui"]),
            "every_engine_has_ui_named_companies_missing_from_api": _all(names, lambda e: totals[e]["ui_only"] > 0),
            "also_named_never_exceeds_named_in_ui": all(c["also_named_in_api"] <= c["named_in_ui"] for c in every),
        },
    }


def _d_v2(v2: dict[str, Any]) -> dict[str, Any]:
    names = ordered(v2["engines"])
    eng = v2["engines"]
    alert, no_product = v2["alert_worthy_at_or_above"], v2["no_product_at_or_below"]
    median = {e: eng[e]["median_excess_drift"] for e in names}
    beyond = {e: eng[e]["label"] == REAL_LABEL for e in names}
    return {
        "median_excess_drift": extreme(median),
        "alert_worthy_share": extreme({e: eng[e]["alert_worthy_share"] for e in names}),
        "null_share": extreme({e: eng[e]["null_share"] for e in names}),
        "ratio_to_null": extreme({e: eng[e]["ratio_to_null"] for e in names}),
        "share_all_zero": extreme({e: eng[e]["share_all_zero"] for e in names}),
        "n_units": extreme({e: eng[e]["n_units"] for e in names}),
        "beyond_noise": beyond,
        "counts": {
            "engines": len(names),
            "units_total": sum(eng[e]["n_units"] for e in names),
            "engines_beyond_noise": sum(beyond.values()),
            "engines_consistent_with_noise": len(names) - sum(beyond.values()),
        },
        "flags": {
            "median_excess_drift_zero_in_every_engine": _all(names, lambda e: median[e] == 0),
            "median_at_or_below_no_product_line_in_every_engine": _all(names, lambda e: median[e] <= no_product),
            "no_engine_median_at_or_above_alert_line": not any(median[e] >= alert for e in names),
            "median_company_shows_no_drift_worth_alerting": not any(median[e] >= alert for e in names)
            and _all(names, lambda e: median[e] <= no_product),
            "stored_verdict_identical_in_every_engine": len({eng[e]["stored_verdict"] for e in names}) == 1,
            "alert_worthy_share_above_noise_only_share_in_every_engine": _all(names, lambda e: eng[e]["alert_worthy_share"] > eng[e]["null_share"]),
            "alert_worthy_units_a_minority_in_every_engine": _all(names, lambda e: eng[e]["alert_worthy_share"] < 0.5),
            "no_engine_drifts_beyond_noise": not any(beyond.values()),
            "some_engine_drifts_beyond_noise": any(beyond.values()),
            "every_engine_drifts_beyond_noise": all(beyond.values()),
            "openai_chatgpt_only_engine_consistent_with_noise": [e for e in names if not beyond[e]] == [OPENAI],
            "most_units_have_no_drift_in_every_engine": _all(names, lambda e: eng[e]["share_all_zero"] > 0.5),
            "highest_alert_worthy_share_engine_is_anthropic_claude": extreme({e: eng[e]["alert_worthy_share"] for e in names})["max"]["engine"] == CLAUDE,
            "lowest_alert_worthy_share_engine_is_openai_chatgpt": extreme({e: eng[e]["alert_worthy_share"] for e in names})["min"]["engine"] == OPENAI,
        },
    }


def _d_a2(a2: dict[str, Any]) -> dict[str, Any]:
    pairs = a2["pairs"]
    line = a2["shared_movement_at_or_above"]
    r = {p: pairs[p]["correlation"] for p in pairs}
    excl = {p: pairs[p]["excludes_zero"] for p in pairs}
    return {
        "correlation": _extreme_keyed(r, "pair"),
        "mean_correlation": a2["mean_correlation"],
        "counts": {
            "pairs": len(pairs),
            "pairs_excluding_zero": sum(excl.values()),
            "pairs_including_zero": len(pairs) - sum(excl.values()),
            "pairs_negative": sum(v < 0 for v in r.values()),
            "pairs_positive": sum(v > 0 for v in r.values()),
        },
        "expected_false_exclusions": len(pairs) * (1 - a2["ci_level"]),
        "excludes_zero": excl,
        "flags": {
            "mean_correlation_below_shared_movement_line": a2["mean_correlation"] < line,
            "no_pair_at_or_above_shared_movement_line": not any(v >= line for v in r.values()),
            "mean_correlation_negative": a2["mean_correlation"] < 0,
            "some_pair_interval_excludes_zero": any(excl.values()),
            "most_pairs_interval_includes_zero": sum(excl.values()) * 2 < len(pairs),
            "every_pair_excluding_zero_is_negative": all(r[p] < 0 for p in pairs if excl[p]),
            "no_pair_excluding_zero_is_positive": not any(r[p] > 0 for p in pairs if excl[p]),
            "most_pairs_negative": sum(v < 0 for v in r.values()) * 2 > len(pairs),
        },
    }


def _d_change_detection(v2: dict[str, Any], d_v2: dict[str, Any], d_a2: dict[str, Any]) -> dict[str, Any]:
    names = ordered(v2["engines"])
    no_drift = d_v2["flags"]["median_company_shows_no_drift_worth_alerting"]
    no_shared = d_a2["flags"]["mean_correlation_below_shared_movement_line"] and d_a2["flags"]["no_pair_at_or_above_shared_movement_line"]
    tail = d_v2["flags"]["some_engine_drifts_beyond_noise"]
    return {
        "counts": {"engines": len(names), "engines_beyond_noise": d_v2["counts"]["engines_beyond_noise"],
                   "pairs": d_a2["counts"]["pairs"]},
        "flags": {
            "median_company_shows_no_drift_worth_alerting": no_drift,
            "no_evidence_of_shared_movement": no_shared,
            "change_detection_product_not_supported_as_pre_specified": no_drift and no_shared,
            "small_tail_beyond_noise_in_some_engine": tail and d_v2["flags"]["alert_worthy_units_a_minority_in_every_engine"],
            "tail_is_a_reason_to_look_further_not_a_product": tail and no_drift,
        },
    }


def _d_a3(a3: dict[str, Any]) -> dict[str, Any]:
    names = ordered(a3["engines"])
    eng = a3["engines"]
    corr = {e: eng[e]["rank_correlation"] for e in names}
    flagged = {e: eng[e]["teardown_flagged"] for e in names}
    return {
        "rank_correlation": extreme(corr),
        "teardown_flagged": extreme(flagged),
        "teardown_share": extreme({e: eng[e]["teardown_share"] for e in names}),
        "tie_only_distortion_max": extreme({e: eng[e]["tie_only_distortion_max"] for e in names}),
        "counts": {"engines": len(names), "teardown_flagged_total": sum(flagged.values()),
                   "engines_with_flagged_companies": sum(v > 0 for v in flagged.values()),
                   "engines_without_flagged_companies": sum(v == 0 for v in flagged.values()),
                   "rank_gap": a3["teardown_rank_gap"]},
        "flags": {
            "rank_correlation_positive_in_every_engine": _all(names, lambda e: corr[e] > 0),
            "highest_rank_correlation_engine_is_google_gemini": extreme(corr)["max"]["engine"] == GEMINI,
            "lowest_rank_correlation_engine_is_anthropic_claude": extreme(corr)["min"]["engine"] == CLAUDE,
            "teardown_flags_some_company_in_some_engine": any(v > 0 for v in flagged.values()),
            "google_ai_mode_flags_no_company": flagged[AI_MODE] == 0,
            "teardown_flags_under_half_of_companies_in_every_engine": _all(names, lambda e: eng[e]["teardown_share"] < 0.5),
            "ties_alone_reach_the_gap_in_no_engine": not any(eng[e]["ties_alone_can_reach_gap"] for e in names),
        },
    }


def _d_o2(o2: dict[str, Any]) -> dict[str, Any]:
    names = ordered(o2["engines"])
    eng = o2["engines"]
    return {
        "difference_by_engine": extreme({e: eng[e]["difference"] for e in names}),
        "paraphrase_mean_by_engine": extreme({e: eng[e]["paraphrase_mean"] for e in names}),
        "rep_mean_by_engine": extreme({e: eng[e]["rep_mean"] for e in names}),
        "counts": {
            "engines": len(names), "pairs": o2["n_pairs"],
            "pairs_paraphrase_lower": sum(eng[e]["n_paraphrase_lower"] for e in names),
            "pairs_rep_lower": sum(eng[e]["n_cells"] - eng[e]["n_paraphrase_lower"] for e in names),
        },
        "flags": {
            "paraphrase_overlap_below_rep_overlap_overall": o2["difference"] < 0,
            "interval_excludes_zero": not o2["interval_crosses_zero"],
            "interval_entirely_below_zero": o2["ci_high"] < 0,
            "paraphrase_overlap_below_rep_overlap_in_every_engine": _all(names, lambda e: eng[e]["difference"] < 0),
            "paraphrase_overlap_below_rep_overlap_in_most_pairs": sum(eng[e]["n_paraphrase_lower"] for e in names) * 2 > o2["n_pairs"],
            "one_paraphrase_count_for_every_intent": len(o2["paraphrases_per_intent"]) == 1,
        },
    }


def _d_x1(x1: dict[str, Any]) -> dict[str, Any]:
    names = ordered(x1["engines"])
    eng = x1["engines"]
    modal = {e: eng[e]["modal_class"] for e in names}
    margin = {e: eng[e]["modal_margin"] for e in names}
    vendor = {e: eng[e]["raw_vendor_share"] for e in names}
    return {
        "modal_margin": extreme(margin),
        "raw_vendor_share": extreme(vendor),
        "review_site_share": extreme({e: eng[e]["review_site_share"] for e in names}),
        "modal_share": extreme({e: eng[e]["modal_share"] for e in names}),
        "distinct_modal_classes": len(set(modal.values())),
        "counts": {"engines": len(names), "engines_with_weak_modal": sum(eng[e]["weak_modal"] for e in names),
                   "n_unclassified": sum(eng[e]["n_unclassified"] for e in names)},
        "flags": {
            "modal_class_identical_across_engines": len(set(modal.values())) == 1,
            "modal_class_differs_across_engines": len(set(modal.values())) > 1,
            "stored_identical_flag_matches_data": x1["identical_stored"] == (len(set(modal.values())) == 1),
            "modal_class_is_vendor_content_in_every_engine": _all(names, lambda e: modal[e].startswith(VENDOR_CLASS_PREFIX)),
            "raw_vendor_share_above_half_in_every_engine": _all(names, lambda e: vendor[e] > 0.5),
            "widest_modal_margin_engine_is_openai_chatgpt": extreme(margin)["max"]["engine"] == OPENAI,
            "narrowest_modal_margin_engine_is_google_ai_mode": extreme(margin)["min"]["engine"] == AI_MODE,
            "weak_modal_only_in_google_ai_mode": [e for e in names if eng[e]["weak_modal"]] == [AI_MODE],
        },
    }


def _d_s4(s4: dict[str, Any]) -> dict[str, Any]:
    names = ordered(s4["engines"])
    eng = s4["engines"]
    n = {e: eng[e]["n_not_fired"] for e in names}
    share = {e: eng[e]["share"] for e in names}
    top = s4["gemini_top_paraphrase"]
    intents = top["intents"]
    unbranded = {i: v["rate"] for i, v in intents.items() if not v["branded"]}
    branded = [v["rate"] for v in intents.values() if v["branded"]]
    lead = extreme(n)["max"]["engine"]
    return {
        "n_not_fired": extreme(n),
        "share": extreme(share),
        "counts": {"engines": len(names), "not_fired_total": sum(n.values()),
                   "engines_that_skipped": sum(v > 0 for v in n.values()),
                   "unbranded_intents": len(unbranded), "intents_with_skips_in_top_paraphrase": sum(v > 0 for v in unbranded.values()),
                   "intents_above_half_in_top_paraphrase": sum(v > 0.5 for v in unbranded.values())},
        "gemini_top_paraphrase": {
            "rate": _extreme_keyed(unbranded, "intent"),
            "mean_rate_unbranded": top["mean_rate_unbranded"],
        },
        "flags": {
            "google_gemini_skips_search_most": lead == GEMINI,
            "google_gemini_ahead_of_every_other_engine": n[GEMINI] > max(v for e, v in n.items() if e != GEMINI),
            "google_ai_mode_never_skips_search": n[AI_MODE] == 0,
            "every_other_engine_skips_search_at_least_once": _all([e for e in names if e != AI_MODE], lambda e: n[e] > 0),
            "no_engine_skips_search_in_most_runs": _all(names, lambda e: share[e] < 0.5),
            "no_search_runs_cite_nothing_in_every_engine_that_searched": _all(names, lambda e: eng[e]["no_search_equals_zero_sources_when_search_ran"]),
            "top_paraphrase_skips_most_in_every_unbranded_intent": _all(list(unbranded), lambda i: intents[i]["rate"] > intents[i]["best_other_paraphrase_rate"]),
            "top_paraphrase_mean_above_every_other_paraphrase": top["mean_rate_unbranded"] > top["best_other_mean_rate_unbranded"],
            "top_paraphrase_skips_in_every_unbranded_intent": all(v > 0 for v in unbranded.values()),
            "top_paraphrase_never_skips_in_branded_intent": bool(branded) and not any(branded),
        },
    }


def _d_method(method: dict[str, Any], s3: dict[str, Any]) -> dict[str, Any]:
    names = ordered(method["engines"])
    planned = method["planned_runs"]["engines"]
    schedule = method["n_intents"] * (method["paraphrases_per_intent"] or 0) * method["runs_per_day"] * method["days"]
    coded = method["coded_runs"]
    window = method["window"]
    return {
        "counts": {
            "engines": len(names), "intents": method["n_intents"], "unbranded_intents": method["n_unbranded_intents"],
            "branded_intents": method["n_branded_intents"], "paraphrases_per_intent": method["paraphrases_per_intent"],
            "runs_per_day": method["runs_per_day"], "days": method["days"], "planned_per_engine": planned[names[0]],
            "planned_total": method["planned_runs"]["total"], "usable_runs": method["usable_runs_total"],
            "runs_not_usable": method["planned_runs"]["total"] - method["usable_runs_total"],
            "coded_runs": coded["total"], "roster": method["roster_size"], "eligibility_floor": method["eligibility_floor"],
            "frame_days_after_window": method["inclusion_frame_days_after_window"],
        },
        "flags": {
            "same_planned_runs_in_every_engine": len(set(planned.values())) == 1,
            "planned_runs_equal_schedule": _all(names, lambda e: planned[e] == schedule),
            "window_days_equal_configured_days": window["days"] == method["days"],
            "one_paraphrase_count_for_every_intent": method["paraphrases_per_intent"] is not None,
            "coded_runs_and_no_mention_runs_make_up_usable_runs": coded["total"] + coded["no_extracted_mention"] == method["usable_runs_total"],
            "inclusion_frame_frozen_after_window_closed": method["inclusion_frame_days_after_window"] > 0,
            "every_engine_has_qualifying_companies": _all(names, lambda e: method["qualifying_companies"][e] > 0),
            "branded_intent_reported_apart": method["n_branded_intents"] > 0,
        },
    }


def derive(result: dict[str, Any]) -> dict[str, Any]:
    """Every number and claim flag the user prose of the remaining sections needs, so the template computes
    nothing. Read from the section blocks of `result`; fractions stay fractions."""
    d_v2, d_a2 = _d_v2(result["v2"]), _d_a2(result["a2"])
    return {
        "s3": _d_s3(result["s3"]),
        "s2": _d_s2(result["s2"], result["s3"]),
        "k1": _d_k1(result["k1"]),
        "o1": _d_o1(result["o1"]),
        "k2": _d_k2(result["k2"]),
        "v2": d_v2,
        "a2": d_a2,
        "change_detection": _d_change_detection(result["v2"], d_v2, d_a2),
        "a3": _d_a3(result["a3"]),
        "o2": _d_o2(result["o2"]),
        "x1": _d_x1(result["x1"]),
        "s4": _d_s4(result["s4"]),
        "ui_calibration": _d_ui_calibration(result["ui_calibration"], result["method"]["window"]["end"]),
        "study": _d_method(result["method"], result["s3"]),
    }


def study_constants(
    schedule: dict[str, int], window: dict[str, Any], frame_frozen_date: dt.date | str
) -> dict[str, Any]:
    """The study constants figures.json does not store: `schedule` {runs_per_day, days} from
    config/experiment.yaml, `window` {start, end, days} from the raw MANIFEST, the inclusion frame's frozen
    date. The gap in days is computed here from the two dates."""
    end = dt.date.fromisoformat(str(window["end"]))
    frozen = frame_frozen_date if isinstance(frame_frozen_date, dt.date) else dt.date.fromisoformat(str(frame_frozen_date))
    return {
        "runs_per_day": schedule["runs_per_day"], "days": schedule["days"],
        "window": {"start": str(window["start"]), "end": str(window["end"]), "days": window["days"]},
        "frame_frozen_date": frozen.isoformat(),
        "frame_days_after_window": (frozen - end).days,
    }


def build_rest_figures(figures: dict[str, Any], study: dict[str, Any]) -> dict[str, Any]:
    """`study` holds what figures.json does not store: runs_per_day, days (config/experiment.yaml), window
    {start, end, days} (raw MANIFEST), frame_frozen_date and frame_days_after_window (inclusion frame)."""
    intent_map = figures["intermediates"]["intent_map"]
    s3, o2 = _s3(figures), _o2(figures)
    result: dict[str, Any] = {
        "figure_id": "rest_business",
        "status": "exploratory_derived",
        "note": NOTE,
        "engine_order": list(ENGINE_ORDER),
        "sections": list(SECTIONS),
        "s3": s3,
        "s2": _s2(figures),
        "k1": _k1(figures),
        "o1": _o1(figures),
        "k2": _k2(figures),
        "v2": _v2(figures),
        "a2": _a2(figures),
        "a3": _a3(figures),
        "o2": o2,
        "x1": _x1(figures),
        "s4": _s4(figures, intent_map),
        "ui_calibration": _ui_calibration(figures),
        "method": _method(figures, study, o2, s3),
    }
    result["derived"] = derive(result)
    return result
