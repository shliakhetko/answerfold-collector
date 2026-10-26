"""Business figures for A1 (Google top-10 vs everyone else) and V1 (engine vs industry).

Exploratory and derived, built only from the frozen data/04_results/figures.json.
Nothing is recomputed from raw data and nothing here feeds or changes figures.json.

A1: per engine, the mean mention rate of companies on Google's first page for
the intent vs all other company-intent pairs, with the group sizes. The stored
exploratory block is copied as is; only the ratio is derived.

V1: one split of the study's companies into "engine matters more" (the mention
rate swings more across engines than across industries), "industry matters
more", and "never named in the general question" (excluded from the V1 median
because their engine spread is zero). Aggregate only, no company names.

V1 per engine: for each engine, the mean mention rate for the general question
(intent 1) and for dental, legal and home services (intents 2, 3, 4), over the
companies of the V1 primary set (qualifying in at least one engine) that were named in
the general question on at least one engine: the companies V1 can compare. It is read
only from the frozen rates panel (build_rates_panel), the same source as
analysis/phase_b/figures/v1.py. If the panel does not hold a rate for every
company, engine and intent, nothing is produced (PerEngineRatesError).
"""
from __future__ import annotations

import re
import statistics
from typing import Any

from analysis.coding.b3_aggregates import ELIGIBILITY_FLOOR
from analysis.phase_b.business.extremes import extreme, ordered

ENGINE_MORE = "engine_matters_more"
INDUSTRY_MORE = "industry_matters_more"
NEVER_NAMED = "never_named_general"
V1_GROUPS = (ENGINE_MORE, INDUSTRY_MORE, NEVER_NAMED)

GENERAL_INTENT_ID = 1
INDUSTRY_INTENT_IDS = {"dental": 2, "legal": 3, "home_services": 4}
# Depth of the Google results page the A1 flag is built on: the stored field is `in_google_top10`
# (analysis/coding/b3b_serp_derivation.py, "raw top 10 organic URLs").
SERP_TOP_N = 10
INTENT_KEYS = ("general", *INDUSTRY_INTENT_IDS)
_INTENT_ID_BY_KEY = {"general": GENERAL_INTENT_ID, **INDUSTRY_INTENT_IDS}


class PerEngineRatesError(Exception):
    """The frozen rates panel cannot give a rate for every engine and intent."""


NOTE = (
    "Exploratory and derived; separate from figures.json. Every number is read from figures.json except the "
    "A1 ratio and the V1 shares, which are derived from stored counts; the per-engine V1 rates are derived "
    "read-only from the frozen rates panel."
)


def _a1(figures: dict[str, Any]) -> dict[str, Any]:
    notes = figures["figures"]["A1"]["notes"]
    engines = {}
    for engine, b in notes["exploratory_top10_vs_not"].items():
        engines[engine] = {
            "n_top10_pairs": b["n_top10"],
            "n_other_pairs": b["n_not_top10"],
            "mean_top10": b["mean_top10"],
            "mean_other": b["mean_not_top10"],
            "ratio": b["mean_top10"] / b["mean_not_top10"] if b["mean_not_top10"] else None,
        }
    m = re.search(r"\((\d+) days after", notes["serp_collection_limitation"])
    if m is None:
        raise ValueError("A1 serp_collection_limitation no longer states the days after the window")
    return {"engines": engines, "serp_days_after_window": int(m.group(1))}


def _v1(figures: dict[str, Any]) -> dict[str, Any]:
    v1 = figures["figures"]["V1"]
    notes = v1["notes"]
    counts = {
        ENGINE_MORE: notes["n_above_line_engine_matters_more"],
        INDUSTRY_MORE: notes["n_below_line_industry_matters_more"],
        NEVER_NAMED: notes["excluded_because_zero_mentions_intent1_all_engines"]["n_all_zero"],
    }
    total = sum(counts.values())
    return {
        "n_companies": total,
        "n_on_line": notes["n_on_line"],
        "groups": {k: {"count": c, "share": c / total} for k, c in counts.items()},
        "median_ratio": v1["value"],
        "median_ratio_excluding_zero_industry": notes["exploratory_median_excluding_zero_industry"]["median"],
        "n_zero_industry_in_median": notes["zero_ratio_median_check"]["n_all_zero_industry"],
    }


def per_engine_rates(cells: Any, engines: tuple[str, ...]) -> dict[str, Any]:
    """Mean mention rate per engine and intent over the V1 primary companies (qualifying in at least one
    engine) that were named in the general question on at least one engine (general mention rate above zero
    in some engine). `cells` are RateCell-like rows (entity_id, intent_id, engine, mention_rate, qualifies)."""
    cells = list(cells)
    qualifying = {c.entity_id for c in cells if c.qualifies}
    if not qualifying:
        raise PerEngineRatesError("no qualifying companies in the rates panel")
    by_key = {(c.entity_id, c.intent_id, c.engine): c.mention_rate for c in cells}
    companies = {
        e for e in qualifying
        if any((by_key.get((e, GENERAL_INTENT_ID, engine)) or 0) > 0 for engine in engines)
    }
    if not companies:
        raise PerEngineRatesError("no qualifying company is named in the general question on any engine")
    out: dict[str, Any] = {}
    for engine in engines:
        block: dict[str, float] = {}
        for key in INTENT_KEYS:
            intent_id = _INTENT_ID_BY_KEY[key]
            values = [by_key.get((e, intent_id, engine)) for e in sorted(companies)]
            if any(v is None for v in values):
                raise PerEngineRatesError(f"rates panel has no rate for every company: engine={engine} intent={intent_id}")
            block[key] = statistics.fmean(values)
        out[engine] = block
    return {
        "population": "qualifying companies named in the general question on at least one engine",
        "n_companies": len(companies),
        "eligibility_floor": ELIGIBILITY_FLOOR,
        "engines": out,
    }


def engine_gap(block: dict[str, Any], engines: tuple[str, ...] | list[str]) -> dict[str, Any]:
    """Descriptive spreads of the per-engine block, so the chart and the text use one source: the gap between
    engines on the general question, and per engine the lowest and highest industry rate."""
    rates = block["engines"]
    general = {e: rates[e]["general"] for e in engines}
    industries = {}
    for e in engines:
        vals = {k: rates[e][k] for k in INDUSTRY_INTENT_IDS}
        lo_k, hi_k = min(vals, key=vals.get), max(vals, key=vals.get)
        industries[e] = {"low_key": lo_k, "low": vals[lo_k], "high_key": hi_k, "high": vals[hi_k],
                         "range": vals[hi_k] - vals[lo_k]}
    return {
        "general_low_engine": min(general, key=general.get),
        "general_high_engine": max(general, key=general.get),
        "engine_gap": max(general.values()) - min(general.values()),
        "industry": industries,
        "widest_industry_engine": max(industries, key=lambda e: industries[e]["range"]),
        "industry_gap_max": max(v["range"] for v in industries.values()),
    }


def compute_per_engine_rates(inputs: Any, engines: tuple[str, ...]) -> dict[str, Any]:
    from analysis.phase_b.rates_panel import build_rates_panel  # heavy import, only when reading frozen inputs

    cells, _report = build_rates_panel(inputs)
    seen = {c.engine for c in cells}
    missing = [e for e in engines if e not in seen]
    if missing:
        raise PerEngineRatesError(f"rates panel has no cells for engines: {missing}")
    return per_engine_rates(cells, engines)


def derive(result: dict[str, Any], study: dict[str, int] | None = None) -> dict[str, Any]:
    """Every number and claim flag the user prose of A1 and V1 needs, so the template computes nothing.
    Read from the blocks of `result` (a1, v1, v1_per_engine); fractions stay fractions, `*_points` are
    percentage points (fraction times 100)."""
    a1, v1 = result["a1"]["engines"], result["v1"]
    engines = ordered(a1)
    top = {e: a1[e]["mean_top10"] for e in engines}
    rest = {e: a1[e]["mean_other"] for e in engines}
    n_top = {e: a1[e]["n_top10_pairs"] for e in engines}
    n_rest = {e: a1[e]["n_other_pairs"] for e in engines}
    groups = v1["groups"]
    compared = groups[ENGINE_MORE]["count"] + groups[INDUSTRY_MORE]["count"]
    derived: dict[str, Any] = {
        "counts": {"a1_engines": len(engines), "v1_groups": len(groups), "industries": len(INDUSTRY_INTENT_IDS),
                   "serp_top_n": SERP_TOP_N},
        "a1": {
            "top10_rate": extreme(top),
            "other_rate": extreme(rest),
            "ratio": extreme({e: a1[e]["ratio"] for e in engines}),
            "n_top10_pairs": extreme(n_top),
            "n_other_pairs": extreme(n_rest),
            "flags": {
                "top10_named_more_than_other_in_every_engine": all(top[e] > rest[e] for e in engines),
                "top10_rate_below_half_in_every_engine": all(top[e] < 0.5 for e in engines),
                "top10_group_smaller_than_other_group": max(n_top.values()) < min(n_rest.values()),
            },
        },
        "v1": {
            "n_compared": compared,
            "flags": {"engine_matters_more_for_most_compared": groups[ENGINE_MORE]["count"] > groups[INDUSTRY_MORE]["count"]},
        },
    }
    per_engine = result.get("v1_per_engine")
    if per_engine is not None and per_engine["engines"]:
        pe_engines = ordered(per_engine["engines"])
        gap = engine_gap(per_engine, pe_engines)
        general = {e: per_engine["engines"][e]["general"] for e in pe_engines}
        industry = gap["industry"]
        derived["v1"].update({
            "general_rate": extreme(general),
            "engine_gap": gap["engine_gap"],
            "engine_gap_points": gap["engine_gap"] * 100,
            "industry_range_points": {e: industry[e]["range"] * 100 for e in pe_engines},
            "industry_range": {
                e: {"low": {"industry": industry[e]["low_key"], "value": industry[e]["low"]},
                    "high": {"industry": industry[e]["high_key"], "value": industry[e]["high"]},
                    "points": industry[e]["range"] * 100}
                for e in pe_engines
            },
            "widest_industry_gap": {"engine": gap["widest_industry_engine"], "value": gap["industry_gap_max"],
                                    "points": gap["industry_gap_max"] * 100},
            "n_companies": per_engine["n_companies"],
        })
        derived["v1"]["flags"]["industry_gap_within_engine_below_engine_gap"] = gap["industry_gap_max"] < gap["engine_gap"]
        derived["v1"]["flags"]["per_engine_population_equals_compared"] = per_engine["n_companies"] == compared
    if study is not None:
        derived["study"] = dict(study)
    return derived


def build_a1_v1_figures(
    figures: dict[str, Any], per_engine: dict[str, Any] | None = None, study: dict[str, int] | None = None
) -> dict[str, Any]:
    result = {
        "figure_id": "A1_V1_business",
        "status": "exploratory_derived",
        "note": NOTE,
        "v1_groups": list(V1_GROUPS),
        "a1": _a1(figures),
        "v1": _v1(figures),
    }
    if per_engine is not None:
        compared = result["v1"]["groups"][ENGINE_MORE]["count"] + result["v1"]["groups"][INDUSTRY_MORE]["count"]
        if per_engine["n_companies"] != compared:
            raise PerEngineRatesError(
                f"per-engine population ({per_engine['n_companies']}) differs from the companies V1 compares ({compared})"
            )
        result["v1_per_engine"] = per_engine
    result["derived"] = derive(result, study)
    return result
