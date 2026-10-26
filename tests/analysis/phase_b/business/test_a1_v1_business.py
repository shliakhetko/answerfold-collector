import json

import pytest

from analysis.phase_b.business import a1_v1_business as ab
from collector import paths

pytestmark = pytest.mark.skipif(not paths.FIGURES_JSON.exists(), reason="frozen figures.json not present")


@pytest.fixture(scope="module")
def frozen():
    return json.loads(paths.FIGURES_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def built(frozen):
    return ab.build_a1_v1_figures(frozen)


def test_labelled_exploratory(built):
    assert built["status"] == "exploratory_derived"


def test_v1_shares_sum_to_one_and_counts_equal_figures_json(built, frozen):
    v1 = built["v1"]
    assert sum(g["share"] for g in v1["groups"].values()) == pytest.approx(1.0)
    notes = frozen["figures"]["V1"]["notes"]
    assert v1["groups"][ab.ENGINE_MORE]["count"] == notes["n_above_line_engine_matters_more"] == 42
    assert v1["groups"][ab.INDUSTRY_MORE]["count"] == notes["n_below_line_industry_matters_more"] == 13
    assert v1["groups"][ab.NEVER_NAMED]["count"] == notes["n_excluded_engine_spread_zero"] == 103
    assert v1["n_companies"] == frozen["figures"]["V1"]["denominator_before_exclusion"]
    assert v1["median_ratio"] == frozen["figures"]["V1"]["value"]
    assert v1["median_ratio_excluding_zero_industry"] == notes["exploratory_median_excluding_zero_industry"]["median"]


def test_v1_split_matches_per_entity_ratios(frozen):
    per = frozen["figures"]["V1"]["notes"]["per_entity"]
    included = [e for e in per if e["ratio"] is not None]
    assert sum(e["ratio"] < 1 for e in included) == 42
    assert sum(e["ratio"] > 1 for e in included) == 13


def test_a1_numbers_equal_figures_json(built, frozen):
    src = frozen["figures"]["A1"]["notes"]["exploratory_top10_vs_not"]
    assert set(built["a1"]["engines"]) == set(src)
    for e, b in built["a1"]["engines"].items():
        assert b["mean_top10"] == src[e]["mean_top10"]
        assert b["mean_other"] == src[e]["mean_not_top10"]
        assert b["n_top10_pairs"] == src[e]["n_top10"]
        assert b["n_other_pairs"] == src[e]["n_not_top10"]
        assert b["ratio"] == pytest.approx(b["mean_top10"] / b["mean_other"])
    assert built["a1"]["serp_days_after_window"] == 11


def test_no_company_names_in_output(built, frozen):
    text = json.dumps(built)
    for e in frozen["figures"]["V1"]["notes"]["per_entity"]:
        assert e["entity_id"] not in text


def test_missing_serp_days_raises(frozen):
    bad = json.loads(json.dumps(frozen))
    bad["figures"]["A1"]["notes"]["serp_collection_limitation"] = "no days here"
    with pytest.raises(ValueError):
        ab.build_a1_v1_figures(bad)


# --- V1 per engine (derived from the rates panel) ---------------------------------------------------------

from types import SimpleNamespace  # noqa: E402


def _cell(entity, intent, engine, rate, qualifies=True):
    return SimpleNamespace(entity_id=entity, intent_id=intent, engine=engine, mention_rate=rate, qualifies=qualifies)


def test_per_engine_rates_mean_over_qualifying_companies_only():
    engines = ("e1", "e2")
    cells = []
    for engine in engines:
        for intent in (1, 2, 3, 4, 5):
            cells.append(_cell("a", intent, engine, 0.2, True))
            cells.append(_cell("b", intent, engine, 0.4 if intent == 1 else 0.0, True))
            cells.append(_cell("c", intent, engine, 0.9, False))
    out = ab.per_engine_rates(cells, engines)
    assert out["n_companies"] == 2  # "c" does not qualify
    from analysis.coding.b3_aggregates import ELIGIBILITY_FLOOR

    assert out["eligibility_floor"] == ELIGIBILITY_FLOOR
    assert out["engines"]["e1"]["general"] == pytest.approx(0.3)
    assert out["engines"]["e2"]["dental"] == pytest.approx(0.1)
    assert set(out["engines"]["e1"]) == {"general", "dental", "legal", "home_services"}


def test_per_engine_rates_only_companies_named_in_general_question():
    engines = ("e1", "e2")
    cells = []
    for engine in engines:
        for intent in (1, 2, 3, 4):
            cells.append(_cell("named_e1_only", intent, engine, (0.4 if engine == "e1" else 0.0) if intent == 1 else 0.1))
            cells.append(_cell("never_named", intent, engine, 0.0 if intent == 1 else 0.5))
    out = ab.per_engine_rates(cells, engines)
    assert out["n_companies"] == 1
    assert out["engines"]["e1"]["general"] == pytest.approx(0.4)
    assert out["engines"]["e2"]["general"] == 0.0
    assert out["engines"]["e1"]["dental"] == pytest.approx(0.1)


def test_per_engine_rates_stops_when_nobody_is_named_in_general_question():
    cells = [_cell("a", i, "e1", 0.0 if i == 1 else 0.2) for i in (1, 2, 3, 4)]
    with pytest.raises(ab.PerEngineRatesError):
        ab.per_engine_rates(cells, ("e1",))


def test_engine_gap_spreads():
    block = {"engines": {
        "e1": {"general": 0.10, "dental": 0.02, "legal": 0.05, "home_services": 0.03},
        "e2": {"general": 0.04, "dental": 0.04, "legal": 0.04, "home_services": 0.06},
    }}
    g = ab.engine_gap(block, ["e1", "e2"])
    assert g["engine_gap"] == pytest.approx(0.06)
    assert (g["general_low_engine"], g["general_high_engine"]) == ("e2", "e1")
    assert g["industry"]["e1"]["range"] == pytest.approx(0.03)
    assert (g["industry"]["e1"]["low_key"], g["industry"]["e1"]["high_key"]) == ("dental", "legal")
    assert g["widest_industry_engine"] == "e1" and g["industry_gap_max"] == pytest.approx(0.03)


def test_per_engine_rates_stops_on_missing_cell():
    cells = [_cell("a", i, "e1", 0.1) for i in (1, 2, 3)]  # no intent 4
    with pytest.raises(ab.PerEngineRatesError):
        ab.per_engine_rates(cells, ("e1",))


def test_per_engine_rates_stops_without_qualifying_companies():
    with pytest.raises(ab.PerEngineRatesError):
        ab.per_engine_rates([_cell("a", 1, "e1", 0.1, False)], ("e1",))


def test_build_adds_per_engine_block_only_when_given(frozen):
    assert "v1_per_engine" not in ab.build_a1_v1_figures(frozen)
    block = {"n_companies": 55, "engines": {}}
    assert ab.build_a1_v1_figures(frozen, block)["v1_per_engine"] == block


def test_build_stops_when_population_differs_from_v1_compared(frozen):
    with pytest.raises(ab.PerEngineRatesError):
        ab.build_a1_v1_figures(frozen, {"n_companies": 157, "engines": {}})


def test_built_json_per_engine_reconciles_with_figures_json(frozen):
    from scripts.results.user_charts import style

    if not paths.BUSINESS_FIGURES_A1_V1_JSON.exists():
        pytest.skip("derived JSON not built")
    built = json.loads(paths.BUSINESS_FIGURES_A1_V1_JSON.read_text(encoding="utf-8"))
    pe = built.get("v1_per_engine")
    if pe is None:
        pytest.skip("derived JSON has no per-engine block")
    assert list(pe["engines"]) == list(style.ENGINE_ORDER)
    # The 55 are the companies V1 compares (42 + 13). Their pooled general rate (mean over engines and
    # companies) equals the stored figures.json intent-1 mean of the included companies.
    dist = frozen["figures"]["V1"]["notes"]["intent1_mention_rate_distribution"]
    pooled = sum(b["general"] for b in pe["engines"].values()) / len(pe["engines"])
    assert pe["n_companies"] == dist["included"]["n"] == frozen["figures"]["V1"]["n_included"]
    assert pooled == pytest.approx(dist["included"]["mean"])


# ---------------------------------------------------------------- derived (prose numbers and claim flags)

def test_derived_extremes_and_flags_equal_figures_json(built, frozen):
    d = ab.derive(built, {"runs_per_day": 3, "days": 7})
    stored = frozen["figures"]["A1"]["notes"]["exploratory_top10_vs_not"]
    top = {e: b["mean_top10"] for e, b in stored.items()}
    assert d["a1"]["top10_rate"]["min"] == {"engine": min(top, key=top.get), "value": min(top.values())}
    assert d["a1"]["top10_rate"]["max"]["value"] == max(top.values())
    assert d["a1"]["n_top10_pairs"]["max"]["value"] == max(b["n_top10"] for b in stored.values())
    assert d["a1"]["n_other_pairs"]["min"]["value"] == min(b["n_not_top10"] for b in stored.values())
    assert d["a1"]["ratio"]["max"]["value"] == pytest.approx(max(b["mean_top10"] / b["mean_not_top10"] for b in stored.values()))
    assert all(d["a1"]["flags"].values())
    notes = frozen["figures"]["V1"]["notes"]
    assert d["v1"]["n_compared"] == notes["n_above_line_engine_matters_more"] + notes["n_below_line_industry_matters_more"]
    assert d["v1"]["flags"]["engine_matters_more_for_most_compared"] is True
    assert d["counts"]["industries"] == 3 and d["counts"]["v1_groups"] == 3
    assert d["study"] == {"runs_per_day": 3, "days": 7}


def test_derived_a1_flags_turn_false():
    result = {
        "a1": {"engines": {
            "e1": {"mean_top10": 0.6, "mean_other": 0.7, "n_top10_pairs": 50, "n_other_pairs": 40, "ratio": 0.9},
            "e2": {"mean_top10": 0.2, "mean_other": 0.1, "n_top10_pairs": 5, "n_other_pairs": 400, "ratio": 2.0},
        }},
        "v1": {"groups": {ab.ENGINE_MORE: {"count": 3}, ab.INDUSTRY_MORE: {"count": 5}, ab.NEVER_NAMED: {"count": 1}}},
    }
    d = ab.derive(result)
    assert d["a1"]["flags"] == {
        "top10_named_more_than_other_in_every_engine": False,
        "top10_rate_below_half_in_every_engine": False,
        "top10_group_smaller_than_other_group": False,
    }
    assert d["v1"]["flags"]["engine_matters_more_for_most_compared"] is False
    assert d["a1"]["ratio"]["min"]["engine"] == "e1" and d["v1"]["n_compared"] == 8
    assert "industry_gap_within_engine_below_engine_gap" not in d["v1"]["flags"]


def test_derived_v1_engine_gap_flag_and_points():
    result = {
        "a1": {"engines": {"e1": {"mean_top10": 0.2, "mean_other": 0.1, "n_top10_pairs": 5, "n_other_pairs": 40, "ratio": 2.0}}},
        "v1": {"groups": {ab.ENGINE_MORE: {"count": 2}, ab.INDUSTRY_MORE: {"count": 1}, ab.NEVER_NAMED: {"count": 1}}},
        "v1_per_engine": {"n_companies": 3, "engines": {
            "e1": {"general": 0.10, "dental": 0.02, "legal": 0.05, "home_services": 0.03},
            "e2": {"general": 0.04, "dental": 0.04, "legal": 0.04, "home_services": 0.06}}},
    }
    v1 = ab.derive(result)["v1"]
    assert v1["engine_gap_points"] == pytest.approx(6.0)
    assert v1["general_rate"]["max"]["engine"] == "e1" and v1["widest_industry_gap"]["engine"] == "e1"
    assert v1["widest_industry_gap"]["points"] == pytest.approx(3.0)
    assert v1["flags"]["industry_gap_within_engine_below_engine_gap"] is True
    assert v1["flags"]["per_engine_population_equals_compared"] is True
    result["v1_per_engine"]["engines"]["e1"]["legal"] = 0.30
    assert ab.derive(result)["v1"]["flags"]["industry_gap_within_engine_below_engine_gap"] is False


@pytest.mark.skipif(not paths.BUSINESS_FIGURES_A1_V1_JSON.exists(), reason="derived JSON not built")
def test_built_json_derived_matches_its_own_blocks_and_study_config():
    from collector.runner.experiment_config import load_experiment_config

    built_json = json.loads(paths.BUSINESS_FIGURES_A1_V1_JSON.read_text(encoding="utf-8"))
    d = built_json["derived"]
    assert d == json.loads(json.dumps(ab.derive(built_json, d["study"])))
    assert all(d["a1"]["flags"].values()) and all(d["v1"]["flags"].values())
    config = load_experiment_config(paths.EXPERIMENT_YAML)
    assert d["study"] == {"runs_per_day": config.reps_per_day, "days": config.days}


def test_derived_industry_range_per_engine_low_high_and_points():
    result = {
        "a1": {"engines": {"e1": {"mean_top10": 0.2, "mean_other": 0.1, "n_top10_pairs": 5, "n_other_pairs": 40, "ratio": None}}},
        "v1": {"groups": {ab.ENGINE_MORE: {"count": 2}, ab.INDUSTRY_MORE: {"count": 1}, ab.NEVER_NAMED: {"count": 1}}},
        "v1_per_engine": {"n_companies": 3, "engines": {
            "e1": {"general": 0.10, "dental": 0.02, "legal": 0.05, "home_services": 0.03},
            "e2": {"general": 0.04, "dental": 0.04, "legal": 0.04, "home_services": 0.06}}},
    }
    d = ab.derive(result)
    assert d["a1"]["ratio"] == {"min": None, "max": None}  # a None ratio does not break the extremes
    r = d["v1"]["industry_range"]
    assert r["e1"]["low"] == {"industry": "dental", "value": 0.02}
    assert r["e1"]["high"] == {"industry": "legal", "value": 0.05}
    assert r["e1"]["points"] == pytest.approx(3.0)
    assert r["e2"]["high"] == {"industry": "home_services", "value": 0.06}
    assert r["e2"]["low"]["industry"] == "dental"  # a tie goes to the first industry


@pytest.mark.skipif(not paths.BUSINESS_FIGURES_A1_V1_JSON.exists(), reason="derived JSON not built")
def test_built_industry_range_matches_stored_per_engine_rates():
    built_json = json.loads(paths.BUSINESS_FIGURES_A1_V1_JSON.read_text(encoding="utf-8"))
    for engine, rates in built_json["v1_per_engine"]["engines"].items():
        block = built_json["derived"]["v1"]["industry_range"][engine]
        industries = {k: rates[k] for k in ab.INDUSTRY_INTENT_IDS}
        assert block["low"]["value"] == min(industries.values()) == industries[block["low"]["industry"]]
        assert block["high"]["value"] == max(industries.values()) == industries[block["high"]["industry"]]
        assert block["points"] == pytest.approx((block["high"]["value"] - block["low"]["value"]) * 100)
