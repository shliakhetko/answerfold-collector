import copy
import datetime
import json

import pytest

from analysis.phase_b.business import rest_business as rb
from analysis.phase_b.business.extremes import ENGINE_ORDER, ordered
from collector import paths

pytestmark = pytest.mark.skipif(not paths.FIGURES_JSON.exists(), reason="frozen figures.json not present")

ENGINES = list(ENGINE_ORDER)
STUDY = rb.study_constants(
    {"runs_per_day": 3, "days": 7}, {"start": "2026-08-20", "end": "2026-08-26", "days": 7}, datetime.date(2026, 9, 5)
)


@pytest.fixture(scope="module")
def frozen():
    return json.loads(paths.FIGURES_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def built(frozen):
    return rb.build_rest_figures(frozen, STUDY)


def _flags(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "flags":
                yield from ((f"{path}.{name}", value) for name, value in v.items())
            else:
                yield from _flags(v, f"{path}.{k}")


def _strings(node):
    if isinstance(node, dict):
        for k, v in node.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v)
    elif isinstance(node, str):
        yield node


def test_labelled_exploratory_and_shape(built):
    assert built["status"] == "exploratory_derived"
    assert built["engine_order"] == ENGINES
    for section in rb.SECTIONS:
        assert set(built[section]) >= {"engines"} or section == "a2"
        assert section in built["derived"]
    assert "change_detection" in built["derived"] and "study" in built["derived"]


def test_every_flag_is_a_bool_and_every_section_has_flags(built):
    flags = dict(_flags(built["derived"]))
    assert flags and all(isinstance(v, bool) for v in flags.values())
    for section in (*rb.SECTIONS, "change_detection", "study"):
        assert built["derived"][section]["flags"]


def test_engines_in_study_order(built):
    for section in ("s3", "s2", "k1", "o1", "v2", "a3", "o2", "x1", "s4"):
        assert list(built[section]["engines"]) == ordered(ENGINES)


def test_s3_s2_equal_figures_json(built, frozen):
    s3, s2 = frozen["figures"]["S3"], frozen["figures"]["S2"]
    for e in ENGINES:
        b = built["s3"]["engines"][e]
        assert b["share"] == s3["value"][e]
        assert b["n_citing_nothing"] / b["n_answers"] == pytest.approx(b["share"])
        r = built["s2"]["engines"][e]
        assert r["restricted"] == s2["value"][e]["restricted"]
        assert r["unrestricted"] == s2["value"][e]["unrestricted"]
        assert r["n_restricted"] == s2["denominator_after_exclusion"][e]["restricted"]
        teardown = s2["notes"]["teardown_by_engine"][e]
        assert r["teardown_under"] == teardown["entities_under_10pct_own_citation"] == len(teardown["entity_ids_under_10pct"])
        assert r["teardown_qualifying"] == teardown["qualifying_entities"]
    d = built["derived"]
    assert d["s3"]["share"]["max"]["engine"] == "google_gemini"
    assert d["s2"]["restricted"]["min"]["engine"] == "openai_chatgpt"
    assert d["s2"]["restricted"]["max"]["value"] == max(s2["value"][e]["restricted"] for e in ENGINES)


def test_k1_o1_equal_figures_json(built, frozen):
    k1, o1 = frozen["figures"]["K1"], frozen["figures"]["O1"]
    for e in ENGINES:
        b = built["k1"]["engines"][e]
        assert b["top_share"] == k1["value"][e] and b["count_to_half"] == k1["notes"]["count_to_50pct"][e]
        assert b["top_n"] == len(k1["notes"]["top_5_ranked"][e])
        h = built["o1"]["engines"][e]
        assert h["head_overlap"] == o1["value"][e]["head_mean_jaccard"]
        assert h["tail_overlap"] == o1["value"][e]["tail_mean_jaccard"]
        assert h["head_label_words"] == "consistent with noise"
        assert h["stable_head_churning_tail_met"] is o1["notes"]["prereg_band_met_by_engine"][e]
    assert built["k1"]["roster_size"] == k1["notes"]["flat_baseline_fixed"]["n"]
    assert built["o1"]["head_size"] == o1["notes"]["head_size"]
    d = built["derived"]
    assert d["k1"]["count_to_half"]["min"] == {"engine": "openai_chatgpt", "value": 8}
    assert d["k1"]["flags"]["highest_top_share_engine_is_openai_chatgpt"]
    assert d["o1"]["flags"]["stable_head_churning_tail_met_in_no_engine"]
    assert d["o1"]["counts"]["engines_head_above_tail"] == sum(
        o1["value"][e]["head_mean_jaccard"] > o1["value"][e]["tail_mean_jaccard"] for e in ENGINES
    )


def test_k2_equals_figures_json_and_bands_read_from_the_stored_note(built, frozen):
    k2 = frozen["figures"]["K2"]
    for e in ENGINES:
        for t in built["k2"]["treatments"]:
            assert built["k2"]["engines"][e][t]["share"] == k2["value"][e][t]
            assert built["k2"]["engines"][e][t]["distinct_domains"] == k2["notes"]["unique_domains"][e][t]
    assert built["k2"]["pooled"]["included"]["domains_to_half"] == k2["notes"]["count_to_50pct"]["pooled"]["included"]
    assert built["k2"]["bands"] == {"high": 0.6, "low": 0.4}
    reading = built["derived"]["k2"]["reading"]["included"]
    assert reading == {"openai_chatgpt": "between", "google_ai_mode": "below", "google_gemini": "between",
                       "anthropic_claude": "between"}
    broken = copy.deepcopy(frozen)
    broken["figures"]["K2"]["notes"]["pre_reg_bands_note"] = "no bands here"
    with pytest.raises(rb.RestBusinessError):
        rb.build_rest_figures(broken, STUDY)


def test_v2_a2_equal_figures_json(built, frozen):
    v2, a2 = frozen["figures"]["V2"], frozen["figures"]["A2"]
    for e in ENGINES:
        b = built["v2"]["engines"][e]
        assert b["median_excess_drift"] == v2["value"][e]
        assert b["alert_worthy_share"] == v2["notes"]["share_alert_worthy_by_engine"][e]
        assert b["null_share"] == v2["verdict"]["noise_comparison_by_engine"][e]["null_share_alert_worthy"]
        assert b["label"] == v2["verdict"]["noise_comparison_by_engine"][e]["label"]
        assert b["n_units"] == v2["n_included"][e]
    assert built["v2"]["engines"]["openai_chatgpt"]["label_words"] == "consistent with noise"
    assert built["a2"]["mean_correlation"] == a2["value"]["mean"]
    assert len(built["a2"]["pairs"]) == len(a2["notes"]["ci_by_pair"])
    for raw, ci in a2["notes"]["ci_by_pair"].items():
        p = built["a2"]["pairs"][raw.replace("|", "__")]
        assert (p["correlation"], p["ci_low"], p["ci_high"], p["excludes_zero"]) == (
            a2["value"][raw], ci["ci_low"], ci["ci_high"], ci["excludes_zero"])
    d = built["derived"]
    assert d["a2"]["counts"]["pairs_excluding_zero"] == len(a2["verdict"]["ci_excludes_zero_pairs"])
    assert d["a2"]["expected_false_exclusions"] == pytest.approx(0.3)
    assert d["v2"]["counts"]["engines_beyond_noise"] == 3
    assert d["v2"]["flags"]["median_excess_drift_zero_in_every_engine"]
    # the stored verdicts say the same thing as the derived claim flags
    assert all(v == "no_product" for v in v2["verdict"]["verdict_by_engine"].values())
    assert d["change_detection"]["flags"]["median_company_shows_no_drift_worth_alerting"]
    assert a2["verdict"]["verdict"] == "no_evidence_of_shared_movement"
    assert d["change_detection"]["flags"]["no_evidence_of_shared_movement"]


def test_a3_o2_x1_s4_equal_figures_json(built, frozen):
    f = frozen["figures"]
    for e in ENGINES:
        assert built["a3"]["engines"][e]["rank_correlation"] == f["A3"]["value"][e]
        assert built["a3"]["engines"][e]["teardown_flagged"] == len(f["A3"]["notes"]["teardown_by_engine"][e])
        assert built["x1"]["engines"][e]["modal_class"] == f["X1"]["notes"]["modal_class_by_engine"][e]
        assert built["s4"]["engines"][e]["n_not_fired"] == f["S4"]["notes"]["n_search_not_fired_by_engine"][e]
    o2 = f["O2"]["notes"]
    assert built["o2"]["difference"] == o2["overall_diff_paraphrase_minus_rep"]
    assert (built["o2"]["ci_low"], built["o2"]["ci_high"]) == (o2["bootstrap"]["ci_low"], o2["bootstrap"]["ci_high"])
    assert built["o2"]["difference"] == pytest.approx(built["o2"]["paraphrase_mean"] - built["o2"]["rep_mean"])
    assert built["a3"]["teardown_rank_gap"] == f["A3"]["notes"]["teardown_rank_gap"]
    top = built["s4"]["gemini_top_paraphrase"]
    for i, cell in top["intents"].items():
        assert cell["rate"] == f["S4"]["value"]["google_gemini"][f"i{i}-p{top['paraphrase_id']}"]
    d = built["derived"]
    assert d["s4"]["flags"]["google_gemini_skips_search_most"]
    assert d["s4"]["n_not_fired"]["max"] == {"engine": "google_gemini", "value": 109}
    assert d["x1"]["flags"]["modal_class_differs_across_engines"]


def test_method_counts(built, frozen):
    m, study = built["method"], built["derived"]["study"]
    assert m["window"] == {"start": "2026-08-20", "end": "2026-08-26", "days": 7}
    assert m["inclusion_frame_days_after_window"] == 10 and m["inclusion_frame_frozen"] == "2026-09-05"
    assert m["roster_size"] == frozen["figures"]["K1"]["notes"]["flat_baseline_fixed"]["n"]
    assert m["coded_runs"]["total"] == 2790
    assert m["usable_runs_total"] == frozen["intermediates"]["citation_panel"]["usable_runs_total"]
    assert m["n_unbranded_intents"] == 6 and m["paraphrases_per_intent"] == 5
    assert all(v for k, v in study["flags"].items())
    assert study["counts"]["planned_total"] == sum(m["planned_runs"]["engines"].values())


def test_study_constants_gap_in_days():
    out = rb.study_constants({"runs_per_day": 2, "days": 4}, {"start": "2026-01-01", "end": "2026-01-04", "days": 4},
                             "2026-01-10")
    assert out["frame_days_after_window"] == 6 and out["frame_frozen_date"] == "2026-01-10"


def test_flags_turn_false_when_the_data_stops_saying_it(frozen):
    changed = copy.deepcopy(frozen)
    f = changed["figures"]
    f["V2"]["verdict"]["noise_comparison_by_engine"]["openai_chatgpt"]["label"] = "real"
    f["S4"]["notes"]["n_search_not_fired_by_engine"]["google_ai_mode"] = 500
    f["S3"]["value"]["google_ai_mode"] = 0.9
    f["X1"]["notes"]["modal_class_by_engine"]["google_ai_mode"] = "vendor listicle"
    f["X1"]["notes"]["modal_class_by_engine"]["openai_chatgpt"] = "vendor listicle"
    f["X1"]["value"]["google_ai_mode"]["vendor listicle"] = 0.5
    f["X1"]["value"]["openai_chatgpt"]["vendor listicle"] = 0.5
    f["X1"]["notes"]["modal_class_identical_across_engines"] = True
    d = rb.build_rest_figures(changed, STUDY)["derived"]
    assert d["v2"]["flags"]["every_engine_drifts_beyond_noise"]
    assert not d["v2"]["flags"]["openai_chatgpt_only_engine_consistent_with_noise"]
    assert not d["s4"]["flags"]["google_ai_mode_never_skips_search"]
    assert not d["s3"]["flags"]["highest_share_engine_is_google_gemini"]
    assert d["x1"]["flags"]["modal_class_identical_across_engines"]


def test_no_company_names_or_domains(built, frozen):
    f = frozen["figures"]
    forbidden = set()
    for e in ENGINES:
        forbidden |= set(f["S2"]["notes"]["teardown_by_engine"][e]["entity_ids_under_10pct"])
        forbidden |= {d["entity_id"] for d in f["S2"]["notes"]["teardown_by_engine"][e]["qualifying_entity_details"]}
        forbidden |= {d["entity_id"] for d in f["K1"]["notes"]["top_5_ranked"][e]}
        forbidden |= {d["entity_id"] for d in f["A3"]["notes"]["teardown_by_engine"][e]}
        forbidden |= {d["entity_id"] for d in f["A3"]["notes"]["scatter_points_by_engine"][e]}
        for day in f["O1"]["notes"]["head_sets_by_day"][e].values():
            forbidden |= set(day)
        for lst in f["K2"]["notes"]["top_10_ranked"][e].values():
            forbidden |= {d["domain"] for d in lst}
    forbidden |= set(f["CLAUDE_PROBATION"]["notes"]["set_a"]) | set(f["CLAUDE_PROBATION"]["notes"]["set_b"])
    assert len(forbidden) > 100
    strings = set(_strings(built))
    assert not (forbidden & strings)
    assert "entity_id" not in json.dumps(built)


def test_deterministic(frozen, built):
    again = rb.build_rest_figures(copy.deepcopy(frozen), STUDY)
    assert json.dumps(again, sort_keys=False) == json.dumps(built, sort_keys=False)


def test_runner_writes_the_same_json_read_only(tmp_path, built):
    from scripts.results import run_business_figures_rest as runner

    data = paths.REPO_ROOT / "data"
    if not paths.RAW_MANIFEST.exists() or not data.is_dir():
        pytest.skip("frozen data directory not present")
    before = paths.FIGURES_JSON.read_bytes()
    out = tmp_path / "rest.json"
    assert runner.main(["--data-root", str(data), "--output", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads(json.dumps(built))
    assert paths.FIGURES_JSON.read_bytes() == before


def test_ui_calibration_equals_figures_json_and_totals_add_up(built, frozen):
    stored = frozen["figures"]["UI_CALIBRATION"]["value"]
    block = built["ui_calibration"]["engines"]
    assert list(block) == [e for e in ENGINES if e in stored]
    for e, cells in stored.items():
        assert list(block[e]["prompts"]) == ["1", "2", "3", "4", "5"]
        for pid, cell in cells.items():
            got = block[e]["prompts"][pid[1]]
            assert got["named_in_ui"] == cell["companies_named_in_ui"]
            assert got["also_named_in_api"] == cell["of_which_also_named_in_api"]
            assert got["ui_only"] == len(cell["ui_only_companies"])
    assert block["google_gemini"]["prompts"]["1"]["named_in_ui"] == 16
    assert block["google_gemini"]["prompts"]["1"]["also_named_in_api"] == 15
    assert block["openai_chatgpt"]["prompts"]["2"]["named_in_ui"] == 11
    assert block["openai_chatgpt"]["prompts"]["2"]["also_named_in_api"] == 6
    d = built["derived"]["ui_calibration"]
    assert d["overall"]["ui_runs"] == 30 and d["counts"]["engines"] == 2 and d["counts"]["prompts"] == 5
    for key in ("named_in_ui", "also_named_in_api", "ui_only"):
        assert d["overall"][key] == sum(t[key] for t in d["totals"].values())
    assert all(t["named_in_ui"] == t["also_named_in_api"] + t["ui_only"] for t in d["totals"].values())
    assert d["counts"]["ui_days_after_window"] == 3 and all(d["flags"].values())
    assert "ui_calibration" in built["sections"]


def test_ui_calibration_flags_turn_false_and_hold_no_company(frozen):
    bad = copy.deepcopy(frozen)
    cells = bad["figures"]["UI_CALIBRATION"]["value"]["google_gemini"]
    for cell in cells.values():
        cell["of_which_also_named_in_api"] = 0
        cell["ui_only_companies"] = []
    flags = rb.build_rest_figures(bad, STUDY)["derived"]["ui_calibration"]["flags"]
    assert not flags["most_ui_named_companies_also_named_in_api_in_every_engine"]
    assert not flags["every_engine_has_ui_named_companies_missing_from_api"]
    bad["figures"]["UI_CALIBRATION"]["notes"]["caveats"] = "no date"
    with pytest.raises(rb.RestBusinessError):
        rb.build_rest_figures(bad, STUDY)
