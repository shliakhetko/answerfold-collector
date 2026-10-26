"""Derived blocks for the public report (analysis/phase_b/business/public_business.py)."""
import json

import pytest

from analysis.phase_b.business import public_business as pb
from analysis.phase_b.business import source_classes as sc
from analysis.phase_b.business.extremes import ENGINE_ORDER
from analysis.phase_b.reading import read_frozen_inputs
from collector import paths

OPENAI, AI_MODE, GEMINI, CLAUDE = ENGINE_ORDER
NEEDED = (paths.FIGURES_JSON, paths.BUSINESS_FIGURES_JSON, paths.BUSINESS_FIGURES_REST_JSON)
DATA = paths.REPO_ROOT / "data"

needs_data = pytest.mark.skipif(
    not all(p.exists() for p in NEEDED) or not paths.RAW_MANIFEST.exists() or not DATA.is_dir(),
    reason="frozen data or built business figures not present",
)


# --- pure helpers ------------------------------------------------------------------------------------

def test_gemini_queries_read_the_first_candidate_and_tolerate_missing_metadata():
    resp = {"candidates": [{"grounding_metadata": {"web_search_queries": ["a", "b"]}}]}
    assert pb.gemini_queries(resp) == ["a", "b"]
    assert pb.gemini_queries({"candidates": [{"grounding_metadata": None}]}) == []
    assert pb.gemini_queries({"candidates": None}) == []
    assert pb.gemini_queries({}) == []


def test_chatgpt_calls_keep_query_groups_and_skip_open_page_actions():
    resp = {"output": [
        {"type": "web_search_call", "action": {"type": "search", "queries": ["q1", "site:x.com q2"]}},
        {"type": "web_search_call", "action": {"type": "open_page"}},
        {"type": "message"},
        {"type": "web_search_call", "action": {"type": "search", "query": "q3"}},
    ]}
    assert pb.chatgpt_search_calls(resp) == [["q1", "site:x.com q2"], ["q3"]]
    assert pb._queries_for(OPENAI, resp) == ["q1", "site:x.com q2", "q3"]


def test_claude_queries_count_only_web_search_server_tool_use():
    resp = {"content": [
        {"type": "server_tool_use", "name": "web_search", "input": {"query": "one"}},
        {"type": "server_tool_use", "name": "web_fetch", "input": {"url": "u"}},
        {"type": "text", "text": "t"},
        {"type": "server_tool_use", "name": "web_search", "input": {"query": "two"}},
    ]}
    assert pb.claude_queries(resp) == ["one", "two"]


def test_ai_mode_has_no_stored_queries():
    assert pb._queries_for(AI_MODE, {"text_blocks": []}) is None


def test_aggregated_video_flag_marks_youtube_only():
    assert pb._is_aggregated_video("https://youtube.com/watch")
    assert pb._is_aggregated_video("https://youtu.be/abc")
    assert not pb._is_aggregated_video("https://smith.ai/ai-receptionist")


def test_public_class_keys_cover_the_named_classes_and_nothing_else():
    assert set(pb.PUBLIC_CLASS_KEYS) <= set(sc.DISPLAY_CLASSES)
    assert set(pb.PUBLIC_CLASS_KEYS.values()) == {
        "vendor_published", "review_sites", "news_and_press", "forums", "video_and_social"}


# --- the built block ------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stored():
    return [json.loads(p.read_text(encoding="utf-8")) for p in NEEDED]


@pytest.fixture(scope="module")
def built(stored):
    figures, s1, rest = stored
    inputs = read_frozen_inputs(paths.REPO_ROOT, DATA)
    return pb.build_public_figures(inputs, figures, rest, s1)


@needs_data
def test_labelled_exploratory_and_all_engines_present(built):
    assert built["status"] == "exploratory_derived"
    assert built["engine_order"] == list(ENGINE_ORDER)
    for block in ("source_classes", "urls", "non_vendor_domains", "vendor_domains"):
        assert list(built[block]["engines"]) == list(ENGINE_ORDER)


@needs_data
def test_source_class_counts_equal_s1_and_shares_use_the_classified_denominator(built, stored):
    s1 = stored[1]
    for e, b in built["source_classes"]["engines"].items():
        assert b["n_classified"] == s1["engines"][e]["classified_citations"]
        assert b["classes"]["vendor_published"]["count"] == s1["engines"][e]["vendor_published"]["count"]
        assert sum(c["count"] for c in b["classes"].values()) == b["n_classified"]
        assert sum(c["count"] for c in b["display_classes"].values()) == b["n_classified"]
        for cell in list(b["classes"].values()) + list(b["domains"].values()):
            assert cell["total"] == b["n_classified"] and cell["share"] == pytest.approx(cell["count"] / b["n_classified"])


@needs_data
def test_youtube_and_reddit_are_domains_not_classes(built):
    for b in built["source_classes"]["engines"].values():
        assert "youtube_com" not in b["classes"] and "reddit_com" not in b["classes"]
        assert b["domains"]["youtube_com"]["count"] <= b["classes"]["video_and_social"]["count"]
        assert b["domains"]["reddit_com"]["count"] <= b["classes"]["forums"]["count"]
    src = built["source_classes"]["engines"]
    assert src[OPENAI]["domains"]["youtube_com"]["count"] == src[CLAUDE]["domains"]["youtube_com"]["count"] == 0
    assert src[GEMINI]["domains"]["youtube_com"]["count"] > src[AI_MODE]["domains"]["youtube_com"]["count"] > 0


@needs_data
def test_unique_url_vendor_share_equals_s1_note_and_top_urls_are_ranked(built, stored):
    note = stored[0]["figures"]["S1"]["notes"]["unique_url_vendor_share_by_engine"]
    for e, b in built["urls"]["engines"].items():
        assert b["unique_vendor_urls"]["share"] == pytest.approx(note[e])
        assert b["vendor_citations"]["share"] == pytest.approx(stored[1]["engines"][e]["vendor_published"]["share"])
        counts = [t["citations"] for t in b["top_urls"]]
        assert len(counts) == 5 and counts == sorted(counts, reverse=True)
        assert all(t["distinct_runs"] <= t["citations"] for t in b["top_urls"])


@needs_data
def test_youtube_watch_entry_is_flagged_as_aggregated_video(built):
    urls = built["urls"]["engines"]
    for e in (AI_MODE, GEMINI):
        assert urls[e]["top_urls"][0]["url"].endswith("youtube.com/watch")
        assert urls[e]["top_urls"][0]["is_aggregated_video"] is True
    for e in (OPENAI, CLAUDE):
        assert not any(t["is_aggregated_video"] for t in urls[e]["top_urls"])
    assert "collapses" in built["urls"]["video_note"]


@needs_data
def test_vendor_page_types_match_x1(built, stored):
    x1 = stored[2]["x1"]["engines"]
    for e, b in built["vendor_page_types"]["engines"].items():
        assert b["modal_class"] == x1[e]["modal_class"] and b["modal_share"] == x1[e]["modal_share"]
    flags = built["vendor_page_types"]["flags"]
    assert all(flags.values())


@needs_data
def test_non_vendor_domains_exclude_vendor_published_and_are_ranked(built):
    for e, b in built["non_vendor_domains"]["engines"].items():
        assert len(b["top_domains"]) == 10
        counts = [d["count"] for d in b["top_domains"]]
        assert counts == sorted(counts, reverse=True)
        assert all(d["display_class"] != sc.VENDOR for d in b["top_domains"])
        assert all(d["total"] == b["n_classified"] for d in b["top_domains"])
        assert sum(counts) <= b["n_non_vendor"]
    # S1 vendor share is the complement of the non-vendor citations
    for e, b in built["non_vendor_domains"]["engines"].items():
        assert b["n_non_vendor"] + built["source_classes"]["engines"][e]["classes"]["vendor_published"]["count"] == b["n_classified"]


@needs_data
def test_three_shades_sum_to_the_domain_total_and_match_s1(built, stored):
    s1 = stored[1]
    for e, b in built["vendor_domains"]["engines"].items():
        s1_top = s1["engines"][e]["vendor_published"]["top_domains"]
        assert [(d["domain"], d["count"]) for d in b["top_domains"]] == [(d["domain"], d["count"]) for d in s1_top]
        for d, ref in zip(b["top_domains"], s1_top):
            assert d["named_in_scope"] + d["out_of_scope_vendor"] + d["other_vendor_not_named"] == d["count"]
            assert d["named_in_scope"] == ref["self_reference_count"]
            assert sum(d["other_vendor_split"].values()) == d["other_vendor_not_named"]
            assert d["out_of_scope_named_in_answer_count"] <= d["out_of_scope_vendor"]


@needs_data
def test_each_domain_records_its_entity_and_scope_so_a_named_company_is_not_shown_as_never_named(built):
    seen_out_of_scope = 0
    for b in built["vendor_domains"]["engines"].values():
        for d in b["top_domains"]:
            assert d["entity_id"] is not None and d["entity_name"]
            assert d["entity_in_scope"] == (d["entity_meets_frame"] == "in")
            if d["entity_in_scope"]:
                assert d["out_of_scope_vendor"] == 0
            else:
                seen_out_of_scope += 1
                assert d["named_in_scope"] == 0 and d["other_vendor_not_named"] == 0
                assert d["entity_runs_naming_it"]["total"] > 0
    assert seen_out_of_scope > 0
    chatgpt = {d["domain"]: d for d in built["vendor_domains"]["engines"][OPENAI]["top_domains"]}
    # an out-of-scope company can still be named in the answers that cite it; the data records how often
    assert chatgpt["getweave.com"]["out_of_scope_named_in_answer_count"] > 0


@needs_data
def test_alias_entities_do_not_take_over_a_domain(built):
    chatgpt = {d["domain"]: d for d in built["vendor_domains"]["engines"][OPENAI]["top_domains"]}
    assert chatgpt["smith.ai"]["entity_id"] == "smith-ai"
    assert chatgpt["ringcentral.com"]["entity_id"] == "ringcentral"
    assert len(chatgpt["ruby.com"]["entity_ids"]) > 1 and chatgpt["ruby.com"]["entity_id"] == "ruby"


@needs_data
def test_search_blocks_and_examples(built):
    s = built["search"]["engines"]
    assert s[AI_MODE]["available"] is False and "queries" in s[AI_MODE]["reason"]
    for e in (OPENAI, GEMINI, CLAUDE):
        assert s[e]["available"] is True
        assert s[e]["mean_queries_per_answer"] == pytest.approx(s[e]["n_queries"] / s[e]["n_answers"])
    site = s[OPENAI]["site_restricted_queries"]
    assert 0 < site["count"] < site["total"] == s[OPENAI]["n_queries"]
    ex = built["search"]["examples"]["engines"]
    assert ex[GEMINI]["run_id"] == "google_gemini-i1-p1-d1-r1" and ex[OPENAI]["run_id"] == "openai_chatgpt-i1-p1-d1-r1"
    assert ex[GEMINI]["question"] == ex[OPENAI]["question"] == "best AI receptionist"
    assert ex[GEMINI]["queries"] and ex[OPENAI]["queries"] == [q for g in ex[OPENAI]["query_groups"] for q in g]
    assert any("site:" in q for q in ex[OPENAI]["queries"])


@needs_data
def test_wording3_is_verbatim_from_prompts_csv_and_counts_match_s4(built, stored):
    import csv

    w = built["wording3"]
    with paths.PROMPTS_CSV.open(newline="", encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if r["intent_id"] == "1" and r["paraphrase_id"] == "3")
    assert w["prompt_text"] == row["prompt_text"] == "is there any way to stop missing calls for our company, what tool should we use?"
    s4 = stored[2]["s4"]
    assert w["gemini_no_search_runs"] == s4["engines"][GEMINI]["n_not_fired"]
    assert w["gemini_no_search_runs_from_wording3"]["count"] + w["other_wordings_no_search_runs"] == w["gemini_no_search_runs"]
    assert w["mean_unbranded_rate"] == pytest.approx(s4["gemini_top_paraphrase"]["mean_rate_unbranded"])
    assert w["wording3_question1_runs_that_skip_search"]["total"] == 21


@needs_data
def test_k1_top5_names_shares_and_newly_in_frame(built, stored):
    k1 = stored[0]["figures"]["K1"]
    for e, b in built["k1"]["engines"].items():
        assert len(b["top5"]) == 5 and b["count_to_half"] == k1["notes"]["count_to_50pct"][e]
        assert b["top5_share"] == k1["value"][e] == pytest.approx(sum(t["share"] for t in b["top5"]))
        assert all(t["mentions_total"] == b["mentions_total"] for t in b["top5"])
        assert [t["rank"] for t in b["top5"]] == [1, 2, 3, 4, 5]
        assert all(t["name"] for t in b["top5"])
    flags = built["k1"]["flags"]
    assert built["k1"]["newly_in_frame_after_relabel"] == ["retell-ai", "dograh", "pipecat"]
    assert flags["retell_ai_in_a_top5"] is True and flags["dograh_in_a_top5"] is False and flags["pipecat_in_a_top5"] is False
    assert built["k1"]["engines"][GEMINI]["newly_in_frame_in_top5"] == ["retell-ai"]
    assert built["k1"]["engines"][OPENAI]["newly_in_frame_in_top5"] == []


@needs_data
def test_models_and_study_counts(built, stored):
    rest = stored[2]
    for e, b in built["models"]["engines"].items():
        assert sum(b["model_returned"].values()) == b["n_usable"]
    assert set(built["models"]["engines"][GEMINI]["model_returned"]) and built["models"]["engines"][GEMINI]["n_usable"] == 734
    st = built["study"]
    assert st["planned_runs"] == 2940 and st["usable_runs"] == 2939 and st["runs_not_usable"] == 1
    assert st["questions"] * st["wordings_per_question"] * st["runs_per_day"] * st["days"] == st["schedule_product"] == 735
    assert st["schedule_matches_planned_per_engine"] is True and st["runs_per_question_wording_day_values"] == [3]
    assert st["qualifying_companies_by_engine"] == rest["method"]["qualifying_companies"]
    assert st["eligibility_floor_mentions"] == 5
    assert st["in_scope_companies_in_registry"] == 418 and st["in_scope_companies_frozen_baseline"] == 415
    assert st["registry_matches_frozen_baseline"] is False
    k1_out = stored[0]["figures"]["K1"]["notes"]["out_of_frame_named_by_engine"]
    assert st["out_of_scope_entities_named_by_engine"] == {e: k1_out[e]["total_out_of_frame_entities"] for e in ENGINE_ORDER}
    assert st["api_runs_per_question_wording"] == 21 and st["ui_runs_per_question"] == 3


@needs_data
def test_derived_flags_are_bools_and_the_headline_claims_hold(built):
    d = built["derived"]
    assert all(isinstance(v, bool) for v in d["flags"].values())
    for name in (
        "vendor_share_above_55_percent_in_every_engine", "review_sites_under_1_percent_in_every_engine",
        "youtube_present_in_both_google_engines", "youtube_absent_in_other_engines",
        "reddit_present_in_both_google_engines", "reddit_absent_in_other_engines",
        "google_ai_mode_stores_no_search_queries", "wording3_skips_search_in_most_runs",
        "top_url_is_aggregated_video_in_google_engines",
    ):
        assert d["flags"][name] is True, name
    assert d["vendor_share"]["min"]["engine"] == AI_MODE and d["vendor_share"]["max"]["engine"] == CLAUDE
    assert d["youtube_share_google_engines"]["min"]["engine"] == AI_MODE


@needs_data
def test_v1_sensitivity_reproduces_stored_v1_and_drops_no_search_runs(built, stored):
    v = built["v1_sensitivity"]
    notes = stored[0]["figures"]["V1"]["notes"]
    assert v["all_runs"]["engine_matters_more"] == notes["n_above_line_engine_matters_more"]
    assert v["all_runs"]["industry_matters_more"] == notes["n_below_line_industry_matters_more"]
    assert v["all_runs"]["median_ratio"] == pytest.approx(stored[0]["figures"]["V1"]["value"])
    assert v["no_search_runs_removed"] == stored[2]["derived"]["s4"]["counts"]["not_fired_total"]
    f = v["without_no_search_runs"]
    assert f["engine_over_industry_median"] == pytest.approx(1 / f["median_ratio"])
    assert built["derived"]["flags"]["v1_without_no_search_runs_engine_matters_more"] is v["same_conclusion"]


@needs_data
def test_study_window_labels_and_relabel(built):
    st = built["study"]
    assert st["window"]["month_label"] == "August 2026" and st["window"]["range_label"] == "20 to 26 August 2026"
    assert st["relabel"] == {"date": "2026-10-12", "calls": 8, "invalid_again_decided_by_hand": 6, "accepted_model_output": 2,
                              "set_by_hand_no_call": 1, "ledger_entry": "frame-label-fix-2026-10-12-outcome"}


@needs_data
def test_no_flag_stays_true_when_the_numbers_it_reads_move(built):
    import copy

    moved = copy.deepcopy(built)
    for e in ENGINE_ORDER:
        moved["source_classes"]["engines"][e]["classes"]["review_sites"]["share"] = 0.02
    assert pb._derive(moved)["flags"]["review_sites_under_1_percent_in_every_engine"] is False


@needs_data
def test_runner_writes_the_same_json_and_does_not_touch_the_inputs(tmp_path, built):
    runner = pytest.importorskip("scripts.results.run_business_figures_public")

    before = [p.read_bytes() for p in NEEDED]
    out = tmp_path / "public.json"
    assert runner.main(["--data-root", str(DATA), "--output", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads(json.dumps(built))
    assert [p.read_bytes() for p in NEEDED] == before


def test_runner_stops_without_the_other_business_figures(tmp_path, monkeypatch, capsys):
    runner = pytest.importorskip("scripts.results.run_business_figures_public")

    monkeypatch.setattr(paths, "BUSINESS_FIGURES_REST_JSON", tmp_path / "missing.json")
    assert runner.main(["--data-root", str(tmp_path), "--output", str(tmp_path / "x.json")]) == 1
    assert "missing" in capsys.readouterr().err


def test_relabel_record_matches_the_text_of_the_ledger_entry():
    """The counts in config/relabel_record_2026-10-12.json cannot drift from reports/ambiguity_ledger.md."""
    import re
    from analysis.phase_b.business import public_business as pb

    if not pb.RELABEL_RECORD_PATH.exists():
        pytest.skip("study relabel record not in this checkout")
    rec = json.loads(pb.RELABEL_RECORD_PATH.read_text(encoding="utf-8"))
    ledger = (paths.REPO_ROOT / "reports" / "ambiguity_ledger.md").read_text(encoding="utf-8")
    start = ledger.index(f"## {rec['date']} — {rec['ledger_entry']}")
    entry = ledger[start:ledger.index("\n## ", start + 5)]
    assert int(re.search(r"(\d+) frame-classification calls were made", entry).group(1)) == len(rec["calls_made"])
    malformed = entry[entry.index("Malformed again"):]
    for entity in rec["calls_made"]:
        assert entity in entry
    for entity in rec["invalid_again_decided_by_hand"]:
        assert entity in malformed.split("Decision.")[0]
    for entity in rec["accepted_model_output"]:
        assert entity not in malformed.split("Decision.")[0].split(":", 1)[1]
    for entity in rec["set_by_hand_no_call"]:
        assert f"{entity}: no call was made" in entry
    assert set(rec["invalid_again_decided_by_hand"]) | set(rec["accepted_model_output"]) == set(rec["calls_made"])
    assert not set(rec["set_by_hand_no_call"]) & set(rec["calls_made"])


@needs_data
def test_two_shade_view_counts_every_vendor_named_or_not(built):
    for b in built["vendor_domains"]["engines"].values():
        for d in b["top_domains"]:
            assert d["named_in_answer"] + d["not_named_in_answer"] == d["count"]
            assert d["named_in_answer"] == d["named_in_scope"] + d["out_of_scope_named_in_answer_count"]
