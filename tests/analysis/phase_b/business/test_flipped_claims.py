"""The four claims that flipped after the 2026-10-12 relabel: the computed flags stay false, and new true
flags and values carry what the data now says. Values are read from the stored figures, never retyped."""
import copy
import datetime
import json

import pytest

from analysis.phase_b.business import rest_business as rb
from analysis.phase_b.business.extremes import ENGINE_ORDER
from collector import paths

pytestmark = pytest.mark.skipif(not paths.FIGURES_JSON.exists(), reason="frozen figures.json not present")

OPENAI, AI_MODE, GEMINI, CLAUDE = ENGINE_ORDER
STUDY = rb.study_constants(
    {"runs_per_day": 3, "days": 7}, {"start": "2026-08-20", "end": "2026-08-26", "days": 7}, datetime.date(2026, 9, 5)
)


@pytest.fixture(scope="module")
def frozen():
    return json.loads(paths.FIGURES_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def built(frozen):
    return rb.build_rest_figures(frozen, STUDY)


def test_the_four_old_flags_evaluate_false_on_the_relabelled_data(built):
    d = built["derived"]
    assert d["s2"]["flags"]["every_other_engine_restricted_above_half"] is False
    assert d["s2"]["flags"]["highest_restricted_engine_is_anthropic_claude"] is False
    assert d["o1"]["flags"]["no_head_overlap_at_or_above_stable_line"] is False
    assert d["k2"]["flags"]["dropping_population_domains_raises_share_in_every_engine"] is False


def test_s2_per_engine_values_come_from_figures_json_and_split_by_half(built, frozen):
    d = built["derived"]["s2"]
    stored = frozen["figures"]["S2"]["value"]
    for e in ENGINE_ORDER:
        assert d["restricted_by_engine"][e] == stored[e]["restricted"]
    assert d["engines_above_half"] == [GEMINI, CLAUDE]
    assert d["engines_about_half"] == [AI_MODE]
    assert d["engines_below_half"] == [OPENAI]
    assert sorted(d["engines_above_half"] + d["engines_about_half"] + d["engines_below_half"]) == sorted(ENGINE_ORDER)
    assert d["restricted_ranked_engines"] == [GEMINI, CLAUDE, AI_MODE, OPENAI]
    labels = {e: p["label"] for e, p in d["restricted_plain_fraction"].items()}
    assert labels == {OPENAI: "1 in 6", AI_MODE: "1 in 2", GEMINI: "2 in 3", CLAUDE: "2 in 3"}
    for p in d["restricted_plain_fraction"].values():
        assert p["distance"] <= d["restricted_fraction_tolerance"]
    assert d["restricted"]["max"]["engine"] == GEMINI and d["restricted"]["min"]["engine"] == OPENAI


def test_s2_new_flags_are_true(built):
    flags = built["derived"]["s2"]["flags"]
    for name in (
        "highest_restricted_engine_is_google_gemini",
        "highest_two_restricted_engines_are_google_gemini_and_anthropic_claude",
        "google_gemini_and_anthropic_claude_clearly_above_half",
        "google_gemini_and_anthropic_claude_about_two_thirds",
        "google_ai_mode_about_half",
        "google_ai_mode_below_half",
        "openai_chatgpt_about_one_in_six",
        "exactly_two_engines_clearly_above_half",
    ):
        assert flags[name] is True, name


def test_s2_plain_fraction_says_none_when_nothing_fits():
    assert rb._plain_fraction(0.40)["label"] is None  # between 1/3 and 1/2, both farther than the tolerance
    assert rb._plain_fraction(0.50)["label"] == "1 in 2"
    assert rb._vs_half(0.5499) == "about" and rb._vs_half(0.551) == "above" and rb._vs_half(0.449) == "below"


def test_s2_new_flags_turn_false_when_the_data_moves(frozen):
    bad = copy.deepcopy(frozen)
    s2 = bad["figures"]["S2"]["value"]
    s2[CLAUDE]["restricted"], s2[GEMINI]["restricted"] = 0.80, 0.30
    flags = rb.build_rest_figures(bad, STUDY)["derived"]["s2"]["flags"]
    assert flags["highest_restricted_engine_is_anthropic_claude"] is True
    assert flags["highest_restricted_engine_is_google_gemini"] is False
    assert flags["google_gemini_and_anthropic_claude_about_two_thirds"] is False


def test_o1_gemini_is_the_only_engine_at_or_above_the_stable_line(built, frozen):
    d = built["derived"]["o1"]
    stored = frozen["figures"]["O1"]["value"]
    line = built["o1"]["head_stable_at_or_above"]
    for e in ENGINE_ORDER:
        assert d["head_overlap_by_engine"][e] == stored[e]["head_mean_jaccard"]
        assert d["head_gap_to_stable_line"][e] == pytest.approx(stored[e]["head_mean_jaccard"] - line)
    assert d["engines_head_at_or_above_stable_line"] == [GEMINI]
    assert d["head_overlap"]["max"]["engine"] == GEMINI and d["head_overlap"]["max"]["value"] >= line
    flags = d["flags"]
    for name in (
        "head_overlap_at_or_above_stable_line_in_some_engine",
        "head_overlap_at_or_above_stable_line_in_exactly_one_engine",
        "only_google_gemini_head_overlap_at_or_above_stable_line",
        "highest_head_overlap_engine_is_at_or_above_stable_line",
        "stable_head_without_churning_tail_in_some_engine",
        "head_overlap_at_or_above_stable_line_in_google_gemini",
    ):
        assert flags[name] is True, name
    for e in (OPENAI, AI_MODE, CLAUDE):
        assert flags[f"head_overlap_at_or_above_stable_line_in_{e}"] is False
    # The full picture (stable head AND churning tail) is still met in no engine.
    assert flags["stable_head_churning_tail_met_in_no_engine"] is True
    assert flags["stable_head_churning_tail_met_in_some_engine"] is False


def test_o1_new_flags_follow_the_data(frozen):
    bad = copy.deepcopy(frozen)
    for e in ENGINE_ORDER:
        bad["figures"]["O1"]["value"][e]["head_mean_jaccard"] = 0.5
    flags = rb.build_rest_figures(bad, STUDY)["derived"]["o1"]["flags"]
    assert flags["no_head_overlap_at_or_above_stable_line"] is True
    assert flags["head_overlap_at_or_above_stable_line_in_some_engine"] is False
    assert flags["head_overlap_at_or_above_stable_line_in_google_gemini"] is False


def test_k2_exclusion_raises_three_engines_and_lowers_claude(built, frozen):
    d = built["derived"]["k2"]
    stored = frozen["figures"]["K2"]["value"]
    for e in ENGINE_ORDER:
        assert d["share_by_engine"]["included"][e] == stored[e]["included"]
        assert d["share_by_engine"]["excluded_population"][e] == stored[e]["excluded_population"]
    assert d["engines_exclusion_raises_share"] == [OPENAI, AI_MODE, GEMINI]
    assert d["engines_exclusion_lowers_share"] == [CLAUDE]
    assert d["population_minus_included"][CLAUDE] < 0
    flags = d["flags"]
    for name in (
        "dropping_population_domains_raises_share_in_some_engine",
        "dropping_population_domains_raises_share_in_every_engine_but_one",
        "dropping_population_domains_lowers_share_only_in_anthropic_claude",
        "dropping_population_domains_raises_share_in_every_engine_except_anthropic_claude",
    ):
        assert flags[name] is True, name


def test_k2_new_flags_turn_false_when_claude_rises_too(frozen):
    bad = copy.deepcopy(frozen)
    k2 = bad["figures"]["K2"]["value"][CLAUDE]
    k2["excluded_population"] = k2["included"] + 0.05
    flags = rb.build_rest_figures(bad, STUDY)["derived"]["k2"]["flags"]
    assert flags["dropping_population_domains_raises_share_in_every_engine"] is True
    assert flags["dropping_population_domains_lowers_share_only_in_anthropic_claude"] is False
