"""Flags added in the 2026-10-12 prose sweep of the appendix: each backs one qualitative sentence."""
import copy
import datetime
import json

import pytest

from analysis.phase_b.business import rest_business as rb
from analysis.phase_b.business.extremes import ENGINE_ORDER
from collector import paths

pytestmark = pytest.mark.skipif(not paths.FIGURES_JSON.exists(), reason="frozen figures.json not present")

STUDY = rb.study_constants(
    {"runs_per_day": 3, "days": 7}, {"start": "2026-08-20", "end": "2026-08-26", "days": 7}, datetime.date(2026, 9, 5)
)


@pytest.fixture(scope="module")
def frozen():
    return json.loads(paths.FIGURES_JSON.read_text(encoding="utf-8"))


def test_o1_no_tail_overlap_at_or_below_the_churning_line(frozen):
    built = rb.build_rest_figures(frozen, STUDY)
    d = built["derived"]["o1"]
    line = built["o1"]["tail_churning_at_or_below"]
    assert d["flags"]["no_tail_overlap_at_or_below_churning_line"] is True
    assert all(v > line for v in d["tail_overlap_by_engine"].values())
    bad = copy.deepcopy(frozen)
    bad["figures"]["O1"]["value"][ENGINE_ORDER[0]]["tail_mean_jaccard"] = 0.3
    flags = rb.build_rest_figures(bad, STUDY)["derived"]["o1"]["flags"]
    assert flags["no_tail_overlap_at_or_below_churning_line"] is False


def test_k2_top_sites_hold_under_half_with_every_citation_counted(frozen):
    d = rb.build_rest_figures(frozen, STUDY)["derived"]["k2"]
    assert d["flags"]["top_sites_hold_under_half_of_citations_with_every_citation_counted"] is True
    assert all(v < 0.5 for v in d["share_by_engine"]["included"].values())
    bad = copy.deepcopy(frozen)
    bad["figures"]["K2"]["value"][ENGINE_ORDER[0]]["included"] = 0.55
    flags = rb.build_rest_figures(bad, STUDY)["derived"]["k2"]["flags"]
    assert flags["top_sites_hold_under_half_of_citations_with_every_citation_counted"] is False


@pytest.mark.skipif(not paths.BUSINESS_FIGURES_JSON.exists(), reason="business figures not built")
def test_s1_own_share_below_half_outside_chatgpt_matches_the_shares():
    s1 = json.loads(paths.BUSINESS_FIGURES_JSON.read_text(encoding="utf-8"))["derived"]
    shares = s1["own_page_plus_listicle_share"]
    assert s1["flags"]["own_share_below_half_outside_openai_chatgpt"] is True
    assert all(v < 0.5 for e, v in shares.items() if e != "openai_chatgpt") and shares["openai_chatgpt"] > 0.5


@pytest.mark.skipif(not paths.BUSINESS_FIGURES_PUBLIC_JSON.exists(), reason="public figures not built")
def test_public_flags_for_a1_v1_and_unique_url_sentences_match_the_stored_numbers():
    pub = json.loads(paths.BUSINESS_FIGURES_PUBLIC_JSON.read_text(encoding="utf-8"))
    flags = pub["derived"]["flags"]
    assert flags["a1_stored_values_null_in_every_engine"] is True
    assert pub["study"]["a1_rule"]["stored_value_null_in_every_engine"] is True
    assert flags["v1_median_ratio_at_or_below_planned_line"] is True
    assert pub["v1_sensitivity"]["all_runs"]["median_ratio"] <= pub["v1_sensitivity"]["planned_falsifier_at_or_below"]
    gap = pub["derived"]["unique_vendor_minus_citation_vendor_share"]
    assert flags["unique_vendor_share_above_citation_share_in_both_google_engines"] is True
    assert gap["google_ai_mode"] > 0 and gap["google_gemini"] > 0
