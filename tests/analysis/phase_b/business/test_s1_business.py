import json

import pytest

from analysis.phase_b.business import s1_business as sb
from analysis.phase_b.business import source_classes as sc
from analysis.phase_b.figures import s1
from analysis.phase_b.reading import FrozenInputs
from collector import paths


def _run(run_id, engine):
    return {"run_id": run_id, "engine": engine, "intent_id": 1, "paraphrase_id": 1, "error": None,
            "raw_response_text": "some text", "is_manual": None}


def _source(url):
    return {"position": 1, "url_raw": url, "domain": None, "title": None}


def _registry(url, url_class, poc):
    return {"url_normalized": url, "url_class": url_class, "publisher_is_in_category": poc,
            "settled_class": url_class, "settled_publisher_is_in_category": poc}


def _inputs():
    engines = ["openai_chatgpt", "google_ai_mode"]
    raw_runs = [_run("r1", engines[0]), _run("r2", engines[1])]
    sources = {
        "r1": [
            _source("https://acme.com/pricing"),          # own page, Acme named in r1
            _source("https://acme.com/best-tools"),       # own listicle
            _source("https://other-vendor.com/page"),     # vendor of a company not named
            _source("https://www.youtube.com/watch"),
            _source("https://reddit.com/r/x"),
            _source("https://nowhere.example/unregistered"),
        ],
        "r2": [_source("https://acme.com/pricing"), _source("https://www.g2.com/p")],
    }
    registry = [
        _registry("https://acme.com/pricing", "vendor page", True),
        _registry("https://acme.com/best-tools", "vendor listicle", True),
        _registry("https://other-vendor.com/page", "vendor page", True),
        _registry("https://youtube.com/watch", "other", False),
        _registry("https://reddit.com/r/x", "forum", False),
        _registry("https://g2.com/p", "review site", False),
    ]
    entities = [{"entity_id": "acme", "homepage_url": "https://acme.com", "surface_forms": "Acme", "meets_frame": "in"}]
    mentions = [{"run_id": "r1", "surface_form": "Acme", "evidence_span": "x", "is_recommended": False, "votes": []}]
    return FrozenInputs(entities=entities, alt_domains_rows=[], entity_verification_rows=[], mentions_raw=mentions,
                        raw_runs=raw_runs, sources_by_run=sources, url_registry_rows=registry)


@pytest.fixture(scope="module")
def built():
    inputs = _inputs()
    return sb.build_business_figures(inputs), s1.compute(inputs)


def test_source_mix_sums_to_one_and_vendor_share_equals_s1(built):
    figures, s1_result = built
    for engine, block in figures["engines"].items():
        assert sum(c["share"] for c in block["source_mix"].values()) == pytest.approx(1.0)
        assert sum(c["count"] for c in block["source_mix"].values()) == block["classified_citations"]
        assert block["source_mix"][sc.VENDOR]["share"] == s1_result.value[engine]
        assert block["vendor_published"]["share"] == s1_result.value[engine]


def test_self_reference_split_sums_to_vendor_count(built):
    figures, _ = built
    for block in figures["engines"].values():
        vp = block["vendor_published"]
        assert sum(p["count"] for p in vp["self_reference"].values()) == vp["count"]
        assert sum(d["count"] for d in vp["top_domains"]) <= vp["count"]


def test_self_reference_buckets_on_synthetic_runs(built):
    figures, _ = built
    chatgpt = figures["engines"]["openai_chatgpt"]["vendor_published"]
    assert {k: v["count"] for k, v in chatgpt["self_reference"].items()} == {
        sb.OWN_PAGE: 1, sb.OWN_LISTICLE: 1, sb.OTHER_VENDOR: 1}
    # same Acme page in r2 where Acme is not named: other vendor
    ai_mode = figures["engines"]["google_ai_mode"]["vendor_published"]
    assert ai_mode["self_reference"][sb.OTHER_VENDOR]["count"] == 1
    assert chatgpt["top_domains"][0] == {"domain": "acme.com", "count": 2, "self_reference_count": 2}


def test_unclassified_citation_is_left_out(built):
    figures, _ = built
    assert figures["engines"]["openai_chatgpt"]["classified_citations"] == 5


def test_main_result_flags(built):
    figures, _ = built
    assert figures["engines"]["openai_chatgpt"]["main_result"] is True


@pytest.mark.skipif(
    not (paths.BUSINESS_FIGURES_JSON.exists() and paths.FIGURES_JSON.exists()),
    reason="built business_figures.json / figures.json not present",
)
def test_built_file_matches_figures_json_s1_value():
    business = json.loads(paths.BUSINESS_FIGURES_JSON.read_text())
    figures = json.loads(paths.FIGURES_JSON.read_text())["figures"]
    s1_value = (figures["S1"] if isinstance(figures, dict) else next(f for f in figures if f["figure_id"] == "S1"))["value"]
    for engine, block in business["engines"].items():
        assert sum(c["share"] for c in block["source_mix"].values()) == pytest.approx(1.0)
        assert block["source_mix"][sc.VENDOR]["share"] == s1_value[engine]
        vp = block["vendor_published"]
        assert sum(p["count"] for p in vp["self_reference"].values()) == vp["count"]
    for engine, rep in business["leftover_other"].items():
        if engine in sc.MAIN_ENGINES:
            assert rep["other_share"] < sc.LEFTOVER_TARGET


# ---------------------------------------------------------------- derived (prose numbers and claim flags)

def _engine_block(vendor_count, mix_counts, top, self_ref):
    n = sum(mix_counts.values())
    return {
        "source_mix": {c: {"count": mix_counts.get(c, 0), "share": mix_counts.get(c, 0) / n} for c in sc.DISPLAY_CLASSES},
        "vendor_published": {
            "count": vendor_count, "share": vendor_count / n, "distinct_domains": len(top),
            "top_domains": [{"domain": d, "count": c, "self_reference_count": s} for d, c, s in top],
            "self_reference": {p: {"count": 0, "share": v} for p, v in self_ref.items()},
        },
    }


def _synthetic():
    google = _engine_block(10, {sc.VENDOR: 10, sc.VIDEO_SOCIAL: 4, sc.FORUMS: 2}, [("a.com", 6, 0), ("b.com", 2, 0)],
                           {"own_page": 0.1, "own_listicle": 0.0, "other_vendor": 0.9})
    chat = _engine_block(10, {sc.VENDOR: 10, sc.NEWS: 2}, [("c.com", 8, 7), ("d.com", 2, 0)],
                         {"own_page": 0.6, "own_listicle": 0.1, "other_vendor": 0.3})
    engines = {"google_ai_mode": google, "openai_chatgpt": chat}
    leftover = {e: {"other_share": 0.0} for e in engines}
    return engines, leftover


def test_derived_flags_and_numbers_on_synthetic_engines():
    engines, leftover = _synthetic()
    d = sb.derive(engines, leftover, {"runs_per_day": 3, "days": 7})
    assert d["counts"]["engines"] == 2 and d["study"] == {"runs_per_day": 3, "days": 7}
    assert d["metrics"]["video_social_share"]["max"] == {"engine": "google_ai_mode", "value": 0.25}
    assert d["metrics"]["video_social_share"]["min"]["engine"] == "openai_chatgpt"
    assert d["own_page_plus_listicle_share"]["openai_chatgpt"] == pytest.approx(0.7)
    assert d["own_page_plus_listicle_leader_engine"] == "openai_chatgpt"
    assert d["top10_vendor_domain_share"]["google_ai_mode"] == pytest.approx(0.8)
    assert d["leader_domain"]["openai_chatgpt"] == {"domain": "c.com", "count": 8, "self_reference_count": 7}
    assert d["flags"]["video_social_and_forums_only_in_google_engines"] is True
    assert d["flags"]["own_share_leader_is_openai_chatgpt"] is True
    assert d["flags"]["leader_domain_of_own_share_leader_mostly_self_reference"] is True
    assert d["flags"]["vendor_largest_class_in_every_engine"] is True


def test_derived_flags_turn_false_when_the_claim_stops_holding():
    engines, leftover = _synthetic()
    engines["openai_chatgpt"]["source_mix"][sc.FORUMS]["count"] = 1
    engines["openai_chatgpt"]["vendor_published"]["self_reference"]["own_page"]["share"] = 0.05
    engines["openai_chatgpt"]["vendor_published"]["self_reference"]["own_listicle"]["share"] = 0.0
    engines["openai_chatgpt"]["vendor_published"]["top_domains"][0]["self_reference_count"] = 1
    engines["google_ai_mode"]["source_mix"][sc.NEWS]["count"] = 99
    flags = sb.derive(engines, leftover)["flags"]
    assert flags["video_social_and_forums_only_in_google_engines"] is False
    assert flags["own_share_leader_is_openai_chatgpt"] is False
    assert flags["leader_domain_of_own_share_leader_mostly_self_reference"] is False
    assert flags["vendor_largest_class_in_every_engine"] is False
    assert "study" not in sb.derive(engines, leftover)


@pytest.mark.skipif(
    not (paths.BUSINESS_FIGURES_JSON.exists() and paths.FIGURES_JSON.exists()),
    reason="built business_figures.json / figures.json not present",
)
def test_built_derived_equals_figures_json_and_claims_hold():
    business = json.loads(paths.BUSINESS_FIGURES_JSON.read_text())
    figures = json.loads(paths.FIGURES_JSON.read_text())["figures"]
    d = business["derived"]
    s1_value = figures["S1"]["value"]
    vendor = d["metrics"]["vendor_share"]
    assert vendor["min"]["value"] == min(s1_value.values()) and s1_value[vendor["min"]["engine"]] == vendor["min"]["value"]
    assert vendor["max"]["value"] == max(s1_value.values()) and s1_value[vendor["max"]["engine"]] == vendor["max"]["value"]
    assert d["counts"]["engines"] == len(s1_value) == 4
    assert all(d["flags"].values())
    for engine, block in business["engines"].items():
        vp = block["vendor_published"]
        assert d["distinct_vendor_domains"][engine] == vp["distinct_domains"]
        assert d["leader_domain"][engine]["domain"] == vp["top_domains"][0]["domain"]
        assert d["top10_vendor_domain_share"][engine] == pytest.approx(sum(t["count"] for t in vp["top_domains"]) / vp["count"])
    # the study constants come from the experiment config
    from collector.runner.experiment_config import load_experiment_config
    config = load_experiment_config(paths.EXPERIMENT_YAML)
    assert d["study"] == {"runs_per_day": config.reps_per_day, "days": config.days}


def _flags(engines, leftover=None):
    return sb.derive(engines, leftover or {e: {"other_share": 0.0} for e in engines})["flags"]


def test_flag_forums_below_video_social_true_and_false():
    engines, _ = _synthetic()
    assert _flags(engines)["forums_below_video_social_in_google_engines"] is True  # 0.2 < 0.4
    engines["google_ai_mode"]["source_mix"][sc.FORUMS]["share"] = 0.4  # equal is not below
    assert _flags(engines)["forums_below_video_social_in_google_engines"] is False


def test_flag_own_share_leader_ahead_of_every_other_engine_true_and_false():
    engines, _ = _synthetic()
    assert _flags(engines)["own_share_leader_ahead_of_every_other_engine"] is True  # 0.7 against 0.1
    engines["google_ai_mode"]["vendor_published"]["self_reference"]["own_page"]["share"] = 0.7
    engines["google_ai_mode"]["vendor_published"]["self_reference"]["own_listicle"]["share"] = 0.0
    engines["openai_chatgpt"]["vendor_published"]["self_reference"]["own_page"]["share"] = 0.6
    engines["openai_chatgpt"]["vendor_published"]["self_reference"]["own_listicle"]["share"] = 0.1
    d = sb.derive(engines, {e: {"other_share": 0.0} for e in engines})
    assert d["own_page_plus_listicle_share"]["google_ai_mode"] == d["own_page_plus_listicle_share"]["openai_chatgpt"]
    assert d["flags"]["own_share_leader_ahead_of_every_other_engine"] is False  # a tie is not ahead


def test_flag_top10_share_below_half_in_google_engines_true_and_false():
    engines, _ = _synthetic()
    engines["google_ai_mode"]["vendor_published"]["top_domains"] = [{"domain": "a.com", "count": 4, "self_reference_count": 0}]
    assert _flags(engines)["top10_domains_share_below_half_of_vendor_citations_in_google_engines"] is True  # 0.4
    engines["google_ai_mode"]["vendor_published"]["top_domains"][0]["count"] = 5  # exactly half is not below
    assert _flags(engines)["top10_domains_share_below_half_of_vendor_citations_in_google_engines"] is False


def test_extremes_tie_by_study_engine_order_and_skip_none():
    from analysis.phase_b.business.extremes import ENGINE_ORDER, extreme
    style = pytest.importorskip("scripts.results.user_charts.style")

    assert ENGINE_ORDER == style.ENGINE_ORDER
    tied = extreme({"anthropic_claude": 1.0, "openai_chatgpt": 1.0, "google_ai_mode": 0.5})
    assert tied["max"]["engine"] == "openai_chatgpt" and tied["min"]["engine"] == "google_ai_mode"
    assert extreme({"a": None, "b": 2.0, "c": 1.0}) == {"min": {"engine": "c", "value": 1.0}, "max": {"engine": "b", "value": 2.0}}
    assert extreme({"a": None}) == {"min": None, "max": None}
