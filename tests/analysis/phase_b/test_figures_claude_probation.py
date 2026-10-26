from analysis.phase_b.figures import FIGURES, claude_probation as cp
from analysis.phase_b.reading import FrozenInputs


def _entity(entity_id):
    return {"entity_id": entity_id, "homepage_url": f"https://{entity_id}.com", "surface_forms": entity_id, "meets_frame": "in"}


def _build(*, cited: bool, search_fired: bool, n_c=4, n_d=4, n_runs=10):
    raw_runs, mentions, sources = [], [], {}
    c_ids = [f"c{i}" for i in range(n_c)]
    d_ids = [f"d{i}" for i in range(n_d)]
    for engine, ids in (("anthropic_claude", c_ids), ("google_ai_mode", d_ids)):
        for k in range(n_runs):
            run_id = f"{engine}-{k}"
            raw_runs.append({
                "run_id": run_id, "engine": engine, "intent_id": 1, "paraphrase_id": 1, "day_index": 1 + k % 7,
                "rep_index": 1, "error": None, "raw_response_text": "text", "is_manual": None,
                "search_fired": search_fired,
            })
            sources[run_id] = [{"url_raw": "https://x.com/a", "domain": "x.com"}] if cited else []
            for e in ids:
                mentions.append({"run_id": run_id, "surface_form": e, "evidence_span": "x", "is_recommended": False, "votes": []})
    entities = [_entity(e) for e in c_ids + d_ids]
    return FrozenInputs(
        entities=entities, alt_domains_rows=[], entity_verification_rows=[], mentions_raw=mentions,
        raw_runs=raw_runs, sources_by_run=sources,
    )


def test_registered():
    assert cp.FIGURE_ID in FIGURES


def test_keep_when_sets_big_and_set_a_mentions_without_source():
    r = cp.compute(_build(cited=False, search_fired=True))
    assert r.value["set_a_size"] == 4 and r.value["set_b_size"] == 4
    assert r.value["share_set_a_mentions_without_search"] == 1.0
    assert r.value["verdict"] == "keep" and r.value["failed_conditions"] == []


def test_cut_names_failed_condition_when_search_and_sources_present():
    r = cp.compute(_build(cited=True, search_fired=True))
    assert r.value["verdict"] == "cut"
    assert r.value["failed_conditions"] == [cp.COND_NO_SEARCH]
    assert r.value["share_set_a_mentions_without_search"] == 0.0


def test_cut_on_set_size_below_three():
    r = cp.compute(_build(cited=False, search_fired=True, n_c=2, n_d=2))
    assert cp.COND_SET_SIZE in r.value["failed_conditions"]
    assert r.value["verdict"] == "cut"


def test_figure_shape_fields_present():
    r = cp.compute(_build(cited=False, search_fired=True))
    d = r.to_json()
    for k in ("denominator_before_exclusion", "denominator_after_exclusion", "n_included", "n_excluded", "threshold"):
        assert k in d
    assert d["n_excluded"] == {"zero_in_both_engines": 0}
