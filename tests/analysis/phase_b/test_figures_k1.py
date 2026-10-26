from analysis.phase_b.figures import k1
from analysis.phase_b.reading import FrozenInputs


def _mention(run_id, surface_form):
    return {"run_id": run_id, "surface_form": surface_form, "evidence_span": "x", "is_recommended": False, "votes": []}


def _entity(entity_id, homepage_url, surface_forms, meets_frame="in"):
    return {"entity_id": entity_id, "homepage_url": homepage_url, "surface_forms": surface_forms, "meets_frame": meets_frame}


def _raw_run(run_id, *, engine="openai_chatgpt", intent_id=1, error=None):
    return {
        "run_id": run_id, "engine": engine, "intent_id": intent_id, "paraphrase_id": 1,
        "day_index": 1, "error": error, "raw_response_text": "x", "is_manual": None,
    }


def _inputs(raw_runs, mentions_raw, entities):
    return FrozenInputs(
        entities=entities,
        alt_domains_rows=[],
        entity_verification_rows=[],
        mentions_raw=mentions_raw,
        raw_runs=raw_runs,
        sources_by_run={},
    )


def test_zero_cells_is_untestable():
    result = k1.compute(_inputs([], [], []))
    assert result.status == "untestable"


def test_top5_share_and_count_to_50pct():
    # 6 companies, one dominant with 10 mentions, five with 1 each -> total 15
    raw_runs, mentions, entities = [], [], []
    for i in range(10):
        raw_runs.append(_raw_run(f"big-{i}", intent_id=1))
        mentions.append(_mention(f"big-{i}", "big"))
    entities.append(_entity("big", "https://big.com", "big"))
    for name in ("s1", "s2", "s3", "s4", "s5"):
        raw_runs.append(_raw_run(f"{name}-run", intent_id=1))
        mentions.append(_mention(f"{name}-run", name))
        entities.append(_entity(name, f"https://{name}.com", name))

    result = k1.compute(_inputs(raw_runs, mentions, entities))
    # top 5 = big(10) + 4 of the small ones (1 each) = 14/15
    assert result.value["openai_chatgpt"] == 14 / 15
    assert result.notes["count_to_50pct"]["openai_chatgpt"] == 1  # "big" alone already >= 50%
    assert result.denominator_after_exclusion["openai_chatgpt"] == 15


def test_both_flat_baselines_present_and_distinct():
    raw_runs = [_raw_run("run1", intent_id=1)]
    mentions = [_mention("run1", "acme")]
    entities = [_entity("acme", "https://acme.com", "acme")]
    result = k1.compute(_inputs(raw_runs, mentions, entities))

    fixed = result.notes["flat_baseline_fixed"]
    recomputed = result.notes["flat_baseline_recomputed_by_engine"]["openai_chatgpt"]
    assert fixed["n"] == 415
    assert fixed["share"] == 5 / 415
    assert recomputed == 5 / 1  # only 1 distinct company mentioned in this fixture
    assert fixed["share"] != recomputed


def test_out_of_frame_brand_counted_by_bucket():
    raw_runs = [_raw_run("run1", intent_id=1)]
    mentions = [
        _mention("run1", "InFrameCo"),
        _mention("run1", "OutCo"),
        _mention("run1", "BorderlineCo"),
        _mention("run1", "UnresolvedCo"),
        _mention("run1", "NeverRegistered"),
    ]
    entities = [
        _entity("in-co", "https://in.com", "InFrameCo", meets_frame="in"),
        _entity("out-co", "https://out.com", "OutCo", meets_frame="out"),
        _entity("border-co", "https://border.com", "BorderlineCo", meets_frame="borderline"),
        _entity("unres-co", "https://unres.com", "UnresolvedCo", meets_frame="unresolved"),
        # "NeverRegistered" has no entity row at all -- unmatched surface form
    ]
    result = k1.compute(_inputs(raw_runs, mentions, entities))
    oof = result.notes["out_of_frame_named_by_engine"]["openai_chatgpt"]
    assert oof["out"] == 1
    assert oof["borderline"] == 1
    assert oof["unresolved"] == 1
    assert oof["unmatched_surface_forms"] == 1
    assert oof["total_out_of_frame_entities"] == 3  # out + borderline + unresolved, never in-frame


def test_branded_intent_7_mentions_excluded():
    raw_runs = [_raw_run("run1", intent_id=7)]
    mentions = [_mention("run1", "acme")]
    entities = [_entity("acme", "https://acme.com", "acme")]
    result = k1.compute(_inputs(raw_runs, mentions, entities))
    assert result.status == "untestable"


def test_errored_run_excluded_from_denominator():
    raw_runs = [_raw_run("run1", intent_id=1), _raw_run("run2", intent_id=1, error="collection_error")]
    mentions = [_mention("run1", "acme"), _mention("run2", "acme")]
    entities = [_entity("acme", "https://acme.com", "acme")]
    result = k1.compute(_inputs(raw_runs, mentions, entities))
    assert result.denominator_after_exclusion["openai_chatgpt"] == 1  # run2 excluded
