import pytest

from analysis.phase_b.figures import o2
from analysis.phase_b.reading import FrozenInputs


def _mention(run_id, surface_form):
    return {"run_id": run_id, "surface_form": surface_form, "evidence_span": "x", "is_recommended": False, "votes": []}


def _entity(entity_id, homepage_url, surface_forms, meets_frame="in"):
    return {"entity_id": entity_id, "homepage_url": homepage_url, "surface_forms": surface_forms, "meets_frame": meets_frame}


def _run(run_id, *, engine, intent_id, paraphrase_id, day_index, rep_index, error=None):
    return {
        "run_id": run_id, "engine": engine, "intent_id": intent_id, "paraphrase_id": paraphrase_id,
        "day_index": day_index, "rep_index": rep_index, "error": error,
        "raw_response_text": "some text", "is_manual": None,
    }


def _inputs(raw_runs, mentions_raw, entities):
    return FrozenInputs(
        entities=entities, alt_domains_rows=[], entity_verification_rows=[],
        mentions_raw=mentions_raw, raw_runs=raw_runs, sources_by_run={},
    )


def _full_week(engine="openai_chatgpt", intent_id=1, paraphrases=(1, 2, 3, 4, 5), days=range(1, 8), reps=(1, 2, 3)):
    for paraphrase_id in paraphrases:
        for day_index in days:
            for rep_index in reps:
                yield paraphrase_id, day_index, rep_index


def test_zero_cells_is_untestable():
    result = o2.compute(_inputs([], [], []))
    assert result.status == "untestable"


def test_identical_sets_everywhere_gives_tied_direction():
    engine, intent_id = "openai_chatgpt", 1
    raw_runs, mentions = [], []
    for paraphrase_id, day_index, rep_index in _full_week(engine, intent_id):
        run_id = f"run-p{paraphrase_id}-d{day_index}-r{rep_index}"
        raw_runs.append(_run(run_id, engine=engine, intent_id=intent_id, paraphrase_id=paraphrase_id, day_index=day_index, rep_index=rep_index))
        mentions.append(_mention(run_id, "e1"))
        mentions.append(_mention(run_id, "e2"))
    entities = [_entity("e1", "https://e1.com", "e1"), _entity("e2", "https://e2.com", "e2")]

    result = o2.compute(_inputs(raw_runs, mentions, entities))
    cell = result.value[engine][intent_id]
    assert cell["paraphrase_jaccard"] == pytest.approx(1.0)
    assert cell["rep_jaccard"] == pytest.approx(1.0)
    assert cell["direction"] == "tied"
    # Figure-level: a zero difference gives a degenerate (single-pair) bootstrap
    # CI of [0, 0], which crosses zero -> the new no_meaningful_difference label.
    assert result.notes["direction"] == "no_meaningful_difference"
    assert result.notes["bootstrap"]["crosses_zero"] is True


def test_disjoint_paraphrases_identical_reps_gives_paraphrase_lower():
    engine, intent_id = "openai_chatgpt", 1
    raw_runs, mentions = [], []
    for paraphrase_id, day_index, rep_index in _full_week(engine, intent_id):
        run_id = f"run-p{paraphrase_id}-d{day_index}-r{rep_index}"
        raw_runs.append(_run(run_id, engine=engine, intent_id=intent_id, paraphrase_id=paraphrase_id, day_index=day_index, rep_index=rep_index))
        # each paraphrase always mentions its own unique entity, every day, every rep
        mentions.append(_mention(run_id, f"e_p{paraphrase_id}"))
    entities = [_entity(f"e_p{p}", f"https://e{p}.com", f"e_p{p}") for p in (1, 2, 3, 4, 5)]

    result = o2.compute(_inputs(raw_runs, mentions, entities))
    cell = result.value[engine][intent_id]
    assert cell["paraphrase_jaccard"] == pytest.approx(0.0)
    assert cell["rep_jaccard"] == pytest.approx(1.0)
    assert cell["direction"] == "paraphrase_lower"
    assert result.notes["direction"] == "paraphrase_lower"


def test_identical_paraphrases_disjoint_reps_gives_rep_lower():
    engine, intent_id = "openai_chatgpt", 1
    raw_runs, mentions = [], []
    for paraphrase_id, day_index, rep_index in _full_week(engine, intent_id):
        run_id = f"run-p{paraphrase_id}-d{day_index}-r{rep_index}"
        raw_runs.append(_run(run_id, engine=engine, intent_id=intent_id, paraphrase_id=paraphrase_id, day_index=day_index, rep_index=rep_index))
        # rep position alone determines which entity is mentioned, identically
        # across every day and every paraphrase -- pooled per paraphrase this
        # still yields the SAME 3-entity set for every paraphrase.
        mentions.append(_mention(run_id, f"r{rep_index}"))
    entities = [_entity(f"r{r}", f"https://r{r}.com", f"r{r}") for r in (1, 2, 3)]

    result = o2.compute(_inputs(raw_runs, mentions, entities))
    cell = result.value[engine][intent_id]
    assert cell["paraphrase_jaccard"] == pytest.approx(1.0)
    assert cell["rep_jaccard"] == pytest.approx(0.0)
    assert cell["direction"] == "rep_lower"
    assert result.notes["direction"] == "rep_lower"
    assert result.notes["bootstrap"]["crosses_zero"] is False


def test_paraphrase_overlap_is_same_day_not_pooled_across_days():
    engine, intent_id = "openai_chatgpt", 1
    raw_runs, mentions = [], []
    # Day 1: paraphrase 1 mentions "a", paraphrase 2 mentions "b". Day 2: swapped.
    # Pooled-across-days (the old, fixed bug) would give both paraphrases the
    # SAME set {a,b} -> Jaccard 1.0. Same-day (the fix) gives each day a
    # disjoint pair -> Jaccard 0.0 every day -> mean 0.0.
    pattern = {(1, 1): "a", (1, 2): "b", (2, 1): "b", (2, 2): "a"}
    for (day_index, paraphrase_id), entity_id in pattern.items():
        for rep_index in (1, 2, 3):
            run_id = f"run-p{paraphrase_id}-d{day_index}-r{rep_index}"
            raw_runs.append(
                _run(run_id, engine=engine, intent_id=intent_id, paraphrase_id=paraphrase_id, day_index=day_index, rep_index=rep_index)
            )
            mentions.append(_mention(run_id, entity_id))
    entities = [_entity("a", "https://a.com", "a"), _entity("b", "https://b.com", "b")]

    result = o2.compute(_inputs(raw_runs, mentions, entities))
    cell = result.value[engine][intent_id]
    assert cell["paraphrase_jaccard"] == pytest.approx(0.0)
    assert cell["n_paraphrase_overlap_components"] == 2  # one component per day


def test_bootstrap_ci_is_deterministic_given_fixed_seed():
    engine, intent_id = "openai_chatgpt", 1
    raw_runs, mentions = [], []
    for paraphrase_id, day_index, rep_index in _full_week(engine, intent_id):
        run_id = f"run-p{paraphrase_id}-d{day_index}-r{rep_index}"
        raw_runs.append(_run(run_id, engine=engine, intent_id=intent_id, paraphrase_id=paraphrase_id, day_index=day_index, rep_index=rep_index))
        mentions.append(_mention(run_id, f"e_p{paraphrase_id}"))
    entities = [_entity(f"e_p{p}", f"https://e{p}.com", f"e_p{p}") for p in (1, 2, 3, 4, 5)]
    inputs = _inputs(raw_runs, mentions, entities)

    result1 = o2.compute(inputs)
    result2 = o2.compute(inputs)
    assert result1.notes["bootstrap"]["ci_low"] == result2.notes["bootstrap"]["ci_low"]
    assert result1.notes["bootstrap"]["ci_high"] == result2.notes["bootstrap"]["ci_high"]


def test_intent_name_present_in_value():
    engine, intent_id = "openai_chatgpt", 2  # dental, per a1.INTENT_THEME_NAMES
    raw_runs, mentions = [], []
    for paraphrase_id, day_index, rep_index in _full_week(engine, intent_id):
        run_id = f"run-p{paraphrase_id}-d{day_index}-r{rep_index}"
        raw_runs.append(_run(run_id, engine=engine, intent_id=intent_id, paraphrase_id=paraphrase_id, day_index=day_index, rep_index=rep_index))
        mentions.append(_mention(run_id, "e1"))
    entities = [_entity("e1", "https://e1.com", "e1")]

    result = o2.compute(_inputs(raw_runs, mentions, entities))
    assert result.value[engine][intent_id]["intent_name"] == "dental"
