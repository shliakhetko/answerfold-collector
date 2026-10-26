from analysis.phase_b.figures import x1
from analysis.phase_b.reading import FrozenInputs


def _run(run_id, engine, *, intent_id=1, error=None, raw_response_text="some text"):
    return {
        "run_id": run_id,
        "engine": engine,
        "intent_id": intent_id,
        "paraphrase_id": 1,
        "error": error,
        "raw_response_text": raw_response_text,
        "is_manual": None,
    }


def _source(url):
    return {"position": 1, "url_raw": url, "domain": None}


def _registry(url, url_class, publisher_is_in_category=False):
    return {
        "url_normalized": url,
        "url_class": url_class,
        "publisher_is_in_category": publisher_is_in_category,
        "settled_class": url_class,
        "settled_publisher_is_in_category": publisher_is_in_category,
    }


def _inputs(raw_runs, sources_by_run, url_registry_rows=()):
    return FrozenInputs(
        entities=[],
        alt_domains_rows=[],
        entity_verification_rows=[],
        mentions_raw=[],
        raw_runs=raw_runs,
        sources_by_run=sources_by_run,
        url_registry_rows=list(url_registry_rows),
    )


def test_crosstab_shares_sum_to_one_per_engine():
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {"r1": [_source("https://a.example"), _source("https://b.example"), _source("https://c.example")]}
    registry = [
        _registry("https://a.example", "vendor page"),
        _registry("https://b.example", "vendor page"),
        _registry("https://c.example", "forum"),
    ]
    result = x1.compute(_inputs(raw_runs, sources_by_run, registry))
    assert result.value["openai_chatgpt"] == {"vendor page": 2 / 3, "forum": 1 / 3}
    assert abs(sum(result.value["openai_chatgpt"].values()) - 1.0) < 1e-9


def test_modal_class_per_engine():
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {"r1": [_source("https://a.example"), _source("https://b.example"), _source("https://c.example")]}
    registry = [
        _registry("https://a.example", "vendor page"),
        _registry("https://b.example", "vendor page"),
        _registry("https://c.example", "forum"),
    ]
    result = x1.compute(_inputs(raw_runs, sources_by_run, registry))
    assert result.notes["modal_class_by_engine"]["openai_chatgpt"] == "vendor page"


def test_modal_class_identical_true_when_all_engines_agree():
    engines = ["openai_chatgpt", "anthropic_claude"]
    raw_runs = [_run(f"r{i}", e) for i, e in enumerate(engines)]
    sources_by_run = {}
    registry = []
    for i, e in enumerate(engines):
        url = f"https://{e}.example"
        sources_by_run[f"r{i}"] = [_source(url)]
        registry.append(_registry(url, "vendor page"))
    result = x1.compute(_inputs(raw_runs, sources_by_run, registry))
    assert result.notes["modal_class_identical_across_engines"] is True


def test_modal_class_identical_false_when_engines_disagree():
    raw_runs = [_run("r0", "openai_chatgpt"), _run("r1", "anthropic_claude")]
    sources_by_run = {
        "r0": [_source("https://a.example")],
        "r1": [_source("https://b.example")],
    }
    registry = [
        _registry("https://a.example", "vendor page"),
        _registry("https://b.example", "forum"),
    ]
    result = x1.compute(_inputs(raw_runs, sources_by_run, registry))
    assert result.notes["modal_class_identical_across_engines"] is False


def test_unclassified_excluded_from_denominator_and_counted():
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {"r1": [_source("https://known.example"), _source("https://never-classified.example")]}
    registry = [_registry("https://known.example", "vendor page")]
    result = x1.compute(_inputs(raw_runs, sources_by_run, registry))
    assert result.denominator_before_exclusion["openai_chatgpt"] == 2
    assert result.denominator_after_exclusion["openai_chatgpt"] == 1
    assert result.n_excluded["openai_chatgpt"] == {"unclassified": 1}


def test_branded_intent_7_excluded_via_citation_panel():
    raw_runs = [_run("r1", "openai_chatgpt", intent_id=7)]
    sources_by_run = {"r1": [_source("https://a.example")]}
    registry = [_registry("https://a.example", "vendor page")]
    result = x1.compute(_inputs(raw_runs, sources_by_run, registry))
    assert result.status == "untestable"


def test_no_pre_registered_threshold():
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {"r1": [_source("https://a.example")]}
    registry = [_registry("https://a.example", "vendor page")]
    result = x1.compute(_inputs(raw_runs, sources_by_run, registry))
    assert result.threshold is None


def test_modal_margin_and_weak_flag():
    raw_runs = [_run("r1", "openai_chatgpt")]
    # page: 42/100, listicle: 36/100 -- 6pp margin, below the 10pp weak threshold
    sources_by_run = {"r1": [_source(f"https://p{i}.example") for i in range(42)] + [_source(f"https://l{i}.example") for i in range(36)] + [_source(f"https://o{i}.example") for i in range(22)]}
    registry = (
        [_registry(f"https://p{i}.example", "vendor page") for i in range(42)]
        + [_registry(f"https://l{i}.example", "vendor listicle") for i in range(36)]
        + [_registry(f"https://o{i}.example", "other") for i in range(22)]
    )
    result = x1.compute(_inputs(raw_runs, sources_by_run, registry))
    assert result.notes["modal_class_by_engine"]["openai_chatgpt"] == "vendor page"
    assert abs(result.notes["modal_margin_by_engine"]["openai_chatgpt"] - 0.06) < 1e-9
    assert result.notes["weak_modal_by_engine"]["openai_chatgpt"] is True


def test_modal_margin_not_weak_with_decisive_gap():
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {"r1": [_source(f"https://p{i}.example") for i in range(90)] + [_source(f"https://l{i}.example") for i in range(10)]}
    registry = [_registry(f"https://p{i}.example", "vendor page") for i in range(90)] + [
        _registry(f"https://l{i}.example", "vendor listicle") for i in range(10)
    ]
    result = x1.compute(_inputs(raw_runs, sources_by_run, registry))
    assert result.notes["weak_modal_by_engine"]["openai_chatgpt"] is False


def test_raw_vendor_share_and_review_site_share():
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {"r1": [_source("https://a.example"), _source("https://b.example"), _source("https://c.example")]}
    registry = [
        _registry("https://a.example", "vendor page"),
        _registry("https://b.example", "vendor listicle"),
        _registry("https://c.example", "review site"),
    ]
    result = x1.compute(_inputs(raw_runs, sources_by_run, registry))
    assert abs(result.notes["raw_vendor_share_by_engine"]["openai_chatgpt"] - 2 / 3) < 1e-9
    assert abs(result.notes["review_site_share_by_engine"]["openai_chatgpt"] - 1 / 3) < 1e-9
