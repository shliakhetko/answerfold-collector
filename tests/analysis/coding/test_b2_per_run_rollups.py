from analysis.coding.b1_per_run_company import build_per_run_company
from analysis.coding.b2_per_run_rollups import build_per_run_rollups

ENTITY_REGISTRY_ROWS = [
    {
        "entity_id": "goodcall", "canonical_name": "Goodcall", "surface_forms": "Goodcall",
        "homepage_url": "https://goodcall.com/", "meets_frame": "in",
    },
    {
        "entity_id": "vapi", "canonical_name": "Vapi", "surface_forms": "Vapi",
        "homepage_url": "https://vapi.ai/", "meets_frame": "in",
    },
    {
        "entity_id": "metro-answering", "canonical_name": "Metro Answering Service",
        "surface_forms": "Metro Answering Service", "homepage_url": "https://metroanswering.example.com/",
        "meets_frame": "out",
    },
]

RUN_ROWS = [
    {
        "run_id": "run-1",
        "raw_response_text": "Unlike Vapi, which needs setup, Goodcall works out of the box.",
    },
    {"run_id": "run-2", "raw_response_text": "Metro Answering Service is a live human option."},
]

SOURCES_BY_RUN = {
    "run-1": [
        {"url_raw": "https://goodcall.com/pricing", "domain": "goodcall.com", "title": "Goodcall Pricing"},
        {"url_raw": "https://reviews.example.com/goodcall", "domain": "reviews.example.com", "title": "Review"},
    ],
    "run-2": [],
}

MENTIONS_RAW_ROWS = [
    {"run_id": "run-1", "surface_form": "Goodcall", "evidence_span": "Goodcall works out of the box.", "is_recommended": True, "votes": ["a", "b"]},
    {"run_id": "run-1", "surface_form": "Vapi", "evidence_span": "Unlike Vapi, which needs setup,", "is_recommended": False, "votes": ["a", "b"]},
    {"run_id": "run-2", "surface_form": "Metro Answering Service", "evidence_span": "Metro Answering Service is a live human option.", "is_recommended": True, "votes": ["a", "b"]},
]


def _per_run_company_rows():
    return build_per_run_company(RUN_ROWS, MENTIONS_RAW_ROWS, ENTITY_REGISTRY_ROWS, SOURCES_BY_RUN, alt_domains_rows=[])


def test_n_companies_mentioned_counts_in_frame_only():
    rollups = build_per_run_rollups(
        _per_run_company_rows(), RUN_ROWS, MENTIONS_RAW_ROWS, SOURCES_BY_RUN, ENTITY_REGISTRY_ROWS
    )
    by_run = {r.run_id: r for r in rollups}
    assert by_run["run-1"].n_companies_mentioned == 2  # Goodcall + Vapi
    assert by_run["run-2"].n_companies_mentioned == 0  # Metro is out-of-frame, doesn't count


def test_n_sources_matches_sources_table():
    rollups = build_per_run_rollups(
        _per_run_company_rows(), RUN_ROWS, MENTIONS_RAW_ROWS, SOURCES_BY_RUN, ENTITY_REGISTRY_ROWS
    )
    by_run = {r.run_id: r for r in rollups}
    assert by_run["run-1"].n_sources == 2
    assert by_run["run-2"].n_sources == 0


def test_vendor_source_share_uses_url_class_lookup():
    url_class_by_domain = {"goodcall.com": "vendor page", "reviews.example.com": "review site"}
    rollups = build_per_run_rollups(
        _per_run_company_rows(), RUN_ROWS, MENTIONS_RAW_ROWS, SOURCES_BY_RUN, ENTITY_REGISTRY_ROWS, url_class_by_domain
    )
    by_run = {r.run_id: r for r in rollups}
    assert by_run["run-1"].vendor_source_share == 0.5  # 1 of 2 sources is a vendor page
    assert by_run["run-2"].vendor_source_share is None  # no sources -- not 0.0


def test_n_out_of_frame_brands_is_a_count_only_never_a_denominator_input():
    rollups = build_per_run_rollups(
        _per_run_company_rows(), RUN_ROWS, MENTIONS_RAW_ROWS, SOURCES_BY_RUN, ENTITY_REGISTRY_ROWS
    )
    by_run = {r.run_id: r for r in rollups}
    assert by_run["run-1"].n_out_of_frame_brands == 0
    assert by_run["run-2"].n_out_of_frame_brands == 1  # Metro Answering Service
