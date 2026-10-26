from analysis.coding.b3b_serp_derivation import derive_serp_binary

ENTITY_REGISTRY_ROWS = [
    {
        "entity_id": "goodcall", "canonical_name": "Goodcall", "surface_forms": "Goodcall",
        "homepage_url": "https://goodcall.com/", "meets_frame": "in",
    },
    {
        "entity_id": "metro-answering", "canonical_name": "Metro Answering Service",
        "surface_forms": "Metro Answering Service", "homepage_url": "https://metroanswering.example.com/",
        "meets_frame": "out",  # out-of-frame -- must never get a row
    },
]


def test_derive_serp_binary_matches_homepage_domain():
    raw_serp_rows = [
        {"intent_id": 1, "position": 1, "url": "https://goodcall.com/pricing"},
        {"intent_id": 1, "position": 2, "url": "https://unrelated.example.com/"},
    ]
    result = derive_serp_binary(raw_serp_rows, ENTITY_REGISTRY_ROWS, alt_domains_rows=[])
    assert result == [{"entity_id": "goodcall", "intent_id": 1, "in_google_top10": True}]


def test_derive_serp_binary_matches_alt_domain():
    raw_serp_rows = [{"intent_id": 2, "position": 1, "url": "https://goodcall.io/blog"}]
    result = derive_serp_binary(
        raw_serp_rows, ENTITY_REGISTRY_ROWS, alt_domains_rows=[{"entity_id": "goodcall", "alt_domain": "goodcall.io"}]
    )
    assert result == [{"entity_id": "goodcall", "intent_id": 2, "in_google_top10": True}]


def test_derive_serp_binary_excludes_out_of_frame_entities():
    raw_serp_rows = [{"intent_id": 1, "position": 1, "url": "https://metroanswering.example.com/"}]
    result = derive_serp_binary(raw_serp_rows, ENTITY_REGISTRY_ROWS, alt_domains_rows=[])
    assert result == []


def test_derive_serp_binary_no_matches_is_empty():
    raw_serp_rows = [{"intent_id": 1, "position": 1, "url": "https://nowhere.example.com/"}]
    assert derive_serp_binary(raw_serp_rows, ENTITY_REGISTRY_ROWS, alt_domains_rows=[]) == []


def test_derive_serp_binary_one_row_per_entity_intent_pair_even_with_multiple_urls():
    raw_serp_rows = [
        {"intent_id": 1, "position": 1, "url": "https://goodcall.com/"},
        {"intent_id": 1, "position": 3, "url": "https://goodcall.com/pricing"},
    ]
    result = derive_serp_binary(raw_serp_rows, ENTITY_REGISTRY_ROWS, alt_domains_rows=[])
    assert result == [{"entity_id": "goodcall", "intent_id": 1, "in_google_top10": True}]
