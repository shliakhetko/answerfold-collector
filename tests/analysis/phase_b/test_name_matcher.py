from analysis.phase_b.name_matcher import build_matcher, match_entities


def _e(eid, name, sf="", frame="in"):
    return {"entity_id": eid, "canonical_name": name, "surface_forms": sf or name, "meets_frame": frame}


def test_case_insensitive_for_long_names_and_boundaries():
    m = build_matcher([_e("smith", "Smith.ai")])
    assert match_entities("Try SMITH.AI today", m) == {"smith"}
    assert match_entities("smith.aid", m) == set()


def test_short_names_are_case_sensitive():
    m = build_matcher([_e("edge", "Edge")])
    assert match_entities("the leading edge of", m) == set()
    assert match_entities("Edge is a vendor", m) == {"edge"}


def test_only_in_frame_and_surface_forms():
    m = build_matcher([_e("a", "Acme Corp", "Acme Corp;Acme"), _e("b", "Bravo Inc", frame="out")])
    assert match_entities("Acme and Bravo Inc", m) == {"a"}


def test_whitespace_and_nfkc():
    m = build_matcher([_e("a", "Acme Corp")])
    assert match_entities("Acme   Corp", m) == {"a"}
