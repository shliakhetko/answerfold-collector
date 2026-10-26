import pytest

from analysis.phase_b.business import source_classes as sc
from analysis.phase_b.citation_panel import CitationRow


def _row(url, url_class="other", poc=False, engine="e1", run_id="r1"):
    return CitationRow(
        run_id=run_id, engine=engine, intent_id=1, paraphrase_id=1, position=1,
        url_normalized=url, domain=None, url_class=url_class,
        publisher_is_in_category=poc if url_class is not None else None,
        publisher_is_in_category_overridden=False, resolution_status=None, resolution_method=None,
    )


@pytest.fixture(scope="module")
def domain_map():
    return sc.load_domain_map()


@pytest.mark.parametrize(
    "domain,expected",
    [
        ("youtube.com", sc.VIDEO_SOCIAL),
        ("instagram.com", sc.VIDEO_SOCIAL),
        ("americanbar.org", sc.REGULATORS),
        ("fcc.gov", sc.REGULATORS),
        ("medium.com", sc.BLOGS),
        ("dev.to", sc.BLOGS),
        ("heyitsclara.com", sc.COMPANY_SITES),
        ("mconsent.net", sc.COMPANY_SITES),
        ("reddit.com", sc.FORUMS),
        ("g2.com", sc.REVIEW),
    ],
)
def test_map_covers_known_domains(domain_map, domain, expected):
    assert domain_map[domain] == expected


def test_map_has_every_required_class_and_only_valid_ones(domain_map):
    assert set(domain_map.values()) == sc.MAPPED_CLASSES


def test_domain_map_wins_over_raw_label(domain_map):
    assert sc.display_class(_row("https://www.youtube.com/watch?v=1", "other"), domain_map) == sc.VIDEO_SOCIAL
    assert sc.display_class(_row("https://www.facebook.com/x", "forum"), domain_map) == sc.VIDEO_SOCIAL


def test_vendor_published_is_never_relabelled(domain_map):
    assert sc.display_class(_row("https://www.youtube.com/x", "other", poc=True), domain_map) == sc.VENDOR


def test_raw_label_fallback_and_leftover_other(domain_map):
    assert sc.display_class(_row("https://unknown-a.example/x", "news"), domain_map) == sc.NEWS
    assert sc.display_class(_row("https://unknown-b.example/x", "vendor listicle"), domain_map) == sc.COMPANY_SITES
    assert sc.display_class(_row("https://unknown-c.com/x", "other"), domain_map) == sc.OTHER


def test_unclassified_citation_has_no_class(domain_map):
    assert sc.display_class(_row("https://x.example", None), domain_map) is None


def test_counts_sum_and_leftover_report(domain_map):
    rows = [
        _row("https://www.youtube.com/a"),
        _row("https://unknown-c.com/x"),
        _row("https://unknown-c.com/y"),
        _row("https://v.example/p", "vendor page", poc=True),
        _row("https://x.example", None),
    ]
    counts = sc.class_counts_by_engine(rows, domain_map)
    assert sum(counts["e1"].values()) == 4
    rep = sc.leftover_report(rows, domain_map)["e1"]
    assert rep["classified"] == 4
    assert rep["other"] == 2
    assert rep["other_share"] == 0.5
    assert rep["top_unmapped_domains"] == [("unknown-c.com", 2)]


def test_load_domain_map_rejects_unknown_class(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("domain,display_class\na.com,Nonsense\n")
    with pytest.raises(ValueError):
        sc.load_domain_map(p)
