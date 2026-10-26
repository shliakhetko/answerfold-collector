import pytest

from analysis.coding.b3_aggregates import EligibilityReport
from analysis.coding.b4_figures import (
    CONFIRMATORY_FIGURE_IDS,
    EXPLORATORY_FIGURE_IDS,
    FIGURE_IDS,
    TABLE_FIGURE_IDS,
    AggregatesBundle,
    compute_figure,
    figure_tier,
)


def _empty_bundle():
    return AggregatesBundle(
        per_run_company_rows=[], per_run_rollup_rows=[], cell_rates=[], eligibility=EligibilityReport()
    )


def test_figure_ids_cover_every_spec_named_id():
    assert set(FIGURE_IDS) == {
        "S1", "A1", "V1", "S2", "S3", "K1", "K2", "O1", "V2", "A2", "A3", "O2", "X1",
    }
    # No overlap between tiers.
    assert len(FIGURE_IDS) == len(CONFIRMATORY_FIGURE_IDS) + len(TABLE_FIGURE_IDS) + len(EXPLORATORY_FIGURE_IDS)


@pytest.mark.parametrize("figure_id", CONFIRMATORY_FIGURE_IDS)
def test_confirmatory_tier(figure_id):
    assert figure_tier(figure_id) == "confirmatory"


@pytest.mark.parametrize("figure_id", TABLE_FIGURE_IDS)
def test_table_tier(figure_id):
    assert figure_tier(figure_id) == "table"


@pytest.mark.parametrize("figure_id", EXPLORATORY_FIGURE_IDS)
def test_exploratory_tier(figure_id):
    assert figure_tier(figure_id) == "exploratory"


def test_unknown_figure_id_raises_key_error():
    with pytest.raises(KeyError):
        figure_tier("Z99")


@pytest.mark.parametrize("figure_id", FIGURE_IDS)
def test_compute_figure_raises_not_implemented_for_every_known_id(figure_id):
    with pytest.raises(NotImplementedError, match=figure_id):
        compute_figure(figure_id, _empty_bundle())


def test_compute_figure_unknown_id_raises_key_error_not_not_implemented():
    with pytest.raises(KeyError):
        compute_figure("Z99", _empty_bundle())
