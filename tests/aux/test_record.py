from datetime import datetime, timezone

from analysis.aux.record import (
    KYIV_TZ,
    WINDOW_CLOSE_DATE,
    days_after_window_close,
    make_envelope,
    wrap,
)


def test_days_after_window_close_is_three_on_2026_08_29():
    when = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
    assert days_after_window_close(when) == 3


def test_days_after_window_close_zero_on_close_day():
    assert days_after_window_close(datetime(2026, 8, 26, 23, 0, tzinfo=timezone.utc)) == 0


def test_envelope_has_every_required_field():
    now = datetime(2026, 8, 29, 14, 3, tzinfo=timezone.utc)
    env = make_envelope("collect_cited_urls", now=now)
    assert env["post_window"] is True
    assert env["days_after_window_close"] == 3
    assert env["collector"] == "collect_cited_urls"
    assert env["collected_at"] == "2026-08-29T14:03:00+00:00"
    assert env["collected_at_kyiv"] == now.astimezone(KYIV_TZ).isoformat()
    assert env["collector_version"].startswith("geo-collector/0.1.0+")


def test_wrap_merges_fields_and_keeps_raw_verbatim():
    raw = {"nested": [1, 2, 3], "b": "x"}
    rec = wrap("collect_robots", {"domain": "example.com"}, raw, now=datetime(2026, 8, 29, tzinfo=timezone.utc))
    assert rec["domain"] == "example.com"
    assert rec["raw"] == raw
    assert rec["collector"] == "collect_robots"


def test_window_close_date_constant():
    assert (WINDOW_CLOSE_DATE.year, WINDOW_CLOSE_DATE.month, WINDOW_CLOSE_DATE.day) == (2026, 8, 26)
