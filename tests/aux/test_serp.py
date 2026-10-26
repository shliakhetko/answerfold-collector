from collector.config import SerpApiConfig
from analysis.aux.serp import build_payload, run_search
from tests.fakes import FakeSerpApiClient

CFG = SerpApiConfig(api_key="test-key")


def test_payload_pins_hard_rule_5_params():
    p = build_payload("best ai receptionist")
    assert p["engine"] == "google"
    assert p["location"] == "United States"
    assert p["gl"] == "us" and p["hl"] == "en" and p["device"] == "desktop"
    assert p["no_cache"] is True
    assert p["num"] == 10


def test_run_search_stores_full_raw_and_never_leaks_key():
    payload_seen = {}

    class Client(FakeSerpApiClient):
        def search(self, params):
            payload_seen.update(params)
            params["api_key"] = "leaked"
            return {"search_metadata": {"status": "Success"}, "organic_results": [{"link": "https://a.com", "position": 1}]}

    r = run_search("q", config=CFG, client=Client())
    assert r.ok
    assert r.raw["organic_results"][0]["link"] == "https://a.com"
    assert "api_key" not in r.request_payload  # a copy was passed to the client


def test_run_search_reports_logical_failure():
    client = FakeSerpApiClient(response={"search_metadata": {"status": "Error"}, "error": "Invalid query"})
    r = run_search("q", config=CFG, client=client)
    assert r.ok is False
    assert r.error == "Invalid query"
    assert r.raw is not None  # envelope still kept


def test_run_search_never_raises_on_transport_error():
    client = FakeSerpApiClient(exc=RuntimeError("boom"))
    r = run_search("q", config=CFG, client=client)
    assert r.ok is False
    assert "RuntimeError: boom" in r.error
