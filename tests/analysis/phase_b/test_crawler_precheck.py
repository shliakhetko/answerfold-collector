from analysis.phase_b.crawler_precheck import crawler_precheck, domain_verdict


def test_explicit_token_block():
    v = domain_verdict("User-agent: OAI-SearchBot\nDisallow: /\n")
    assert v["readable"] and v["search_blocked"] == ["OAI-SearchBot"] and v["search_blocked_explicit"] == ["OAI-SearchBot"]


def test_wildcard_block_counts_but_not_explicit():
    v = domain_verdict("User-agent: *\nDisallow: /\n")
    assert len(v["search_blocked"]) == 7 and v["search_blocked_explicit"] == []


def test_partial_disallow_is_not_blocking():
    v = domain_verdict("User-agent: *\nDisallow: /private\n")
    assert v["readable"] and v["search_blocked"] == []


def test_html_missing_and_empty_are_unreadable():
    assert domain_verdict("<!DOCTYPE html><html>blocked</html>")["unreadable_reason"] == "html_body_not_robots_txt"
    assert domain_verdict(None)["unreadable_reason"] == "no_robots_file"
    assert domain_verdict("  \n")["unreadable_reason"] == "empty_body"


def test_training_only_block_not_in_headline():
    v = domain_verdict("User-agent: GPTBot\nDisallow: /\n")
    assert v["search_blocked"] == [] and v["training_blocked"] == ["GPTBot"]


def test_precheck_counts_and_decision():
    roster = [{"entity_id": f"c{i}", "primary_domain": f"d{i}.com"} for i in range(4)] + [{"entity_id": "x", "primary_domain": ""}]
    robots = {
        "d0.com": "User-agent: ClaudeBot\nDisallow: /\n\nUser-agent: Claude-User\nDisallow: /\n",
        "d1.com": "User-agent: *\nDisallow:\n",
        "d2.com": "<html>nope</html>",
        "d3.com": None,
    }
    r = crawler_precheck(roster, robots)
    assert r["n_readable_robots"] == 2
    assert r["n_blocking_at_least_one_search_crawler"] == 1
    assert r["n_no_domain"] == 1
    assert r["n_unreadable_by_reason"] == {"html_body_not_robots_txt": 1, "no_robots_file": 1}
    assert r["eligibility_section_stays"] is True
