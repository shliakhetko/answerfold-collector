from analysis.coding.url_normalize import normalize_url


def test_strips_query_string():
    assert normalize_url("https://example.com/pricing?utm_source=google&ref=x") == "https://example.com/pricing"


def test_drops_trailing_slash():
    assert normalize_url("https://example.com/pricing/") == "https://example.com/pricing"


def test_lowercases_host():
    assert normalize_url("https://EXAMPLE.com/Pricing") == "https://example.com/Pricing"


def test_removes_www_prefix():
    assert normalize_url("https://www.example.com/") == "https://example.com"


def test_root_url_has_no_trailing_slash():
    assert normalize_url("https://example.com") == "https://example.com"
    assert normalize_url("https://example.com/") == "https://example.com"


def test_drops_fragment():
    assert normalize_url("https://example.com/pricing#section") == "https://example.com/pricing"


def test_two_urls_that_should_be_the_same_domain_normalize_identically():
    a = normalize_url("https://WWW.Goodcall.com/pricing/?utm_source=chatgpt")
    b = normalize_url("https://goodcall.com/pricing")
    assert a == b
