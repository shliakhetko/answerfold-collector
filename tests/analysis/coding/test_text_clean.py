from analysis.coding.text_clean import strip_markdown_links


def test_strips_link_keeping_anchor_text():
    text = "[Fathom](https://fathom.video/) is the overall best free option."
    assert strip_markdown_links(text) == "Fathom is the overall best free option."


def test_strips_multiple_links():
    text = "Try [Fathom](https://fathom.video/) or [Fireflies.ai](https://fireflies.ai/pricing)."
    assert strip_markdown_links(text) == "Try Fathom or Fireflies.ai."


def test_root_relative_href_also_stripped():
    text = "See [pricing](/pricing) for details."
    assert strip_markdown_links(text) == "See pricing for details."


def test_text_with_no_links_unchanged():
    text = "Fathom is the overall best free option for individual users."
    assert strip_markdown_links(text) == text


def test_bracketed_text_without_parenthetical_url_left_alone():
    # Not link syntax -- no trailing (url) -- must not be touched.
    text = "Deep [HubSpot] integration was noted."
    assert strip_markdown_links(text) == text


def test_non_link_parenthetical_aside_left_alone():
    text = "Fathom (see https://fathom.video/ for pricing) is popular."
    assert strip_markdown_links(text) == text
