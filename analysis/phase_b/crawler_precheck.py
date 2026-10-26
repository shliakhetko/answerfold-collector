"""Crawler eligibility pre-check (EXPERIMENT.md 3.4, "Not findings").

A NOTE, not a figure: serialized under figures.json `intermediates`. The
crawler list and the definition of "blocking" are fixed in the ledger
entry `crawler-precheck-definition-2026-09-29`, written before any count.

Blocking: a parseable robots.txt under which the standard robots-
exclusion evaluation (urllib.robotparser: the group for the token, else
the `*` group) disallows the site root "/" for that token. Missing files
and bodies that are not robots.txt (HTML block pages) are "unreadable",
never counted as blocking. A wildcard `Disallow: /` counts but is
reported apart from explicit token rules.
"""
from __future__ import annotations

from typing import Any
from urllib.robotparser import RobotFileParser

SEARCH_CRAWLERS = (
    "OAI-SearchBot", "ChatGPT-User", "Claude-SearchBot", "Claude-User",
    "PerplexityBot", "Perplexity-User", "Google-Extended",
)
TRAINING_CRAWLERS = ("GPTBot", "ClaudeBot", "CCBot")
ELIGIBILITY_KEEP_AT_OR_ABOVE = 0.10
LEDGER_REF = "crawler-precheck-definition-2026-09-29"


def _looks_like_html(text: str) -> bool:
    head = text.lstrip()[:200].lower()
    return head.startswith("<!doctype") or head.startswith("<html") or "<html" in head or head.startswith("<")


def parse_robots(text: str | None) -> tuple[RobotFileParser | None, str | None]:
    """(parser, None) when readable; (None, reason) when unreadable."""
    if text is None:
        return None, "no_robots_file"
    if not text.strip():
        return None, "empty_body"
    if _looks_like_html(text):
        return None, "html_body_not_robots_txt"
    parser = RobotFileParser()
    parser.parse(text.splitlines())
    return parser, None


def blocks_root(parser: RobotFileParser, token: str) -> tuple[bool, bool]:
    """(blocked, explicit): explicit is True when a group naming this
    token (not just `*`) exists."""
    blocked = not parser.can_fetch(token, "/")
    lowered = token.lower()
    explicit = any(
        ua != "*" and ua.lower() in lowered for entry in parser.entries for ua in entry.useragents
    )
    return blocked, explicit


def domain_verdict(text: str | None) -> dict[str, Any]:
    parser, reason = parse_robots(text)
    if parser is None:
        return {"readable": False, "unreadable_reason": reason, "search_blocked": [], "search_blocked_explicit": [],
                "training_blocked": []}
    search, search_explicit, training = [], [], []
    for token in SEARCH_CRAWLERS:
        blocked, explicit = blocks_root(parser, token)
        if blocked:
            search.append(token)
            if explicit:
                search_explicit.append(token)
    for token in TRAINING_CRAWLERS:
        blocked, _ = blocks_root(parser, token)
        if blocked:
            training.append(token)
    return {"readable": True, "unreadable_reason": None, "search_blocked": search,
            "search_blocked_explicit": search_explicit, "training_blocked": training}


def crawler_precheck(roster_rows: list[dict[str, Any]], robots_by_domain: dict[str, str | None]) -> dict[str, Any]:
    n_roster = len(roster_rows)
    no_domain = 0
    domain_not_collected = 0
    unreadable: dict[str, int] = {}
    readable = 0
    blocking_any = 0
    blocking_explicit_any = 0
    blocking_wildcard_only = 0
    training_only_blocking = 0
    by_crawler = {t: 0 for t in SEARCH_CRAWLERS}
    blocking_companies: list[str] = []
    for row in roster_rows:
        domain = (row.get("primary_domain") or "").strip()
        if not domain:
            no_domain += 1
            continue
        if domain not in robots_by_domain:
            domain_not_collected += 1
            continue
        verdict = domain_verdict(robots_by_domain[domain])
        if not verdict["readable"]:
            unreadable[verdict["unreadable_reason"]] = unreadable.get(verdict["unreadable_reason"], 0) + 1
            continue
        readable += 1
        if verdict["search_blocked"]:
            blocking_any += 1
            blocking_companies.append(row.get("entity_id") or domain)
            for t in verdict["search_blocked"]:
                by_crawler[t] += 1
            if verdict["search_blocked_explicit"]:
                blocking_explicit_any += 1
            else:
                blocking_wildcard_only += 1
        elif verdict["training_blocked"]:
            training_only_blocking += 1
    share = (blocking_any / readable) if readable else None
    keep = None if share is None else share >= ELIGIBILITY_KEEP_AT_OR_ABOVE
    return {
        "note_not_a_figure": True,
        "ledger_ref": LEDGER_REF,
        "search_crawler_list": list(SEARCH_CRAWLERS),
        "training_crawler_list_context_only": list(TRAINING_CRAWLERS),
        "blocking_definition": (
            "parseable robots.txt whose group for the token (else the * group) disallows the site root '/'; "
            "missing files and HTML block pages are unreadable, not blocking"
        ),
        "n_roster_rows": n_roster,
        "n_no_domain": no_domain,
        "n_domain_not_collected": domain_not_collected,
        "n_unreadable_by_reason": unreadable,
        "n_readable_robots": readable,
        "n_blocking_at_least_one_search_crawler": blocking_any,
        "n_of_those_with_explicit_token_rule": blocking_explicit_any,
        "n_of_those_wildcard_only": blocking_wildcard_only,
        "n_blocking_only_training_crawlers": training_only_blocking,
        "n_blocking_by_search_crawler": by_crawler,
        "blocking_company_ids": sorted(blocking_companies),
        "eligibility_keep_rule": f"keep the eligibility section if blocking companies are >= {ELIGIBILITY_KEEP_AT_OR_ABOVE:.0%} of readable-robots companies",
        "eligibility_section_stays": keep,
    }
