"""The AI-crawler user-agent tokens whose access policy Phase A will read out
of the robots.txt / llms.txt files Task 4 stores verbatim.

Recorded here (and copied into data/02_aux/collector_config.json at run time) so
the later parse is reproducible against a frozen list. Task 4 does NOT parse
robots.txt into allow/block booleans -- it stores raw. This list is config
for that future parse, nothing more.
"""
from __future__ import annotations

AI_CRAWLER_AGENTS: tuple[str, ...] = (
    # OpenAI
    "GPTBot",
    "OAI-SearchBot",
    "ChatGPT-User",
    # Anthropic
    "ClaudeBot",
    "Claude-SearchBot",
    "Claude-User",
    # Google
    "Google-Extended",
    "Googlebot",
    # Perplexity
    "PerplexityBot",
    "Perplexity-User",
    # Microsoft / Bing
    "Bingbot",
    # Apple
    "Applebot",
    "Applebot-Extended",
    # Common Crawl
    "CCBot",
    # Meta
    "Meta-ExternalAgent",
    # Amazon
    "Amazonbot",
    # ByteDance
    "Bytespider",
)
