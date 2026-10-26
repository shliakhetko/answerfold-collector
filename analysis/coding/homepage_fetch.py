"""Real HomepageFetcher (analysis/coding/a4_entity_classification.py's
`HomepageFetcher` Protocol) -- one GET plus stdlib HTML-to-text. No new
dependency: matches analysis/coding/cache.py's documented "no dependency
where a few lines of stdlib solves it" style, and this repo has no HTTP
library at all yet (every vendor call goes through its own SDK).

Never raises -- same adapter contract as every collector/engines/*.py
adapter and every analysis/coding/*.py call_* function: on any failure
(timeout, non-2xx, non-HTML response, decode failure) fetch() returns "",
so A4's classification prompt just sees an empty homepage rather than the
whole pipeline crashing on one bad URL.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any, Callable
from urllib.request import Request, urlopen

DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_CHARS = 20_000
USER_AGENT = "geo-collector-research/0.1 (+study data pipeline, non-commercial)"

_SKIP_TAGS = {"script", "style"}

# Block-level tags force a word boundary in the extracted text even when the
# source HTML has no whitespace between them (e.g. "<h1>Goodcall</h1><p>AI
# receptionist</p>") -- without this, adjacent block elements' text would
# run together into one word.
_BLOCK_TAGS = {
    "p", "div", "br", "li", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6",
    "header", "footer", "section", "article", "nav", "ul", "ol", "table", "blockquote",
}


class _TextExtractor(HTMLParser):
    """Strips tags, drops <script>/<style> contents, collapses whitespace."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        if tag in _BLOCK_TAGS:
            self._chunks.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag in _BLOCK_TAGS:
            self._chunks.append(" ")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._chunks.append(data)

    def text(self) -> str:
        return re.sub(r"\s+", " ", "".join(self._chunks)).strip()


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.text()


class RealHomepageFetcher:
    """`opener` is injectable (default `urllib.request.urlopen`) so tests
    never touch a real socket -- same DI pattern as this package's
    MessagesClient/ResponsesClient Protocols, just for HTTP instead of an
    SDK client."""

    def __init__(
        self,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_chars: int = DEFAULT_MAX_CHARS,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_chars = max_chars
        self._opener = opener

    def fetch(self, homepage_url: str) -> str:
        try:
            request = Request(homepage_url, headers={"User-Agent": USER_AGENT})
            with self._opener(request, timeout=self.timeout_seconds) as response:
                content_type = ""
                headers = getattr(response, "headers", None)
                if headers is not None:
                    content_type = headers.get("Content-Type", "") or ""
                if content_type and "html" not in content_type.lower():
                    return ""
                raw = response.read(self.max_chars * 8)  # cap bytes read before decode
            decoded = raw.decode("utf-8", errors="replace")
            return html_to_text(decoded)[: self.max_chars]
        except Exception:  # contract: never raise past this point
            return ""
