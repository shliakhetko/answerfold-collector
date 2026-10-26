"""Pre-extraction text cleanup for A1 (analysis/coding/a1_answer_coding.py).

strip_markdown_links() removes inline citation link markup
(`[Fathom](https://...)` -> `Fathom`) before sending response text to the
extractors -- the URL itself carries no extraction signal (company/product
names live in the link text, not the href) but does add real token cost,
most visibly for google_ai_mode (heaviest inline-link usage of the 4
engines). Anchor text is always preserved verbatim, so no company name is
ever dropped by this step.

Whatever text an extractor actually saw must also be what
a1_answer_coding.py::verify_candidate checks evidence_span against --
run_a1 applies this once per run and reuses the same cleaned string for
both, so extraction and verification never disagree on what "the text" is.
"""
from __future__ import annotations

import re

# Non-greedy link text, http(s):// or root-relative hrefs only -- deliberately
# narrow (won't touch "[note]" that isn't followed by "(...)", won't touch a
# literal "(see https://...)" aside that isn't itself link syntax).
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:https?://|/)[^)]+\)")


def strip_markdown_links(text: str) -> str:
    return _MARKDOWN_LINK_RE.sub(r"\1", text)
