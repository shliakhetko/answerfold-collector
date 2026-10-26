"""Fakes for exercising the OpenAI adapter without network access or a real
API key. Mirror the shapes pinned in collector/engines/openai_chatgpt.py --
see that module's docstring for the re-verification caveat."""
from __future__ import annotations

from types import SimpleNamespace


def fake_success_response() -> SimpleNamespace:
    return SimpleNamespace(
        model="gpt-4o-search-preview",
        status="completed",
        incomplete_details=None,
        output_text="Paris is the capital of France.",
        usage=SimpleNamespace(
            model_dump=lambda: {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}
        ),
        output=[
            SimpleNamespace(
                type="web_search_call",
                action=SimpleNamespace(type="search", query="capital of France", queries=None),
            ),
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(
                        type="output_text",
                        text="Paris is the capital of France.",
                        annotations=[
                            SimpleNamespace(
                                type="url_citation",
                                url="https://en.wikipedia.org/wiki/Paris",
                                title="Paris - Wikipedia",
                            )
                        ],
                    )
                ],
            ),
        ],
        model_dump_json=lambda: '{"id": "resp_fake", "status": "completed"}',
    )


class FakeResponses:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc

    def create(self, **kwargs):
        if self._exc is not None:
            raise self._exc
        return self._response


class FakeClient:
    def __init__(self, responses: FakeResponses):
        self.responses = responses


def fake_gemini_success_response() -> SimpleNamespace:
    return SimpleNamespace(
        model_version="gemini-2.5-flash",
        text="Paris is the capital of France.",
        usage_metadata=SimpleNamespace(
            model_dump=lambda mode="python": {
                "prompt_token_count": 12,
                "candidates_token_count": 8,
                "total_token_count": 20,
            }
        ),
        candidates=[
            SimpleNamespace(
                finish_reason=SimpleNamespace(value="STOP"),
                grounding_metadata=SimpleNamespace(
                    web_search_queries=["capital of France"],
                    grounding_chunks=[
                        SimpleNamespace(
                            web=SimpleNamespace(
                                uri="https://en.wikipedia.org/wiki/Paris",
                                title="Paris - Wikipedia",
                                domain="en.wikipedia.org",
                            )
                        )
                    ],
                ),
            )
        ],
        model_dump_json=lambda: '{"model_version": "gemini-2.5-flash"}',
    )


class FakeGeminiModels:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc

    def generate_content(self, **kwargs):
        if self._exc is not None:
            raise self._exc
        return self._response


class FakeGeminiClient:
    def __init__(self, models: FakeGeminiModels):
        self.models = models


def fake_claude_success_response() -> SimpleNamespace:
    return SimpleNamespace(
        model="claude-opus-5",
        stop_reason="end_turn",
        usage=SimpleNamespace(
            model_dump=lambda: {
                "input_tokens": 12,
                "output_tokens": 8,
                "server_tool_use": {"web_search_requests": 1},
            }
        ),
        content=[
            SimpleNamespace(
                type="server_tool_use",
                name="web_search",
                input={"query": "capital of France"},
            ),
            SimpleNamespace(
                type="web_search_tool_result",
                tool_use_id="srvtoolu_fake",
                content=[
                    SimpleNamespace(
                        type="web_search_result",
                        url="https://en.wikipedia.org/wiki/Paris",
                        title="Paris - Wikipedia",
                    )
                ],
            ),
            SimpleNamespace(
                type="text",
                text="Paris is the capital of France.",
                citations=[
                    SimpleNamespace(
                        type="web_search_result_location",
                        url="https://en.wikipedia.org/wiki/Paris",
                        title="Paris - Wikipedia",
                    )
                ],
            ),
        ],
        model_dump_json=lambda: '{"id": "msg_fake", "stop_reason": "end_turn"}',
    )


class FakeMessages:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self._exc is not None:
            raise self._exc
        return self._response


class FakeAnthropicClient:
    def __init__(self, messages: FakeMessages):
        self.messages = messages


def fake_google_ai_mode_success_response() -> dict:
    # The real serpapi.SerpResults is a dict-like UserDict; a plain dict
    # exercises the same .get()-based access the adapter uses.
    return {
        "search_metadata": {"status": "Success"},
        "reconstructed_markdown": "Paris is the capital of France.",
        "text_blocks": [{"type": "paragraph", "snippet": "Paris is the capital of France."}],
        "references": [
            {
                "title": "Paris - Wikipedia",
                "link": "https://en.wikipedia.org/wiki/Paris",
                "snippet": "Paris is the capital and most populous city of France.",
                "index": 1,
            }
        ],
    }


def fake_anthropic_extraction_response(
    mentions_json: str, *, model: str = "claude-haiku-4-5-20251001", input_tokens: int = 100, output_tokens: int = 40
) -> SimpleNamespace:
    """A1 extractor response shape (analysis/coding/extractors/anthropic_extractor.py)
    -- not the web-search-tool shape fake_claude_success_response() models."""
    return SimpleNamespace(
        model=model,
        content=[SimpleNamespace(type="text", text=mentions_json)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def fake_openai_extraction_response(
    mentions_json: str, *, model: str = "gpt-5-mini", input_tokens: int = 100, output_tokens: int = 40
) -> SimpleNamespace:
    """A1 extractor response shape (analysis/coding/extractors/openai_extractor.py)
    -- not the web-search-tool shape fake_success_response() models."""
    return SimpleNamespace(
        model=model,
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=mentions_json)],
            )
        ],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def fake_gemini_extraction_response(
    mentions_json: str,
    *,
    model_version: str = "gemini-flash-lite-latest",
    input_tokens: int = 100,
    output_tokens: int = 40,
) -> SimpleNamespace:
    """A1 extractor response shape (analysis/coding/extractors/gemini_extractor.py)
    -- not the search-grounded shape fake_gemini_success_response() models."""
    return SimpleNamespace(
        model_version=model_version,
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text=mentions_json)]))],
        usage_metadata=SimpleNamespace(prompt_token_count=input_tokens, candidates_token_count=output_tokens),
    )


def fake_entity_resolution_response(
    entities_json: str, *, model: str = "claude-haiku-4-5-20251001", input_tokens: int = 200, output_tokens: int = 80
) -> SimpleNamespace:
    """A3 entity-resolution response shape (analysis/coding/a3_entity_resolution.py)."""
    return SimpleNamespace(
        model=model,
        content=[SimpleNamespace(type="text", text=entities_json)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def fake_classification_response(
    classification_json: str, *, model: str = "claude-haiku-4-5-20251001", input_tokens: int = 300, output_tokens: int = 30
) -> SimpleNamespace:
    """A4 entity-classification response shape (analysis/coding/a4_entity_classification.py)."""
    return SimpleNamespace(
        model=model,
        content=[SimpleNamespace(type="text", text=classification_json)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class FakeHomepageFetcher:
    """analysis/coding/a4_entity_classification.py's HomepageFetcher Protocol
    -- no real HTTP fetch dependency exists in this repo yet."""

    def __init__(self, pages: dict[str, str] | None = None):
        self._pages = pages or {}

    def fetch(self, homepage_url: str) -> str:
        return self._pages.get(homepage_url, "")


def fake_arbiter_response(
    votes_json: str, *, model: str = "claude-haiku-4-5-20251001", input_tokens: int = 150, output_tokens: int = 30
) -> SimpleNamespace:
    """A1 arbiter response shape (analysis/coding/extractors/arbiter.py)."""
    return SimpleNamespace(
        model=model,
        content=[SimpleNamespace(type="text", text=votes_json)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def fake_url_classification_response(
    classification_json: str, *, model: str = "claude-haiku-4-5-20251001", input_tokens: int = 150, output_tokens: int = 20
) -> SimpleNamespace:
    """A5 URL-classification response shape (analysis/coding/a5_url_classification.py)."""
    return SimpleNamespace(
        model=model,
        content=[SimpleNamespace(type="text", text=classification_json)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def fake_entity_verification_response(
    verification_json: str, *, model: str = "claude-sonnet-5", input_tokens: int = 400, output_tokens: int = 60,
    preamble: str = "",
) -> SimpleNamespace:
    """A4-prime / a3b entity-verification response shape
    (analysis/coding/a3b_entity_verification.py). `preamble` prepends
    narration before the JSON, as a search-using model tends to do."""
    return SimpleNamespace(
        model=model,
        content=[SimpleNamespace(type="text", text=f"{preamble}{verification_json}")],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class FakeHTTPResponse:
    """Context-manager response object matching urllib.request.urlopen's
    return value shape -- analysis/coding/homepage_fetch.py's
    RealHomepageFetcher injects an `opener` callable in tests instead of
    hitting a real socket."""

    def __init__(self, *, body: bytes, content_type: str = "text/html; charset=utf-8"):
        self._body = body
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self, n: int = -1) -> bytes:
        return self._body


def fake_opener(response: FakeHTTPResponse | None = None, exc: BaseException | None = None):
    """Returns a callable matching RealHomepageFetcher's `opener(request,
    timeout=...)` call shape."""

    def opener(request, timeout=None):
        if exc is not None:
            raise exc
        return response

    return opener


class FakeAnthropicBatches:
    """Fakes client.messages.batches.{create,retrieve,results} -- the
    Anthropic Message Batches API shape
    analysis/coding/extractors/anthropic_extractor.py's
    submit_anthropic_batch/poll_anthropic_batch/fetch_anthropic_batch_results
    (and arbiter.py's arbiter-batch equivalents) use. `statuses` lets a test
    drive the poll loop through multiple retrieve() calls before going
    terminal -- the last entry repeats once exhausted."""

    def __init__(self, *, batch_id="batch_fake", statuses=("ended",), results=None):
        self.batch_id = batch_id
        self._statuses = list(statuses)
        self._retrieve_calls = 0
        self.results_list = results or []
        self.create_kwargs = None

    def create(self, **kwargs):
        self.create_kwargs = kwargs
        return SimpleNamespace(id=self.batch_id)

    def retrieve(self, batch_id):
        idx = min(self._retrieve_calls, len(self._statuses) - 1)
        self._retrieve_calls += 1
        return SimpleNamespace(processing_status=self._statuses[idx])

    def results(self, batch_id):
        return self.results_list


class FakeAnthropicBatchClient:
    def __init__(self, batches: FakeAnthropicBatches):
        self.messages = SimpleNamespace(batches=batches)


def fake_anthropic_batch_result_entry(*, custom_id: str, text_json: str, model: str = "claude-haiku-4-5-20251001"):
    """One MessageBatchIndividualResponse-shaped succeeded entry, as
    fetch_anthropic_batch_results/fetch_arbiter_batch_results iterate over.
    text_json is whatever the message's text content should be -- mentions
    JSON for the extractor's batch, votes JSON for the arbiter's; the
    envelope shape (model/content/usage) is identical either way, only the
    downstream parser (parse_mentions_json vs parse_votes_json) differs."""
    message = fake_anthropic_extraction_response(text_json, model=model)
    return SimpleNamespace(custom_id=custom_id, result=SimpleNamespace(type="succeeded", message=message))


def fake_anthropic_batch_error_entry(*, custom_id: str, result_type: str = "errored"):
    return SimpleNamespace(custom_id=custom_id, result=SimpleNamespace(type=result_type, message=None))


class FakeOpenAIFiles:
    """Fakes client.files.{create,content} for the Batch API's upload/
    download roundtrip."""

    def __init__(self, *, file_id="file_fake", content_by_id=None, status="processed"):
        self.file_id = file_id
        self._content_by_id = content_by_id or {}
        self._status = status
        self.create_kwargs = None

    def create(self, **kwargs):
        self.create_kwargs = kwargs
        return SimpleNamespace(id=self.file_id)

    def retrieve(self, file_id):
        return SimpleNamespace(id=file_id, status=self._status)

    def content(self, file_id):
        return SimpleNamespace(text=self._content_by_id.get(file_id, ""))


class FakeOpenAIBatches:
    def __init__(self, *, batch_id="batch_fake", statuses=("completed",), output_file_id=None, error_file_id=None):
        self.batch_id = batch_id
        self._statuses = list(statuses)
        self._retrieve_calls = 0
        self.output_file_id = output_file_id
        self.error_file_id = error_file_id
        self.create_kwargs = None

    def create(self, **kwargs):
        self.create_kwargs = kwargs
        return SimpleNamespace(id=self.batch_id)

    def retrieve(self, batch_id):
        idx = min(self._retrieve_calls, len(self._statuses) - 1)
        self._retrieve_calls += 1
        return SimpleNamespace(
            status=self._statuses[idx], output_file_id=self.output_file_id, error_file_id=self.error_file_id
        )


class FakeOpenAIUploads:
    """Fakes client.uploads.upload_file_chunked -- records the call and
    returns an Upload-shaped object whose .file.id is the given file_id."""

    def __init__(self, *, file_id="file_fake"):
        self.file_id = file_id
        self.chunked_kwargs = None

    def upload_file_chunked(self, **kwargs):
        self.chunked_kwargs = kwargs
        return SimpleNamespace(id="upload_fake", file=SimpleNamespace(id=self.file_id))


class FakeOpenAIBatchClient:
    def __init__(self, batches: FakeOpenAIBatches, files: FakeOpenAIFiles, uploads: "FakeOpenAIUploads | None" = None):
        self.batches = batches
        self.files = files
        if uploads is not None:
            self.uploads = uploads


class FakeGeminiBatches:
    """Fakes client.batches.{create,get} -- the google-genai Batch API
    shape gemini_extractor.py's submit_gemini_batch/poll_gemini_batch/
    fetch_gemini_batch_results use. inlined_responses is returned
    positionally (see fetch_gemini_batch_results' docstring on why)."""

    def __init__(self, *, job_name="batches/fake", states=("JOB_STATE_SUCCEEDED",), inlined_responses=None):
        self.job_name = job_name
        self._states = list(states)
        self._get_calls = 0
        self._inlined_responses = inlined_responses if inlined_responses is not None else []
        self.create_kwargs = None

    def create(self, **kwargs):
        self.create_kwargs = kwargs
        return SimpleNamespace(name=self.job_name)

    def get(self, name):
        idx = min(self._get_calls, len(self._states) - 1)
        self._get_calls += 1
        return SimpleNamespace(
            state=self._states[idx], name=self.job_name,
            dest=SimpleNamespace(inlined_responses=self._inlined_responses),
        )


class FakeGeminiBatchClient:
    def __init__(self, batches: FakeGeminiBatches):
        self.batches = batches


class FakeSerpApiClient:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc

    def search(self, params: dict):
        # Mirrors the real serpapi.Client.request(), which mutates the
        # params dict it's given in place, injecting the real api_key --
        # exercises that the adapter never hands over the dict it also
        # stores as request_payload (see google_ai_mode.py's comment on the
        # dict(request_payload) copy).
        params["api_key"] = "leaked-if-adapter-passes-the-same-dict-object"
        if self._exc is not None:
            raise self._exc
        return self._response


# ---------------------------------------------------------------------------
# analysis/aux/http.py -- PoliteSession injects a requests.Session-shaped
# object; tests supply one of these so nothing hits a real socket.
# ---------------------------------------------------------------------------


class FakeRequestsResponse:
    def __init__(
        self,
        *,
        url: str,
        status_code: int = 200,
        headers: dict | None = None,
        body: bytes = b"",
        history: list | None = None,
    ):
        self.url = url
        self.status_code = status_code
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}
        self._body = body
        self.history = history or []

    def iter_content(self, chunk_size: int = 16384):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]

    def close(self):
        pass


class FakeRequestsSession:
    """Matches the slice of requests.Session that PoliteSession uses:
    .headers, .max_redirects, .request(), .close(). `routes` maps a URL (or
    a substring) to a FakeRequestsResponse or an Exception to raise."""

    def __init__(self, routes: dict | None = None, default: object | None = None):
        self.headers: dict = {}
        self.max_redirects = 30
        self._routes = routes or {}
        self._default = default
        self.calls: list[tuple[str, str]] = []

    def request(self, method, url, *, timeout=None, allow_redirects=True, stream=False, headers=None):
        self.calls.append((method, url))
        match = None
        if url in self._routes:
            match = self._routes[url]
        else:
            for key, value in self._routes.items():
                if key in url:
                    match = value
                    break
        if match is None:
            match = self._default
        if match is None:
            match = FakeRequestsResponse(url=url, status_code=404, body=b"not found")
        if isinstance(match, BaseException):
            raise match
        if callable(match):
            match = match(url)
        if getattr(match, "url", None) in (None, ""):
            match.url = url
        return match

    def close(self):
        pass
