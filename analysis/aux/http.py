"""PoliteSession -- the one HTTP client for every auxiliary collector.

Design goals, in order:
  1. Never raise. Every failure (DNS, connect, TLS, timeout, too many
     redirects, decode) becomes a populated `error` on an otherwise-complete
     FetchResult -- same contract as collector/engines/*.py adapters and
     analysis/coding/homepage_fetch.py.
  2. Store what cannot be reconstructed later: the full redirect chain, the
     final URL, every response header, the TLS certificate issuer/expiry.
  3. Be polite: an identifying User-Agent with a contact URL, and a minimum
     interval between requests to the same host (different hosts proceed in
     parallel, which is what the Task 5 crawl relies on).

Uses `requests` (added as a direct dep for this batch): its `Response.history`
gives the redirect chain for free, and a pooled `Session` reuses connections
across the ~10k Task 5 URLs. Sitemap XML is parsed with stdlib elsewhere; TLS
inspection is stdlib `ssl` here.

Analysis-layer code (INSTRUMENT.md section 12).
"""
from __future__ import annotations

import socket
import ssl
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

import requests

# Identifies the study and gives a contact point, per the batch's politeness
# rule. Update AUX_CONTACT_URL if the study's public page moves.
AUX_CONTACT_URL = "https://github.com/shliakhetko/geo-collector"
USER_AGENT = f"geo-collector-study/1.0 (post-window auxiliary collection; +{AUX_CONTACT_URL})"

DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MIN_HOST_INTERVAL_SECONDS = 1.0
DEFAULT_MAX_BODY_BYTES = 200 * 1024
DEFAULT_MAX_REDIRECTS = 20


@dataclass
class FetchResult:
    url: str
    method: str = "GET"
    ok: bool = False  # a real HTTP response was received (any status)
    error: str | None = None
    status_code: int | None = None
    final_url: str | None = None
    redirect_chain: list[dict[str, Any]] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)
    content_type: str | None = None
    body_bytes: bytes | None = None
    body_truncated: bool = False
    elapsed_ms: int = 0
    tls: dict[str, Any] | None = None

    @property
    def final_registrable_host(self) -> str | None:
        target = self.final_url or self.url
        try:
            return urlsplit(target).hostname
        except Exception:
            return None

    def meta(self) -> dict[str, Any]:
        """The result minus the raw body -- what goes into a JSONL row.
        The body itself is stored as a separate blob by the caller."""
        return {
            "url": self.url,
            "method": self.method,
            "ok": self.ok,
            "error": self.error,
            "status_code": self.status_code,
            "final_url": self.final_url,
            "redirect_chain": self.redirect_chain,
            "headers": self.headers,
            "content_type": self.content_type,
            "body_bytes_len": None if self.body_bytes is None else len(self.body_bytes),
            "body_truncated": self.body_truncated,
            "elapsed_ms": self.elapsed_ms,
            "tls": self.tls,
        }


def _host_of(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except Exception:
        return ""


def probe_tls(hostname: str, port: int = 443, *, timeout: float = 10.0) -> dict[str, Any] | None:
    """Certificate issuer/subject/validity for `hostname`. Never raises --
    returns None on any failure (plain HTTP host, handshake error, timeout)."""
    if not hostname:
        return None
    try:
        context = ssl.create_default_context()
        with socket.create_connection((hostname, port), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as tls_sock:
                cert = tls_sock.getpeercert()
        if not cert:
            return None

        def _flatten(seq: Any) -> dict[str, str]:
            out: dict[str, str] = {}
            for rdn in seq or ():
                for key, value in rdn:
                    out[key] = value
            return out

        return {
            "issuer": _flatten(cert.get("issuer")),
            "subject": _flatten(cert.get("subject")),
            "not_before": cert.get("notBefore"),
            "not_after": cert.get("notAfter"),
            "serial_number": cert.get("serialNumber"),
            "subject_alt_names": [v for (k, v) in cert.get("subjectAltName", ()) if k == "DNS"],
        }
    except Exception:
        return None


class PoliteSession:
    """Thread-safe. One instance is shared across the Task 5 thread pool; the
    per-host throttle is what keeps that pool polite."""

    def __init__(
        self,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        min_host_interval_seconds: float = DEFAULT_MIN_HOST_INTERVAL_SECONDS,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        user_agent: str = USER_AGENT,
        session: requests.Session | None = None,
        capture_tls: bool = True,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.min_host_interval_seconds = min_host_interval_seconds
        self.max_body_bytes = max_body_bytes
        self.max_redirects = max_redirects
        self.capture_tls = capture_tls
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": user_agent, "Accept": "*/*"})
        self._session.max_redirects = max_redirects
        self._host_lock = threading.Lock()
        self._host_next_allowed: dict[str, float] = {}

    # -- politeness ------------------------------------------------------

    def _throttle(self, url: str) -> None:
        host = _host_of(url)
        if not host or self.min_host_interval_seconds <= 0:
            return
        while True:
            with self._host_lock:
                now = time.monotonic()
                next_allowed = self._host_next_allowed.get(host, 0.0)
                if now >= next_allowed:
                    self._host_next_allowed[host] = now + self.min_host_interval_seconds
                    return
                wait = next_allowed - now
            time.sleep(wait)

    # -- fetching -------------------------------------------------------

    def _redirect_chain(self, response: requests.Response) -> list[dict[str, Any]]:
        chain: list[dict[str, Any]] = []
        for hop in response.history:
            chain.append(
                {
                    "url": hop.url,
                    "status_code": hop.status_code,
                    "location": hop.headers.get("Location"),
                }
            )
        return chain

    def _request(
        self,
        method: str,
        url: str,
        *,
        read_body: bool,
        allow_redirects: bool = True,
        extra_headers: dict[str, str] | None = None,
    ) -> FetchResult:
        result = FetchResult(url=url, method=method)
        self._throttle(url)
        started = time.monotonic()
        try:
            response = self._session.request(
                method,
                url,
                timeout=self.timeout_seconds,
                allow_redirects=allow_redirects,
                stream=read_body,
                headers=extra_headers or None,
            )
        except requests.exceptions.Timeout as exc:
            result.error = f"timeout: {exc}"
            result.elapsed_ms = int((time.monotonic() - started) * 1000)
            return result
        except requests.exceptions.TooManyRedirects as exc:
            result.error = f"too_many_redirects: {exc}"
            result.elapsed_ms = int((time.monotonic() - started) * 1000)
            return result
        except requests.exceptions.RequestException as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            result.elapsed_ms = int((time.monotonic() - started) * 1000)
            return result
        except Exception as exc:  # contract: never raise past this point
            result.error = f"{type(exc).__name__}: {exc}"
            result.elapsed_ms = int((time.monotonic() - started) * 1000)
            return result

        try:
            result.ok = True
            result.status_code = response.status_code
            result.final_url = response.url
            result.redirect_chain = self._redirect_chain(response)
            result.headers = {k: v for k, v in response.headers.items()}
            result.content_type = response.headers.get("Content-Type")
            if read_body:
                body = b""
                try:
                    for chunk in response.iter_content(chunk_size=16 * 1024):
                        body += chunk
                        if len(body) >= self.max_body_bytes:
                            result.body_truncated = True
                            body = body[: self.max_body_bytes]
                            break
                finally:
                    response.close()
                result.body_bytes = body
            else:
                response.close()
            if self.capture_tls and (result.final_url or url).lower().startswith("https"):
                result.tls = probe_tls(_host_of(result.final_url or url))
        except Exception as exc:  # never raise past here either
            result.error = f"post_response_error: {type(exc).__name__}: {exc}"
        result.elapsed_ms = int((time.monotonic() - started) * 1000)
        return result

    def head(self, url: str, *, allow_redirects: bool = True) -> FetchResult:
        return self._request("HEAD", url, read_body=False, allow_redirects=allow_redirects)

    def get(
        self,
        url: str,
        *,
        read_body: bool = True,
        allow_redirects: bool = True,
        extra_headers: dict[str, str] | None = None,
    ) -> FetchResult:
        return self._request(
            "GET", url, read_body=read_body, allow_redirects=allow_redirects, extra_headers=extra_headers
        )

    def head_then_get(self, url: str) -> tuple[FetchResult, FetchResult | None]:
        """HEAD first (cheap, no body); then GET for the body and to confirm.
        Returns (head_result, get_result) -- get_result is None only if HEAD
        itself never produced a response AND the GET was skipped (it isn't:
        the GET always runs, because many servers reject HEAD)."""
        head = self.head(url)
        get = self.get(url)
        return head, get

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
