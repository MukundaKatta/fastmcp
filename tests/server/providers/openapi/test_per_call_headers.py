"""Per-call HTTP headers on OpenAPI tools (issue #4025).

The shared-FastMCP-instance, multi-tenant case: one server, many callers, each
with their own bearer token. Headers can be injected via:

  - The Python-side ``http_headers=`` kwarg on ``mcp.call_tool(...)``.
  - The wire-side ``_meta.fastmcp.http_headers`` field on a tool call request.

Both paths converge on a ContextVar (`_current_call_http_headers`) that the
OpenAPI tool reads at request-build time and applies *over* the client's
default headers.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from httpx import Response

from fastmcp import FastMCP
from fastmcp.client import Client
from fastmcp.server.dependencies import (
    _current_call_http_headers,
    get_call_http_headers,
)
from fastmcp.server.providers.openapi import OpenAPIProvider


@pytest.fixture
def simple_spec() -> dict:
    """Single-endpoint spec — keeps tests focused on header behavior."""
    return {
        "openapi": "3.0.0",
        "info": {"title": "Test", "version": "1.0.0"},
        "servers": [{"url": "https://api.example.com"}],
        "paths": {
            "/search": {
                "get": {
                    "operationId": "search",
                    "parameters": [
                        {
                            "name": "q",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "ok",
                            "content": {
                                "application/json": {
                                    "schema": {"type": "object"}
                                }
                            },
                        }
                    },
                }
            }
        },
    }


def _build_mock_client(
    *,
    default_headers: dict[str, str] | None = None,
    json_body: dict | None = None,
) -> Mock:
    """Mock httpx.AsyncClient that captures the request that would be sent."""
    client = Mock(spec=httpx.AsyncClient)
    client.base_url = "https://api.example.com"
    client.headers = httpx.Headers(default_headers or {})

    response = Mock(spec=Response)
    response.status_code = 200
    response.json.return_value = json_body or {"results": []}
    response.text = json.dumps(json_body or {"results": []})
    response.raise_for_status = Mock()

    client.send = AsyncMock(return_value=response)
    return client


def _build_server(spec: dict, client: Mock) -> FastMCP:
    provider = OpenAPIProvider(openapi_spec=spec, client=client)
    mcp = FastMCP("test")
    mcp.add_provider(provider)
    return mcp


def _sent_headers(client: Mock) -> dict[str, str]:
    """Pull the headers off the captured request, lower-cased for comparison."""
    request = client.send.call_args[0][0]
    return {k.lower(): v for k, v in request.headers.items()}


class TestDefaultBehaviorPreserved:
    """No per-call headers => behavior is unchanged."""

    async def test_default_client_auth_used_when_no_per_call_headers(
        self, simple_spec
    ):
        client = _build_mock_client(
            default_headers={"Authorization": "Bearer default-token"}
        )
        mcp = _build_server(simple_spec, client)

        result = await mcp.call_tool("search", {"q": "alice"})

        assert result is not None
        sent = _sent_headers(client)
        assert sent["authorization"] == "Bearer default-token"

    async def test_no_auth_at_all_still_works(self, simple_spec):
        client = _build_mock_client()  # no default auth
        mcp = _build_server(simple_spec, client)

        result = await mcp.call_tool("search", {"q": "alice"})

        assert result is not None
        sent = _sent_headers(client)
        assert "authorization" not in sent


class TestPerCallHeadersKwarg:
    """The Python-side ``http_headers=`` kwarg path."""

    async def test_per_call_headers_override_client_defaults(self, simple_spec):
        client = _build_mock_client(
            default_headers={"Authorization": "Bearer default-token"}
        )
        mcp = _build_server(simple_spec, client)

        await mcp.call_tool(
            "search",
            {"q": "alice"},
            http_headers={"Authorization": "Bearer caller-token"},
        )

        sent = _sent_headers(client)
        assert sent["authorization"] == "Bearer caller-token"

    async def test_per_call_headers_added_when_no_default(self, simple_spec):
        client = _build_mock_client()
        mcp = _build_server(simple_spec, client)

        await mcp.call_tool(
            "search",
            {"q": "alice"},
            http_headers={"X-Tenant-Id": "tenant-42"},
        )

        sent = _sent_headers(client)
        assert sent["x-tenant-id"] == "tenant-42"

    async def test_multiple_per_call_headers(self, simple_spec):
        client = _build_mock_client()
        mcp = _build_server(simple_spec, client)

        await mcp.call_tool(
            "search",
            {"q": "alice"},
            http_headers={
                "Authorization": "Bearer per-call",
                "X-Request-Id": "req-123",
            },
        )

        sent = _sent_headers(client)
        assert sent["authorization"] == "Bearer per-call"
        assert sent["x-request-id"] == "req-123"

    async def test_per_call_headers_do_not_leak_into_tool_result(
        self, simple_spec
    ):
        """Per-call auth must not be echoed in any field of the tool result."""
        client = _build_mock_client(json_body={"results": ["alice"]})
        mcp = _build_server(simple_spec, client)

        result = await mcp.call_tool(
            "search",
            {"q": "alice"},
            http_headers={"Authorization": "Bearer secret-token"},
        )

        # Search every place the LLM could conceivably read.
        haystacks: list[str] = []
        if result.structured_content is not None:
            haystacks.append(json.dumps(result.structured_content))
        if result.content:
            for block in result.content:
                text = getattr(block, "text", None)
                if text:
                    haystacks.append(text)
        assert haystacks, "expected some content to inspect"
        for blob in haystacks:
            assert "secret-token" not in blob
            assert "Authorization" not in blob

    async def test_per_call_headers_do_not_leak_across_calls(self, simple_spec):
        """ContextVar must reset between calls so tenants stay isolated."""
        client = _build_mock_client()
        mcp = _build_server(simple_spec, client)

        # Tenant A
        await mcp.call_tool(
            "search",
            {"q": "alice"},
            http_headers={"Authorization": "Bearer tenant-a"},
        )
        sent_a = _sent_headers(client)

        # Tenant B — no headers passed; must NOT inherit tenant-a's token.
        client.send.reset_mock()
        await mcp.call_tool("search", {"q": "bob"})
        sent_b = _sent_headers(client)

        assert sent_a["authorization"] == "Bearer tenant-a"
        assert "authorization" not in sent_b

    async def test_get_call_http_headers_inside_call(self, simple_spec):
        """The public helper sees the per-call headers from inside the call."""

        seen: dict[str, dict[str, str]] = {}

        async def fake_send(request):
            # Read while the ContextVar should still be set.
            seen["headers"] = get_call_http_headers()
            response = Mock(spec=Response)
            response.status_code = 200
            response.json.return_value = {}
            response.text = "{}"
            response.raise_for_status = Mock()
            return response

        client = Mock(spec=httpx.AsyncClient)
        client.base_url = "https://api.example.com"
        client.headers = httpx.Headers()
        client.send = AsyncMock(side_effect=fake_send)

        mcp = _build_server(simple_spec, client)

        await mcp.call_tool(
            "search",
            {"q": "alice"},
            http_headers={"Authorization": "Bearer ctx-token"},
        )

        assert seen["headers"]["Authorization"] == "Bearer ctx-token"

        # And after the call returns, the ContextVar is empty again.
        assert get_call_http_headers() == {}


class TestPerCallHeadersViaMetaProtocol:
    """The wire-side ``_meta.fastmcp.http_headers`` path (in-memory client)."""

    async def test_meta_field_propagates_to_upstream_request(self, simple_spec):
        client = _build_mock_client(
            default_headers={"Authorization": "Bearer default-token"}
        )
        mcp = _build_server(simple_spec, client)

        async with Client(mcp) as mcp_client:
            await mcp_client.call_tool(
                "search",
                {"q": "alice"},
                meta={
                    "fastmcp": {
                        "http_headers": {"Authorization": "Bearer wire-token"}
                    }
                },
            )

        sent = _sent_headers(client)
        assert sent["authorization"] == "Bearer wire-token"

    async def test_meta_field_absent_falls_back_to_client_default(
        self, simple_spec
    ):
        client = _build_mock_client(
            default_headers={"Authorization": "Bearer default-token"}
        )
        mcp = _build_server(simple_spec, client)

        async with Client(mcp) as mcp_client:
            await mcp_client.call_tool("search", {"q": "alice"})

        sent = _sent_headers(client)
        assert sent["authorization"] == "Bearer default-token"

    async def test_malformed_meta_headers_silently_ignored(self, simple_spec):
        """A bad payload shouldn't break the call — it should fall through."""
        client = _build_mock_client(
            default_headers={"Authorization": "Bearer default-token"}
        )
        mcp = _build_server(simple_spec, client)

        async with Client(mcp) as mcp_client:
            await mcp_client.call_tool(
                "search",
                {"q": "alice"},
                # Not a flat str->str mapping.
                meta={"fastmcp": {"http_headers": "not-a-dict"}},
            )

        sent = _sent_headers(client)
        assert sent["authorization"] == "Bearer default-token"


class TestContextVarHygiene:
    """Pure ContextVar checks, independent of OpenAPI."""

    async def test_outside_call_helper_returns_empty_dict(self):
        """No call in flight => empty mapping, never None."""
        assert get_call_http_headers() == {}

    async def test_helper_returns_a_copy(self):
        """Mutating the result must not affect future readers."""
        token = _current_call_http_headers.set({"Authorization": "Bearer x"})
        try:
            view = get_call_http_headers()
            view["Authorization"] = "Bearer hacked"
            assert get_call_http_headers()["Authorization"] == "Bearer x"
        finally:
            _current_call_http_headers.reset(token)
