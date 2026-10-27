"""TDD: gateway retries transient Vertex failures once before surfacing 502."""
import httpx
import pytest

from nanomuse_gateway.server import Gateway


async def test_retries_once_on_vertex_500():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, text="transient")
        return httpx.Response(200, json={
            "candidates": [{"content": {"role": "model",
                                        "parts": [{"text": "ok"}]}}]})

    gw = Gateway(project="p", sa_info={}, gateway_token="t",
                 http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                 token_provider=lambda: "tok")
    tc = httpx.ASGITransport(app=gw.app())
    async with httpx.AsyncClient(transport=tc, base_url="http://gw") as c:
        r = await c.post("/v1/chat/completions",
                         headers={"Authorization": "Bearer t"},
                         json={"messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200 and calls["n"] == 2


async def test_502_after_two_failures():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, text="down")

    gw = Gateway(project="p", sa_info={}, gateway_token="t",
                 http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                 token_provider=lambda: "tok")
    tc = httpx.ASGITransport(app=gw.app())
    async with httpx.AsyncClient(transport=tc, base_url="http://gw") as c:
        r = await c.post("/v1/chat/completions",
                         headers={"Authorization": "Bearer t"},
                         json={"messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 502 and calls["n"] == 2
