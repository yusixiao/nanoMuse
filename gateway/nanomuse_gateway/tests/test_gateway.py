"""TDD: LLM Gateway — stateless OpenAI-compat facade over Vertex Gemini.

The phone speaks OpenAI chat.completions (its only protocol). The gateway
translates to Vertex generateContent, holds the SA credential, and checks
a gateway token. Translation is a pure function; transport is httpx-mocked.
"""
import json

import httpx
import pytest

from nanomuse_gateway.translate import (
    openai_to_gemini,
    gemini_to_openai,
)
from nanomuse_gateway.server import Gateway


# ---------- protocol translation (pure) ----------

def test_messages_to_gemini_contents():
    body = {
        "model": "gemini-2.5-flash",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
            "tool-turns-omitted",
        ],
        "tools": [{"type": "function", "function": {"name": "f",
                                                    "parameters": {}}}],
    }
    # drop the synthetic marker; translator only sees real shapes
    body["messages"] = body["messages"][:3]
    req = openai_to_gemini(body)
    assert req["system_instruction"]["parts"][0]["text"] == "sys"
    assert req["contents"][0]["role"] == "user"
    assert req["contents"][0]["parts"][0]["text"] == "hi"
    assert req["contents"][1]["role"] == "model"
    assert req["tools"][0]["functionDeclarations"][0]["name"] == "f"


def test_gemini_reply_to_openai_choice():
    resp = {
        "candidates": [{"content": {"role": "model", "parts": [
            {"text": "hello there"}]}}],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
    }
    out = gemini_to_openai(resp, model="gemini-2.5-flash")
    assert out["object"] == "chat.completion"
    assert out["choices"][0]["message"] == {"role": "assistant",
                                            "content": "hello there"}
    assert out["usage"]["total_tokens"] == 8


def test_gemini_tool_call_to_openai():
    resp = {"candidates": [{"content": {"role": "model", "parts": [
        {"functionCall": {"name": "get_time", "args": {"tz": "utc"}}}]}}]}
    out = gemini_to_openai(resp, model="m")
    msg = out["choices"][0]["message"]
    assert msg["content"] is None
    tc = msg["tool_calls"][0]
    assert tc["function"]["name"] == "get_time"
    assert json.loads(tc["function"]["arguments"]) == {"tz": "utc"}


# ---------- end-to-end through the gateway (mocked vertex) ----------

@pytest.fixture
def gateway():
    calls = []

    def vertex_handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={
            "candidates": [{"content": {"role": "model", "parts": [
                {"text": "gateway says hi"}]}}],
            "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 4},
        })

    return Gateway(
        project="p1", sa_info={"type": "service_account"},  # token via injected client
        location="global", model="gemini-2.5-flash",
        gateway_token="tok-123",
        http=httpx.AsyncClient(transport=httpx.MockTransport(vertex_handler)),
        token_provider=lambda: "fake-oauth-token",
    ), calls


async def test_gateway_chat_completion(gateway):
    gw, calls = gateway
    app = gw.app()
    tc = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=tc,
                                 base_url="http://gw") as client:
        r = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer tok-123"},
            json={"model": "gemini-2.5-flash",
                  "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "gateway says hi"
    # vertex call carried the SA oauth token and the translated contents
    assert gw.last_auth == "Bearer fake-oauth-token"
    assert calls[0]["contents"][0]["parts"][0]["text"] == "hi"


async def test_gateway_rejects_bad_token(gateway):
    gw, _ = gateway
    tc = httpx.ASGITransport(app=gw.app())
    async with httpx.AsyncClient(transport=tc, base_url="http://gw") as c:
        r = await c.post("/v1/chat/completions",
                         headers={"Authorization": "Bearer wrong"},
                         json={"messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 401
