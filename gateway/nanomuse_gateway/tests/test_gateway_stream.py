"""TDD: SSE streaming for chat.completions (nanoMuse always streams)."""
import json

from nanomuse_gateway.server import Gateway


def _gw():
    import httpx
    def handler(request):
        return httpx.Response(200, json={
            "candidates": [{"content": {"role": "model",
                                        "parts": [{"text": "hello"}]}}],
            "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1},
        })
    gw = Gateway(project="p", sa_info={}, gateway_token="t", model="m",
                 http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                 token_provider=lambda: "tok")
    return gw.app()


async def test_stream_returns_sse_chunks():
    import httpx
    tc = httpx.ASGITransport(app=_gw())
    async with httpx.AsyncClient(transport=tc, base_url="http://g") as c:
        r = await c.post("/v1/chat/completions",
                         headers={"Authorization": "Bearer nm_x"},
                         json={"stream": True,
                               "messages": [{"role": "user", "content": "hi"}]})
    assert r.headers["content-type"].startswith("text/event-stream")
    body = r.text
    assert body.endswith("data: [DONE]\n\n")
    # first chunk carries the role; a later chunk carries the content
    chunks = [json.loads(line[6:]) for line in body.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    joined = "".join(c["choices"][0]["delta"].get("content") or ""
                     for c in chunks)
    assert joined == "hello"


async def test_non_stream_still_json():
    import httpx
    tc = httpx.ASGITransport(app=_gw())
    async with httpx.AsyncClient(transport=tc, base_url="http://g") as c:
        r = await c.post("/v1/chat/completions",
                         headers={"Authorization": "Bearer nm_x"},
                         json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.headers["content-type"].startswith("application/json")
