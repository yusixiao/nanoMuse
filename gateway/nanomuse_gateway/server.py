"""Gateway server: stateless FastAPI app.

Holds the GCP SA credential (never leaves this process), checks a shared
gateway token from the app, and exposes OpenAI-compatible
POST /v1/chat/completions backed by Vertex Gemini.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable

import httpx
import json as _json

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from nanomuse_gateway.translate import gemini_to_openai, openai_to_gemini


class Gateway:
    def __init__(
        self,
        project: str,
        sa_info: dict[str, Any],
        location: str = "global",
        model: str = "gemini-3.8-flash",
        gateway_token: str = "",
        http: httpx.AsyncClient | None = None,
        token_provider: Callable[[], str] | None = None,
    ):
        self.project = project
        self.location = location
        self.model = model
        self.gateway_token = gateway_token
        self.http = http or httpx.AsyncClient(timeout=120)
        self.last_auth: str = ""
        if token_provider is not None:
            self._token_provider = token_provider
        else:
            self._creds = self._make_credentials(sa_info)
            self._token_provider = self._cached_token

    # -- credential handling -------------------------------------------------
    def _make_credentials(self, sa_info: dict):
        import google.oauth2.service_account as sa

        return sa.Credentials.from_service_account_info(
            sa_info, scopes=["https://www.googleapis.com/auth/cloud-platform"])

    def _cached_token(self) -> str:
        import google.auth.transport.requests as tr

        if not getattr(self, "_tok", None) or self._tok_exp - 60 < time.time():
            self._creds.refresh(tr.Request())
            self._tok = self._creds.token
            self._tok_exp = self._creds.expiry.timestamp()
        return self._tok

    # -- ASGI app -------------------------------------------------------------
    def app(self) -> FastAPI:
        app = FastAPI(title="muse-gateway")

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            auth = request.headers.get("authorization", "")
            # gateway token OR a cloud-issued nm_ token (dev mode: any nm_)
            ok = auth == f"Bearer {self.gateway_token}" or \
                auth.startswith("Bearer nm_")
            if not ok:
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            body = await request.json()
            import sys
            print(f"[gateway] req stream={body.get('stream')} "
                  f"model={body.get('model')} msgs={len(body.get('messages', []))}",
                  file=sys.stderr)
            return await self._completion(body, stream=bool(body.get("stream")))

        @app.get("/v1/health")
        async def health():
            return {"ok": True, "model": self.model}

        @app.get("/healthz")
        async def healthz():
            # nanoMuse RelayAddress.check() shape
            return {"ok": True, "version": "0.5",
                    "models": [_menu_entry(self.model)]}

        # ---- nanoMuse-Cloud-compatible auth (sign-in always succeeds;
        # ---- model credentials live server-side, never on the phone) ----
        @app.post("/v1/auth/code")
        async def auth_code(request: Request):
            body = await request.json()
            identifier = (body.get("identifier") or "").strip()
            if not identifier:
                return JSONResponse({"error": "identifier required"},
                                    status_code=400)
            return {"ok": True, "channel": "email",
                    "hint": _mask(identifier)}

        @app.post("/v1/auth/verify")
        async def auth_verify(request: Request):
            body = await request.json()
            identifier = (body.get("identifier") or "").strip()
            # dev mode: any identifier + any code signs in
            return _sign_in_reply(identifier)

        @app.post("/v1/auth/login")
        async def auth_login(request: Request):
            body = await request.json()
            identifier = (body.get("identifier") or "").strip()
            return _sign_in_reply(identifier)

        @app.get("/v1/models")
        async def models_menu(request: Request):
            return {"data": [_menu_entry(self.model)]}

        @app.get("/v1/me")
        async def me(request: Request):
            # identity probe — always valid in dev mode
            return {"account": {"hint": "dev@muse.local", "channel": "email",
                                "member": False},
                    "tokens": {"granted": 0, "used": 0, "used_today": 0,
                               "daily_cap": 0, "unlimited": True},
                    "spend": {"today": 0.0, "total": 0.0, "grant": 0.0,
                              "left": 0.0},
                    "nudges": None}

        @app.get("/v1/sync/changes")
        async def sync_changes(request: Request):
            # no synced conversations yet
            return {"changes": [], "cursor": 0, "has_more": False}

        @app.get("/v1/hub")
        async def hub(request: Request):
            return {"ok": True}

        # The phone's HubClient speaks wss:// — accept the upgrade and hold the
        # socket open so its 1s→30s backoff loop stops. No frames are sent by
        # the standalone gateway; incoming pings keep it alive, closes are fine.
        @app.websocket("/v1/hub")
        async def hub_ws(ws: WebSocket):
            await ws.accept()
            try:
                while True:
                    await ws.receive_text()
            except WebSocketDisconnect:
                pass

        def _mask(identifier: str) -> str:
            if "@" in identifier:
                user, _, host = identifier.partition("@")
                return f"{user[:2]}***@{host}" if len(user) > 2 else f"{user}***@{host}"
            return identifier[:3] + "****" + identifier[-2:] if len(identifier) > 5 else "***"

        def _menu_entry(model: str) -> dict:
            return {
                "id": model,
                "name": model,
                "architecture": {
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                },
                "nanomuse": {"for": ["chat", "gui"], "catalog": False},
            }

        def _sign_in_reply(identifier: str) -> dict:
            return {
                "api_key": "nm_" + identifier.split("@")[0].replace(" ", "_") + "_token",
                "created": True,
                "base_url": None,
                "account": {"channel": "email", "hint": _mask(identifier),
                            "member": False},
                "tokens": {"granted": 0, "used": 0, "used_today": 0,
                           "daily_cap": 0, "unlimited": True},
                "spend": {"today": 0.0, "total": 0.0, "grant": 0.0, "left": 0.0},
                "models": [_menu_entry(self.model)],
                "nudges": None,
            }

        return app

    async def _completion(self, body: dict[str, Any], stream: bool = False):
        model = body.get("model") or self.model
        url = (f"https://aiplatform.googleapis.com/v1/projects/{self.project}"
               f"/locations/{self.location}/publishers/google/models/"
               f"{model}:generateContent")
        token = self._token_provider()
        self.last_auth = f"Bearer {token}"
        payload = openai_to_gemini(body)
        headers = {"Authorization": self.last_auth,
                   "Content-Type": "application/json"}
        # one retry for transient Vertex failures (5xx)
        for attempt in range(2):
            r = await self.http.post(url, json=payload, headers=headers)
            if r.status_code < 500:
                break
        if r.status_code != 200:
            print(f"[gateway] vertex {r.status_code}: {r.text[:500]}")
            return JSONResponse({"error": {"message": r.text[:400]}},
                                status_code=502)
        out = gemini_to_openai(r.json(), model=model)
        import sys
        print(f"[gateway] chat ok finish={out['choices'][0].get('finish_reason')} "
              f"content_len={len(out['choices'][0]['message'].get('content') or '')} "
              f"tool_calls={bool(out['choices'][0]['message'].get('tool_calls'))}",
              file=sys.stderr)
        if not stream:
            return JSONResponse(out)

        # OpenAI-style SSE: role chunk, content deltas, [DONE]
        msg = out["choices"][0]["message"]
        gen = out["id"]

        def chunk(delta: dict, finish: str | None = None) -> str:
            return "data: " + _json.dumps({
                "id": gen, "object": "chat.completion.chunk", "model": model,
                "choices": [{"index": 0, "delta": delta,
                             "finish_reason": finish}],
            }) + "\n\n"

        parts = []
        if msg.get("tool_calls"):
            parts.append(chunk({"role": "assistant"}))
            parts.append(chunk({"tool_calls": msg["tool_calls"]},
                               finish="tool_calls"))
        else:
            content = msg.get("content") or ""
            parts.append(chunk({"role": "assistant"}))
            parts.append(chunk({"content": content}, finish="stop"))
        parts.append("data: [DONE]\n\n")
        return StreamingResponse(iter(parts), media_type="text/event-stream")


def build_app() -> FastAPI:
    """Entry for uvicorn: nanomuse_gateway.server:build_app --factory"""
    import os

    sa_path = os.environ["GATEWAY_SA_FILE"]
    with open(sa_path) as f:
        sa_info = json.load(f)
    return Gateway(
        project=os.environ.get("GATEWAY_PROJECT", sa_info.get("project_id", "")),
        sa_info=sa_info,
        location=os.environ.get("GATEWAY_LOCATION", "global"),
        model=os.environ.get("GATEWAY_MODEL", "gemini-3.8-flash"),
        gateway_token=os.environ.get("GATEWAY_TOKEN", "dev-token"),
    ).app()
