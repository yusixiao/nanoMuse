# Self-hosted nanoMuse gateway: OpenAI-compatible relay to Vertex Gemini.
# Build from the repository root so uv sees the top-level project files:
#   docker build -f gateway/Dockerfile .
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app
# websockets: the phone's hub speaks wss:// — without it uvicorn answers the
# upgrade with a plain 200 and HubClient retries forever. uv-managed venvs
# ship without pip, so extra deps go through uv against that venv.
COPY pyproject.toml uv.lock ./
COPY gateway/nanomuse_gateway ./nanomuse_gateway
RUN uv venv .venv && \
    uv pip install --python .venv/bin/python \
        "fastapi>=0.115" "uvicorn[standard]>=0.54.0" httpx requests \
        "google-auth>=2" pydantic python-dotenv

FROM python:3.12-slim-bookworm
WORKDIR /app
RUN groupadd -r muse && useradd -r -g muse muse
COPY --from=builder /app/.venv /app/.venv
COPY gateway/nanomuse_gateway /app/nanomuse_gateway
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER muse
EXPOSE 8000
CMD ["uvicorn", "nanomuse_gateway.server:build_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "75"]
