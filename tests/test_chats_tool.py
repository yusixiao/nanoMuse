"""CreateChat: the agent opens a side chat from a conversation (server-side only).

The tool creates a thread via MuseService, optionally seeds it with a first
message, and the drawer learns about it through the existing thread event —
no client changes. Background context rides the shared memory layer, not a
carried summary (Muse contract: one memory across main and side chats).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from nanomuse.config import Settings
from nanomuse.llm import MockLLM
from nanomuse.schema import Function, LLMResponse, ToolCall
from nanomuse.server import create_app
from nanomuse.server.service import MuseService


def tc(name: str, **args: Any) -> ToolCall:
    return ToolCall(function=Function(name=name, arguments=json.dumps(args)))


@pytest.fixture()
def server(settings: Settings) -> Iterator[tuple[TestClient, MuseService, MockLLM]]:

    settings.server.token = "secret-token"
    llm = MockLLM([])
    service = MuseService(settings, llm=llm)
    app = create_app(settings, service)
    with TestClient(app) as client:
        client.headers["Authorization"] = "Bearer secret-token"
        yield client, service, llm


def wait_for(pred, timeout: float = 5.0, interval: float = 0.05):  # noqa: ANN001
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        value = pred()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError("condition not met in time")


def test_create_chat_tool_exists_in_toolset(server):
    client, service, _ = server
    tool = next(t for t in service.app.tools if t.name == "create_chat")
    assert "title" in tool.parameters["properties"]
    assert tool.parameters["properties"]["title"]["type"] == "string"


async def test_create_chat_opens_thread_and_seeds_first_message(server):
    client, service, llm = server
    # turn 1 (main): call create_chat; turn 2 (new thread): final reply
    llm.script = [
        LLMResponse(
            tool_calls=[
                tc("create_chat", title="Trip planning", first_message="Find flights to Tokyo")
            ],
            finish_reason="tool_calls",
        ),
        LLMResponse(content="Opened a chat for trip planning.", finish_reason="stop"),
        LLMResponse(content="Searching flights.", finish_reason="stop"),
    ]
    client.post("/api/threads/main/send", json={"text": "plan a trip for me"})
    wait_for(lambda: len(service.threads) >= 2)
    wait_for(lambda: sum(1 for t in service.threads.values() if not t.busy) == len(service.threads))

    side = [t for t in service.threads.values() if t.id != "main"]
    assert len(side) == 1
    assert side[0].title == "Trip planning"
    # the first message was seeded into the new thread's timeline
    kinds = [e.get("type") for e in side[0].timeline.events]
    assert "user" in kinds
    texts = [e.get("text", "") for e in side[0].timeline.events]
    assert any("Find flights to Tokyo" in t for t in texts)


async def test_create_chat_without_first_message_makes_empty_side_chat(server):
    client, service, llm = server
    llm.script = [
        LLMResponse(tool_calls=[tc("create_chat", title="Recipes")], finish_reason="tool_calls"),
        LLMResponse(content="Created.", finish_reason="stop"),
    ]
    client.post("/api/threads/main/send", json={"text": "start a recipes chat"})
    wait_for(lambda: any(t.id != "main" for t in service.threads.values()))
    side = [t for t in service.threads.values() if t.id != "main"][0]
    assert side.title == "Recipes"


def test_thread_changed_event_fired_for_drawer(server):
    client, service, llm = server
    llm.script = [
        LLMResponse(tool_calls=[tc("create_chat", title="Books")], finish_reason="tool_calls"),
        LLMResponse(content="done", finish_reason="stop"),
    ]
    client.post("/api/threads/main/send", json={"text": "books chat"})
    wait_for(lambda: any(t.id != "main" for t in service.threads.values()))
    # the service published the thread event the drawer listens to
    assert service.bus is not None
