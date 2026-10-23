"""CreateChat: the agent opens a side chat from within a conversation.

Muse contract: one persona, one memory across the main chat and all side
chats; a side chat is a topic branch whose background rides the shared
memory layer — so this tool carries no summary of the current conversation.
It creates the thread (the drawer learns through the existing thread event)
and optionally seeds a first message when the agent is delegating a task.
"""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict

from nanomuse.schema import RiskLevel, ToolResult
from nanomuse.tools.base import BaseTool, CallAssessment


class CreateChat(BaseTool):
    """Open a new side chat, optionally delegating a first message to it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "create_chat"
    description: str = (
        "Open a new side chat (a separate conversation in the same drawer), "
        "optionally starting it with a first message. Use it to keep a new "
        "topic out of the main chat, or to hand a self-contained task to a "
        "fresh conversation while you stay with the user here. Give the chat "
        "a short title; put any needed background into the first message in "
        "plain words; the shared memory already knows the long-term facts, "
        "so carry only what this specific topic needs. Tell the user the "
        "chat is ready and where to find it."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short title for the drawer, in the user's language.",
            },
            "first_message": {
                "type": "string",
                "description": (
                    "Optional first message to start the new chat with: a task to run "
                    "there, or background for the topic. Plain words, self-contained."
                ),
            },
        },
        "required": ["title"],
    }
    risk: RiskLevel = RiskLevel.SAFE

    # late binding: the service is created after the tools
    service: Any  # MuseService — create_thread() + _ensure_worker()

    def assess(self, args: dict[str, Any]) -> CallAssessment:
        return CallAssessment(
            risk=RiskLevel.SAFE,
            summary=f"create chat: {str(args.get('title') or '')[:80]}",
        )

    async def execute(self, title: str = "", first_message: str = "", **_: Any) -> ToolResult:
        title = (title or "").strip()
        if not title:
            return ToolResult.fail("title is required")
        thread = self.service.create_thread(title)
        if first_message.strip():
            # the same path as a user send: user event, worker, inbox — one code path
            self.service.send(thread.id, first_message.strip())
            return ToolResult(
                output=f"Side chat '{title}' is open (chat_id={thread.id}) and is already "
                f"working on: {first_message.strip()[:120]}. Tell the user it is in "
                "the drawer and will report back here or in that chat."
            )
        return ToolResult(
            output=f"Side chat '{title}' is open (chat_id={thread.id}). "
            "Tell the user it is ready in the drawer."
        )
