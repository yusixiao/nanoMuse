"""Pure protocol translation: OpenAI chat.completions <-> Vertex Gemini."""
from __future__ import annotations

import json
from typing import Any


def openai_to_gemini(body: dict[str, Any]) -> dict[str, Any]:
    """OpenAI messages/tools -> Vertex generateContent request.

    - system message becomes system_instruction
    - user/assistant map to user/model roles
    - OpenAI role=tool (tool result) maps to functionResponse
    - assistant tool_calls map to functionCall parts
    """
    contents: list[dict] = []
    system = ""
    for m in body.get("messages", []):
        role = m.get("role")
        if role == "system":
            system = m.get("content", "")
        elif role == "user":
            contents.append({"role": "user",
                             "parts": [{"text": m.get("content", "")}]})
        elif role == "assistant":
            parts: list[dict] = []
            if m.get("content"):
                parts.append({"text": m["content"]})
            for tc in m.get("tool_calls") or []:
                fc = {"name": tc["function"]["name"],
                      "args": json.loads(tc["function"]["arguments"] or "{}")}
                sig = tc.get("thought_signature")
                if sig:
                    part = {"functionCall": fc, "thoughtSignature": sig}
                    parts.append(part)
                else:
                    # Unsigned tool turn (pre-signature history): Vertex
                    # rejects unsigned functionCalls outright. Degrade to a
                    # plain text turn that keeps the intent.
                    call_desc = (
                        f"{tc['function']['name']}({tc['function']['arguments']})"
                    )
                    parts.append({"text": f"[calling tool: {call_desc}]"})
            contents.append({"role": "model", "parts": parts or [{"text": ""}]})
        elif role == "tool":
            # Mirror the degraded assistant turn: when the unsigned history
            # made the tool call a text note, the result is a text turn too.
            prev = contents[-1]["parts"] if contents else []
            degraded = any("functionCall" not in p and "text" in p
                           and str(p.get("text", "")).startswith("[calling tool:")
                           for p in prev)
            if degraded:
                tool_result_text = (
                    f"[tool {m.get('name', 'tool')} result]: {m.get('content', '')}"
                )
                contents.append({"role": "user", "parts": [
                    {"text": tool_result_text},
                ]})
            else:
                contents.append({"role": "user", "parts": [{"functionResponse": {
                    "name": m.get("name", "tool"),
                    "response": {"result": m.get("content", "")},
                }}]})
    out: dict[str, Any] = {"contents": contents}
    if system:
        out["system_instruction"] = {"parts": [{"text": system}]}
    if body.get("tools"):
        decls = []
        for t in body["tools"]:
            fn = dict(t["function"])
            params = fn.get("parameters") or {}
            if params.get("type") != "object":
                # Gemini requires an OBJECT schema for every declaration
                params = {"type": "object", "properties": {}}
            fn["parameters"] = params
            decls.append(fn)
        out["tools"] = [{"functionDeclarations": decls}]
    return out


def gemini_to_openai(resp: dict[str, Any], model: str) -> dict[str, Any]:
    """Vertex generateContent response -> OpenAI chat.completion shape."""
    cand = (resp.get("candidates") or [{}])[0]
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if "text" in p)
    tool_calls = []
    for i, p in enumerate(parts):
        if "functionCall" not in p:
            continue
        tc = {"id": f"call_{i}", "type": "function", "function": {
            "name": p["functionCall"]["name"],
            "arguments": json.dumps(p["functionCall"].get("args", {})),
        }}
        if p.get("thoughtSignature"):
            tc["thought_signature"] = p["thoughtSignature"]
        tool_calls.append(tc)
    message: dict[str, Any] = {"role": "assistant",
                               "content": text if text else None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    usage = resp.get("usageMetadata", {})
    return {
        "id": "chatcmpl-gw",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {
            "prompt_tokens": usage.get("promptTokenCount", 0),
            "completion_tokens": usage.get("candidatesTokenCount", 0),
            "total_tokens": usage.get("promptTokenCount", 0)
                            + usage.get("candidatesTokenCount", 0),
        },
    }
