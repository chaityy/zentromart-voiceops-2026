from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .knowledge import KnowledgeBase
from .tools import SupportTools


SYSTEM = """You are ZentroMart's concise, warm customer-support AI. Say you are an AI assistant at the start of a new conversation. Answer only from the supplied current knowledge sections and verified order tool results. If the sources do not answer, say you cannot confirm and offer a human colleague. Ask one clear question when a required detail is missing. Follow policy precedence and effective dates. Never expose order details until lookup_order returns verified=true. Never claim a ticket or action succeeded unless the tool confirms it. Do not repeat verification values. Preserve context for follow-ups. Keep spoken replies short. Return only an answer grounded in the context; source IDs are attached by the application."""

TOOLS = [
    {"type": "function", "function": {"name": "lookup_order", "description": "Look up a synthetic order. First call with only order_id to request verification. Then, after the caller gives exactly one phone suffix or pincode, call again with verification_value.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "verification_value": {"type": "string"}}, "required": ["order_id"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "create_ticket", "description": "Create a support handover after policy requires human review. For order-specific handovers, the order must be verified first.", "parameters": {"type": "object", "properties": {"summary": {"type": "string"}, "order_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["summary"], "additionalProperties": False}}},
]


@dataclass
class AgentResult:
    text: str
    sources: list[str]
    tool_calls: list[str]


class LLMError(RuntimeError):
    pass


class ChatCompletions:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is not set. Add your own LLM credential locally; no live answers were generated.")

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0.1}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        request = urllib.request.Request(
            self.base_url + "/chat/completions", data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(f"LLM request failed: {exc.__class__.__name__}") from exc
        return payload["choices"][0]["message"]


class SupportAgent:
    def __init__(self, kb: KnowledgeBase, data_dir: Path, client: Any | None = None):
        self.kb = kb
        self.tools = SupportTools(data_dir)
        self.client = client or ChatCompletions()
        self.history: list[dict[str, Any]] = []
        self.turn_count = 0
        self.failed_verifications: dict[str, int] = {}

    def respond(self, message: str, *, use_retrieval: bool = True) -> AgentResult:
        self.turn_count += 1
        prior_context = " ".join(m["content"] for m in self.history[-8:] if m.get("role") == "user")
        search_query = (prior_context + " " + message).strip()
        matches = self.kb.search(search_query, top_k=8) if use_retrieval else []
        sources = list(dict.fromkeys(chunk.source_id for chunk, _ in matches))
        context = "\n\n".join(f"[{c.source_id}] {c.heading}\n{c.text}" for c, _ in matches)
        msgs = [{"role": "system", "content": SYSTEM + ("\n\nRetrieved evidence:\n" + context if context else "\n\nNo documents retrieved; do not invent policy." )}]
        msgs.extend(self.history[-8:])
        user_record = {"role": "user", "content": message}
        msgs.append(user_record)
        tool_names: list[str] = []
        for _ in range(4):
            reply = self.client.complete(msgs, TOOLS)
            calls = reply.get("tool_calls") or []
            if not calls:
                text = (reply.get("content") or "I could not prepare a response. Please contact a human colleague.").strip()
                self.history.extend([{"role": "user", "content": user_record["content"]}, {"role": "assistant", "content": text}])
                self.history = self.history[-12:]
                return AgentResult(text, sources, tool_names)
            assistant_call = {**reply, "tool_calls": []}
            for call in calls:
                name = call["function"]["name"]
                args = json.loads(call["function"].get("arguments") or "{}")
                if name == "lookup_order":
                    order_id = "".join(ch for ch in str(args.get("order_id", "")) if ch.isdigit())
                    if args.get("verification_value") and self.failed_verifications.get(order_id, 0) >= 2:
                        result = {"found": True, "verified": False, "error": "Verification attempts exhausted; hand over to a human."}
                    else:
                        result = self.tools.lookup_order(order_id, args.get("verification_value"))
                        if args.get("verification_value") and not result.get("verified"):
                            self.failed_verifications[order_id] = self.failed_verifications.get(order_id, 0) + 1
                    if args.get("verification_value"):
                        safe_args = {k: v for k, v in args.items() if k != "verification_value"}
                        call["function"]["arguments"] = json.dumps(safe_args)
                        user_record["content"] = re.sub(r"(?<!\d)\d{4,6}(?!\d)", "[verification value]", user_record["content"])
                        args.pop("verification_value", None)
                    tool_names.append("lookup_order")
                elif name == "create_ticket":
                    result = self.tools.create_ticket(**args)
                    tool_names.append("create_ticket")
                else:
                    result = {"error": "Unsupported tool"}
                assistant_call["tool_calls"].append(call)
                msgs.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)})
            msgs.insert(len(msgs) - len(calls), assistant_call)
        return AgentResult("I’m unable to complete that safely right now. I can connect you with a human colleague.", sources, tool_names)
