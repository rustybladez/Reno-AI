"""OpenAI-compatible provider (POST {base_url}/chat/completions), used for DeepSeek.

Works with any OpenAI-compatible endpoint. DeepSeek specifics (official docs, api-docs.deepseek.com):
- tools and response_format {"type": "json_object"} can be sent together; json_object does not
  enforce a schema, so the schema is described in the system prompt and guardrails validate it;
- in thinking mode, the assistant's reasoning_content must be sent back with tool calls, so we
  replay the model's own message unchanged;
- tool-call arguments are a JSON string and are not always valid JSON;
- 429 / 500 / 503 mean busy and 402 means no balance. Fail fast (no backoff): the caller can fall
  back to another provider straight away.
"""

import json
from dataclasses import dataclass

import httpx
from pydantic import BaseModel

from .base import ProviderBusyError, ProviderConfigError, ProviderError, ProviderResult, ToolCall

TIMEOUT_S = 60.0
BUSY_CODES = {402, 429, 500, 502, 503, 504}  # 402 = insufficient balance: let a fallback take over
BUSY_FINISH_REASONS = {"insufficient_system_resource"}


@dataclass
class OpenAICompatTurn:
    """The model's own assistant message with tool calls, replayed as-is on the next request."""

    model: str
    message: dict


class OpenAICompatProvider:
    name = "openai_compat"

    def __init__(
        self,
        base_url: str | None,
        api_key: str | None,
        model: str | None,
        thinking: str | None = None,
        reasoning_effort: str | None = None,
    ):
        if not (base_url and api_key and model):
            raise ProviderConfigError(
                "OPENAI_COMPAT_BASE_URL, OPENAI_COMPAT_API_KEY and OPENAI_COMPAT_MODEL must all be set."
            )
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.models = [model]
        self.thinking = thinking  # "enabled" | "disabled" | None (don't send; server default)
        self.reasoning_effort = reasoning_effort  # e.g. "low" | "high" | None
        self.thinking_level = f"thinking {thinking or 'default'}, effort {reasoning_effort or 'default'}"
        self.client = httpx.Client(timeout=TIMEOUT_S)

    def generate(
        self,
        system: str,
        messages: list[dict],
        schema: type[BaseModel],
        tools: list[dict] | None = None,
    ) -> ProviderResult:
        body: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system + _schema_instructions(schema)}]
            + _to_openai_messages(messages, self.model),
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        if tools:
            body["tools"] = [
                {"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                  "parameters": t["parameters"]}}
                for t in tools
            ]
        if self.thinking:
            body["thinking"] = {"type": self.thinking}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort

        data = self._post(body)
        choice = data["choices"][0]
        if choice.get("finish_reason") in BUSY_FINISH_REASONS:
            raise ProviderBusyError(f"{self.model}: {choice['finish_reason']}")
        message = choice["message"]

        if message.get("tool_calls"):
            tool_calls = [
                ToolCall(name=c["function"]["name"], args=_parse_arguments(c["function"].get("arguments")), id=c.get("id"))
                for c in message["tool_calls"]
            ]
            return ProviderResult(tool_calls=tool_calls, raw=OpenAICompatTurn(self.model, message), model=self.model)
        return ProviderResult(text=message.get("content") or None, model=self.model)

    def _post(self, body: dict) -> dict:
        try:
            response = self.client.post(self.url, json=body, headers={"Authorization": f"Bearer {self.api_key}"})
        except httpx.TimeoutException as e:
            raise ProviderBusyError(f"{self.model}: timed out") from e
        except httpx.TransportError as e:
            raise ProviderBusyError(f"{self.model}: connection failed ({e})") from e
        if response.status_code in BUSY_CODES:
            raise ProviderBusyError(f"{self.model}: HTTP {response.status_code}")
        if response.status_code != 200:
            raise ProviderError(f"{self.model}: HTTP {response.status_code}: {response.text[:300]}")
        # Non-streaming responses may start with keep-alive blank lines; json.loads skips whitespace.
        return json.loads(response.text)


def _schema_instructions(schema) -> str:
    """json_object mode doesn't take a schema, so describe it in the prompt."""
    if not hasattr(schema, "model_json_schema"):
        return ""
    return "\n\nThe JSON object must match this JSON Schema:\n" + json.dumps(schema.model_json_schema())


def _parse_arguments(arguments: str | None) -> dict:
    """Tool arguments arrive as a JSON string that is not always valid."""
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {"_invalid_arguments": arguments}  # dispatch() turns this into an error result
    return parsed if isinstance(parsed, dict) else {"_invalid_arguments": arguments}


def _to_openai_messages(messages: list[dict], model: str) -> list[dict]:
    """Provider-neutral messages -> OpenAI chat messages."""
    converted: list[dict] = []
    pending_ids: list[str] = []  # ids given to rebuilt tool calls, for results that have no id
    for m in messages:
        if m["role"] == "tool":
            call_id = m.get("id") or (pending_ids.pop(0) if pending_ids else "call_0")
            converted.append({"role": "tool", "tool_call_id": call_id,
                              "content": json.dumps(m["content"], ensure_ascii=False)})
        elif m.get("tool_calls"):
            raw = m.get("raw")
            if isinstance(raw, OpenAICompatTurn) and raw.model == model:
                # Same model: replay its own message (keeps reasoning_content, as DeepSeek requires).
                keep = ("role", "content", "reasoning_content", "tool_calls")
                converted.append({k: v for k, v in raw.message.items() if k in keep})
            else:
                # Calls made by another provider/model: rebuild them in OpenAI format.
                calls = []
                for i, c in enumerate(m["tool_calls"]):
                    call_id = c.id or f"call_{len(converted)}_{i}"
                    if not c.id:
                        pending_ids.append(call_id)
                    calls.append({"id": call_id, "type": "function",
                                  "function": {"name": c.name, "arguments": json.dumps(c.args)}})
                converted.append({"role": "assistant", "content": "", "tool_calls": calls})
        else:
            converted.append({"role": m["role"], "content": m["content"]})
    return converted
