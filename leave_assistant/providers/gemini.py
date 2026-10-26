"""Google Gemini provider (google-genai SDK) with model cycling.

- One request per round: function calling and the JSON response schema are sent together
  (supported by the Gemini 3.x models; checked at M3).
- Free-tier quotas are per model (and small), so the provider holds a list of models. When
  one is busy (429/503/504/timeout) or retired (404), it switches to the next straight away
  and puts the failed one on a cooldown. It only waits if every model is cooling down.
- The tool loop itself lives in assistant.py; this class converts messages and picks models.
"""

import logging
import re
import time
from dataclasses import dataclass

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from .base import ProviderBusyError, ProviderConfigError, ProviderError, ProviderResult, ToolCall

TIMEOUT_MS = 60_000
BUSY_CODES = {429, 500, 502, 503, 504}  # rate limit / overloaded / temporary server errors
RETIRED_CODES = {404}  # "model no longer available"
DAILY_QUOTA_COOLDOWN_S = 3600.0  # per-day quota exhausted: don't retry that model for a while
DEFAULT_COOLDOWN_S = 30.0  # per-minute quota / overloaded, when the server gives no retryDelay
MAX_WAIT_S = 30.0  # longest we sleep when every model is cooling down
RETIRED = float("inf")

# Google's documented placeholder for function calls not produced by the model being called
# (we use it when a tool loop switches models part-way).
PLACEHOLDER_SIGNATURE = b"skip_thought_signature_validator"

# The SDK warns about function-call parts whenever it builds response.text; we read parts directly.
logging.getLogger("google_genai.types").setLevel(logging.ERROR)


@dataclass
class GeminiTurn:
    """What we keep of a model turn with function calls, to replay it on the next request."""

    model: str
    content: types.Content


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        api_key: str | None,
        models: list[str],
        thinking_level: str | None = None,
    ):
        if not api_key:
            raise ProviderConfigError("GEMINI_API_KEY is not set. See README → Quick start.")
        if not models:
            raise ProviderConfigError("No Gemini model configured (GEMINI_MODEL).")
        self.models = list(dict.fromkeys(models))  # primary first, then fallbacks; no duplicates
        self.thinking_level = thinking_level  # "minimal" | "low" | "medium" | "high" | None (model default)
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=TIMEOUT_MS))
        self.cooldown_until: dict[str, float] = {}  # model -> time.monotonic() when usable again
        self.model = self.models[0]  # last model used (shown in debug)

    def generate(
        self,
        system: str,
        messages: list[dict],
        schema: type[BaseModel],
        tools: list[dict] | None = None,
    ) -> ProviderResult:
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            response_mime_type="application/json",
            response_schema=schema,
            tools=[_to_gemini_tools(tools)] if tools else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=(
                types.ThinkingConfig(thinking_level=self.thinking_level.upper()) if self.thinking_level else None
            ),
        )
        model, response = self._call_with_cycling(messages, config)
        self.model = model

        content = response.candidates[0].content if response.candidates else None
        parts = (content.parts if content else None) or []
        calls = [p.function_call for p in parts if p.function_call]
        if calls:
            tool_calls = [ToolCall(name=c.name, args=dict(c.args or {}), id=c.id) for c in calls]
            return ProviderResult(tool_calls=tool_calls, raw=GeminiTurn(model, content), model=model)
        text = "".join(p.text for p in parts if p.text and not p.thought)
        return ProviderResult(text=text or None, model=model)

    # ---------- model cycling ----------

    def _call_with_cycling(self, messages: list[dict], config: types.GenerateContentConfig):
        errors_seen: list[str] = []
        for _ in range(2):  # one pass over the usable models, then one more after a short wait
            for model in self._usable_models():
                try:
                    response = self.client.models.generate_content(
                        model=model, contents=_to_contents(messages, model), config=config,
                    )
                    return model, response
                except errors.APIError as e:
                    if e.code in RETIRED_CODES:
                        self.cooldown_until[model] = RETIRED
                    elif e.code in BUSY_CODES:
                        self.cooldown_until[model] = time.monotonic() + _cooldown(e)
                    else:
                        raise ProviderError(f"Gemini error {e.code} on {model}: {e.message}") from e
                    errors_seen.append(f"{model}: {e.code}")
                except httpx.TimeoutException:
                    self.cooldown_until[model] = time.monotonic() + DEFAULT_COOLDOWN_S
                    errors_seen.append(f"{model}: timeout")

            wait = self._seconds_until_next_model()
            if wait is None or wait > MAX_WAIT_S:
                break
            time.sleep(wait)
        raise ProviderBusyError("All Gemini models are busy (" + "; ".join(errors_seen) + ")")

    def _usable_models(self) -> list[str]:
        now = time.monotonic()
        return [m for m in self.models if self.cooldown_until.get(m, 0) <= now]

    def _seconds_until_next_model(self) -> float | None:
        pending = [t for t in self.cooldown_until.values() if t != RETIRED]
        if not pending:
            return None
        return max(0.0, min(pending) - time.monotonic())


def _cooldown(error: errors.APIError) -> float:
    """How long to rest a busy model: an hour for a daily quota, else the server's retryDelay."""
    details = str(error.details)
    if "PerDay" in details:
        return DAILY_QUOTA_COOLDOWN_S
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s", details)
    return float(match.group(1)) if match else DEFAULT_COOLDOWN_S


# ---------- conversions ----------

def _to_gemini_tools(tools: list[dict]) -> types.Tool:
    return types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name=t["name"], description=t["description"], parameters_json_schema=t["parameters"],
        )
        for t in tools
    ])


def _to_contents(messages: list[dict], model: str) -> list[types.Content]:
    """Provider-neutral messages -> Gemini contents for `model`."""
    contents: list[types.Content] = []
    for m in messages:
        if m["role"] == "tool":
            part = types.Part.from_function_response(name=m["name"], response=m["content"])
            part.function_response.id = m.get("id")
            # All results for one round of calls go back in a single user turn.
            previous = contents[-1] if contents else None
            if previous and previous.role == "user" and previous.parts[0].function_response:
                previous.parts.append(part)
            else:
                contents.append(types.Content(role="user", parts=[part]))
        elif m.get("tool_calls"):
            raw = m.get("raw")
            if isinstance(raw, GeminiTurn) and raw.model == model:
                # Same model: replay its own turn so its thought signatures go back unchanged.
                contents.append(raw.content)
            else:
                # Different model (or none recorded): rebuild the calls with the placeholder signature.
                contents.append(types.Content(role="model", parts=[
                    types.Part(
                        function_call=types.FunctionCall(name=c.name, args=c.args, id=c.id),
                        thought_signature=PLACEHOLDER_SIGNATURE,
                    )
                    for c in m["tool_calls"]
                ]))
        else:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))
    return contents
