"""Scripted provider for offline tests.

    FakeProvider({
        "question": [ProviderResult(tool_calls=[ToolCall("calculate", {...})]), '{"type": ...}'],
    })

Each call for a question returns the next scripted step (a str is final text). After the
script runs out, the last step repeats.
"""

from pydantic import BaseModel

from .base import ProviderResult


class FakeProvider:
    name = "fake"
    model = "fake"

    def __init__(self, script: dict[str, list[ProviderResult | str]]):
        self.script = script
        self.position: dict[str, int] = {}
        self.calls: list[dict] = []  # what each generate() call received

    def generate(
        self,
        system: str,
        messages: list[dict],
        schema: type[BaseModel],
        tools: list[dict] | None = None,
    ) -> ProviderResult:
        question = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        self.calls.append({"question": question, "messages": list(messages), "tools": tools})

        steps = self.script[question]
        index = self.position.get(question, 0)
        self.position[question] = index + 1
        step = steps[min(index, len(steps) - 1)]
        return ProviderResult(text=step) if isinstance(step, str) else step
