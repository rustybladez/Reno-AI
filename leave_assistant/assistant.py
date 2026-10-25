"""Single entry point: ask(). The UI, evals and tests all call it."""

import time
from datetime import date
from functools import lru_cache

from . import guardrails
from .config import load_settings, today_in_dhaka
from .policy import Policy, load_policy
from .prompt import build_messages, build_system_prompt, format_today
from .providers import get_provider
from .providers.base import Provider
from .schema import AssistantResponse, LLMOutput, Turn
from .tools import TOOL_SPECS, dispatch

MAX_TOOL_ROUNDS = 6


@lru_cache(maxsize=1)
def get_policy() -> Policy:
    return load_policy(load_settings())


def ask(
    question: str,
    history: list[Turn] | None = None,
    today: date | None = None,
    provider: Provider | None = None,
    policy: Policy | None = None,
) -> AssistantResponse:
    started = time.perf_counter()
    today = today or today_in_dhaka()
    policy = policy or get_policy()
    provider = provider or get_provider(load_settings())

    system = build_system_prompt(policy, today)
    messages = build_messages(question, history)
    tool_log: list[dict] = []
    models_used: list[str | None] = []  # one entry per model call

    raw = run_tool_loop(provider, system, messages, tool_log, models_used)
    output = guardrails.parse_output(raw)
    retried = False
    if output is None:
        # One retry for invalid JSON, with the tool results already in the messages.
        retried = True
        result = provider.generate(system, messages, LLMOutput, tools=None)
        models_used.append(result.model)
        raw = result.text
        output = guardrails.parse_output(raw)

    number_sources = (
        [question, today.isoformat(), format_today(today)]
        + [t.content for t in history or []]
        + guardrails.tool_results_as_text(tool_log)
    )
    response = guardrails.apply_guardrails(output, policy, number_sources)

    response.debug.update({
        "provider": f"{provider.name} / {provider.model}",
        "models_used": models_used,
        "raw_citations": [c.model_dump() for c in output.citations] if output else [],
        "tool_calls": tool_log,
        "retried_invalid_json": retried,
        "final_type": response.type,
        "latency_s": round(time.perf_counter() - started, 2),
    })
    return response


def run_tool_loop(
    provider: Provider,
    system: str,
    messages: list[dict],
    tool_log: list[dict],
    models_used: list[str | None],
) -> str | None:
    """Let the model call tools until it returns final text (max MAX_TOOL_ROUNDS rounds).

    Appends tool calls and results to `messages` and `tool_log`, and the model used for each
    call to `models_used`, in place.
    """
    for _ in range(MAX_TOOL_ROUNDS):
        result = provider.generate(system, messages, LLMOutput, tools=TOOL_SPECS)
        models_used.append(result.model)
        if not result.tool_calls:
            return result.text
        messages.append({"role": "assistant", "tool_calls": result.tool_calls, "raw": result.raw})
        for call in result.tool_calls:
            output = dispatch(call.name, call.args)
            tool_log.append({"name": call.name, "args": call.args, "result": output})
            messages.append({"role": "tool", "id": call.id, "name": call.name, "content": output})

    # Out of tool rounds: ask for the final answer without tools.
    result = provider.generate(system, messages, LLMOutput, tools=None)
    models_used.append(result.model)
    return result.text
