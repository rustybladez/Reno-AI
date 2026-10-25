"""Builds the system prompt and the message list sent to the model."""

from datetime import date

from .config import PROJECT_ROOT
from .policy import Policy
from .schema import Turn

SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_prompt.md"
HISTORY_TURNS = 6


def format_today(d: date) -> str:
    """e.g. "Monday 26 October 2026"."""
    return f"{d:%A} {d.day} {d:%B %Y}"


def build_system_prompt(policy: Policy, today: date) -> str:
    template = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    # str.replace, not str.format: the policy text may contain braces.
    return template.replace("{today}", format_today(today)).replace("{policy}", policy.render_for_prompt())


def build_messages(question: str, history: list[Turn] | None = None) -> list[dict]:
    """Last HISTORY_TURNS turns of history, then the new question.

    History tells the model what is being asked ("what about sick leave?"); it is never
    a source of policy facts.
    """
    recent = (history or [])[-HISTORY_TURNS:]
    messages = [{"role": t.role, "content": t.content} for t in recent]
    messages.append({"role": "user", "content": question})
    return messages
