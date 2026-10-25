import json

import pytest

from leave_assistant.config import PROJECT_ROOT
from leave_assistant.policy import Policy, parse_policy

EXAMPLE_POLICY_PATH = PROJECT_ROOT / "data" / "policy.example.json"


@pytest.fixture
def example_policy() -> Policy:
    """The dummy 3-section policy. Tests never depend on the real (gitignored) policy."""
    return parse_policy(json.loads(EXAMPLE_POLICY_PATH.read_text(encoding="utf-8")))
