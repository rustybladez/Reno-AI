"""Command-line check: uv run python -m leave_assistant "How many annual leave days do I get?"

Options:
    --today YYYY-MM-DD   pretend today is this date (default: today in Dhaka)
    --debug              also print the debug details (tool calls, citation checks, ...)
"""

import argparse
import json
import sys
from datetime import date

from .assistant import ask
from .policy import PolicyNotFoundError
from .providers.base import ProviderError


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles may not default to UTF-8

    parser = argparse.ArgumentParser(prog="python -m leave_assistant")
    parser.add_argument("question")
    parser.add_argument("--today", type=date.fromisoformat)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    try:
        response = ask(args.question, today=args.today)
    except (ProviderError, PolicyNotFoundError) as e:
        sys.exit(f"Error: {e}")
    print(response.text)
    if response.calculations:
        print("\nHow this was calculated:")
        for line in response.calculations:
            print(f"  - {line}")
    print(f"\n[type: {response.type} | {response.debug.get('latency_s')}s]")
    if args.debug:
        print(json.dumps(response.debug, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
