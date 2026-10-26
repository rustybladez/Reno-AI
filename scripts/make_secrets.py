"""Print a Streamlit secrets block with the policy as POLICY_JSON, ready to paste into
Streamlit Community Cloud → App settings → Secrets. API keys are left as placeholders.

    uv run python scripts/make_secrets.py > secrets.txt   (don't commit the output)
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "policy.json"
    policy = json.loads(path.read_text(encoding="utf-8"))  # validates the JSON
    compact = json.dumps(policy, ensure_ascii=False, separators=(",", ":"))
    if "'''" in compact:
        sys.exit("Policy text contains ''' and can't go in a TOML literal string.")

    print('LLM_PROVIDER = "openai_compat"')
    print('OPENAI_COMPAT_API_KEY = "your-deepseek-api-key"')
    print('GEMINI_API_KEY = "your-gemini-api-key"')
    print()
    print(f"POLICY_JSON = '''{compact}'''")


if __name__ == "__main__":
    main()
