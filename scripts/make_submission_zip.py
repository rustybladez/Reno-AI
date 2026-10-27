"""Build the zip attached to the submission email: the project plus the confidential files the
public repository leaves out (the policy and the real eval cases), and never any API key.

    uv run python scripts/make_submission_zip.py        -> dist/Reno-AI.zip

What goes in: every file git would commit (tracked or new but not ignored — so .env, .venv,
secrets.txt and the PDFs are excluded by .gitignore), plus data/policy.json and evals/cases.yaml.
Before writing, every file is checked for the API keys in your local .env and for anything that
looks like a key; the zip is not written if one is found.
"""

import re
import subprocess
import sys
import zipfile
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "dist" / "Reno-AI.zip"
TOP_FOLDER = "Reno-AI"
CONFIDENTIAL_EXTRAS = ["data/policy.json", "evals/cases.yaml"]
KEY_SETTINGS = ["GEMINI_API_KEY", "OPENAI_COMPAT_API_KEY"]
KEY_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),  # Google API keys
    re.compile(r"\bsk-[0-9A-Za-z]{20,}"),  # DeepSeek / OpenAI-style keys
]


def project_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    files = [f for f in result.stdout.splitlines() if (ROOT / f).is_file()]  # skip deleted files
    for extra in CONFIDENTIAL_EXTRAS:
        if not (ROOT / extra).is_file():
            sys.exit(f"Missing {extra}: it must be in the submission zip.")
    return sorted(set(files + CONFIDENTIAL_EXTRAS))


def find_secrets(files: list[str]) -> list[str]:
    local_keys = [v for k, v in dotenv_values(ROOT / ".env").items() if k in KEY_SETTINGS and v and len(v) > 8]
    problems = []
    for name in files:
        text = (ROOT / name).read_bytes().decode("utf-8", errors="ignore")
        if any(key in text for key in local_keys):
            problems.append(f"{name}: contains one of your API keys from .env")
        elif any(p.search(text) for p in KEY_PATTERNS):
            problems.append(f"{name}: contains something that looks like an API key")
    return problems


def main() -> None:
    files = project_files()
    problems = find_secrets(files)
    if problems:
        sys.exit("Not writing the zip:\n  " + "\n  ".join(problems))

    OUTPUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in files:
            zf.write(ROOT / name, f"{TOP_FOLDER}/{name}")

    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({len(files)} files, {OUTPUT.stat().st_size // 1024} KB).")
    print("Included confidential files:", ", ".join(CONFIDENTIAL_EXTRAS))
    print("Checked: no API keys inside.")


if __name__ == "__main__":
    main()
