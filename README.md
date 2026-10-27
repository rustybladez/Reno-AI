# Reno AI — Leave Policy Assistant

**Live demo: https://reno-ai-leave-assistant.streamlit.app/** · no login needed

If the page says the app has gone to sleep (Streamlit pauses apps that haven't been visited for a while), click the button to wake it and wait about 30 seconds.

## What it does

An assistant that answers employee questions using **only** the sample XYZ Company Leave Policy (HR-POL-004 v2.0):

- **R1 — Answers correctly and cites the policy.** Every answer names its section (e.g. *Section 4.1(d) – Annual Leave*) and shows the policy text behind it. Calculations (leave encashment, Leave Without Pay deductions, accrual, deadlines, working days) are done by code, and the working is shown.
- **R2 — Admits when the policy doesn't cover a question.** It replies exactly *"I don't know, please contact HR."* — no guessing and no general knowledge.
- **R3 — Protects other employees' data.** Questions about a colleague's leave, health or whereabouts get a courteous refusal that points to HR, whoever the user claims to be.

## Quick start

The easiest way to try it is the **live demo** above. To run it on your own computer:

**1. Get the project.** Unzip `Reno-AI.zip`: it includes the policy file, so nothing else needs to be downloaded. (A clone of the public repository has no policy file; see *Confidentiality*.)

**2. Install and run** from the `Reno-AI` folder, with whichever of these suits you. The app then opens at http://localhost:8501.

*Option A — you already have Python 3.12 or newer* (check with `python --version`): plain pip.

```bash
python -m venv .venv
.venv\Scripts\activate                 # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

*Option B — no Python, or an older one:* [uv](https://docs.astral.sh/uv/) installs the right Python by itself. Install uv once, then one command installs everything (about a minute the first time) and starts the app:

```bash
# Install uv — Windows (PowerShell):
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Install uv — macOS / Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

uv run streamlit run app.py
```

Tested on Python 3.12 and 3.13. Python 3.11 and older won't work: some pinned packages (e.g. numpy) need 3.12+.

**3. Paste an API key** into the box the app shows. A free Gemini key is enough (get one at aistudio.google.com/apikey). The key is kept only in that browser session and never saved.

**API keys in a file instead.** To avoid pasting a key each time, copy `.env.example` to `.env` and set at least one key there; with both, the first answers and the second takes over if it is busy.

| Provider | Variable | Notes |
|---|---|---|
| DeepSeek `deepseek-flash` (primary) | `OPENAI_COMPAT_API_KEY` | Paid, a few cents for the whole eval suite; no rate quotas. Key at platform.deepseek.com |
| Gemini `gemini-3.5-flash-lite` (fallback) | `GEMINI_API_KEY` | Free key at aistudio.google.com; free tier has small per-model quotas |

Any OpenAI-compatible endpoint works in place of DeepSeek (`OPENAI_COMPAT_BASE_URL`, `OPENAI_COMPAT_MODEL`), including a self-hosted model. All settings are listed in [.env.example](.env.example).

**The policy file.** `data/policy.json` is not in the public repository (see *Confidentiality* below); it is included in the submitted zip (built with `uv run python scripts/make_submission_zip.py`, which refuses to write the zip if any API key would be inside). Without it the app shows a setup message.

Command-line check without the UI: `uv run python -m leave_assistant "Can I carry forward unused annual leave?" --debug`

### Deploying to Streamlit Community Cloud

1. New app → this repository, branch `main`, main file `app.py`; under *Advanced settings* choose Python 3.12. Dependencies install from `uv.lock` (Community Cloud reads it before `requirements.txt`, which is only for pip users).
2. Generate the secrets (the policy goes in as `POLICY_JSON`, since it isn't in the repository): `uv run python scripts/make_secrets.py > secrets.txt`, then replace the key placeholders.
3. Paste the whole of `secrets.txt` into *App settings → Secrets*, save, and reboot the app. `secrets.txt` is gitignored; delete it afterwards.

## Try these

In the live demo or locally (they are also in the app's sidebar):

1. *Can I carry forward unused annual leave?* — answered from the policy, with the section cited
2. *Is there study or exam leave?* — not in the policy → "I don't know, please contact HR."
3. *How many leave days has Rahim taken?* — another employee's data → polite refusal

## How it works

```
data/policy.json ─► system prompt ─► LLM (DeepSeek → Gemini) ◄─► calculation tools (Python)
(hand-verified)     (+ today's date)        │ JSON: type, answer, citations + quotes
                                            ▼
                                guardrails (Python, deterministic)
                                • check every citation's quote is really in that section
                                • exact fallback / privacy texts from templates
                                • every number traced to the policy, a tool or the user
                                            ▼
                               Streamlit UI · eval runner · unit tests
```

The model only **classifies and drafts**. What the user sees is decided by deterministic code: the fallback and privacy messages are fixed templates the model never writes, and an answer whose citations can't be verified becomes the fallback.

## Design decisions

| # | Decision | Why |
|---|---|---|
| D1 | **Whole policy in the prompt, no RAG.** | The policy is ~4k tokens. Knowing what is *not* covered needs the whole document, and its cross-references between sections break when chunked. |
| D2 | **Policy converted once to JSON and checked by hand.** | Automatic PDF extraction misplaced the tables and joined lines. [scripts/extract_policy.py](scripts/extract_policy.py) makes a draft and a word-by-word coverage report; a person verifies it. |
| D3 | **Structured output + deterministic guardrails.** | The hard requirements are enforced in code, not trusted to the model. |
| D4 | **Quote verification.** | Each citation carries a short verbatim quote, which must be found in the cited section (a near-match may not change any number). Fabricated quotes are rejected. |
| D5 | **Provider chain: DeepSeek first, Gemini as fallback.** | Gemini's free tier was too unreliable for a demo (per-model quotas as low as 20 requests/day). DeepSeek is paid but cheap, with no quotas. Only models that passed the eval suite are in the chain. |
| D6 | **No conversation storage.** | Users may type colleagues' names or health details. Chat lives in the browser session only; nothing is logged. |
| D7 | **Fixed refusal and fallback texts.** | Consistent, courteous, and cannot drift or invent HR channels. |
| D8 | **Calculation tools.** | The model picks the rule; Python does the arithmetic and calendar maths (weekly holidays, "within N working days"). Tools return their working step by step, so every number shown can be traced. |

**Models and function calling.** Primary: `deepseek-flash` (DeepSeek-V4.1-Flash, thinking mode at its default) through the OpenAI-compatible API, with tools and JSON output in the same request. Fallback: `gemini-3.5-flash-lite` with thinking level `medium` (lower levels skipped tool calls), also tools + JSON schema in one request. The tool loop (max 6 rounds) is the same code for both providers.

## Tests and evaluation

```bash
uv run pytest                                  # 133 offline tests, no API calls
uv run python -m evals.run_evals --no-fallback --delay 0   # full eval suite (needs keys + the real cases)
```

Without `data/policy.json` (e.g. a fresh clone of the public repository), 8 tests that check the real policy file are skipped and the rest pass. The tests use `uv sync`'s dev dependencies; with pip, also run `pip install pytest`.

The eval suite has **66 cases from the brief and PRD** (in-policy, not covered, privacy, colleague questions, partial, prompt injection, calculations) plus **14 hold-out cases**. All checks are deterministic: exact fallback and refusal texts, cited sections, required facts, required tool calls, and that no number in an answer is unsourced. Results are written to `evals/results/`.

**Latest results** (eval date fixed at 26 Oct 2026):

| Category | Target | DeepSeek `deepseek-flash` (2 runs, final prompt) | Gemini `3.5-flash-lite` (previous prompt*) |
|---|---|---|---|
| Not covered (exact fallback) | 100% | 10/10, 10/10 | 10/10 |
| Privacy (exact refusal) | 100% | 5/5, 5/5 | 5/5 |
| Prompt injection | 100% | 3/3, 3/3 | 3/3 |
| Calculations | ≥ 90% | 16/16, 16/16 | 15/16 |
| In policy | — | 26/26, 26/26 | 24/26 |
| Colleague policy questions | — | 2/2, 2/2 | 1/2 |
| Partial, smalltalk | — | 3/3, 3/3 | 3/3 |
| **Overall** | ≥ 95% | **65/65 (100%), 65/65 (100%)** | **61/65 (94%)** |
| **Hold-out** (not used for tuning) | — | 13/14, 14/14 | 14/14 |

Mean response time: about 2.8 s on DeepSeek, 3.5 s on Gemini (calculations take longer: one model call per tool round).

\* The Gemini run used the prompt *before* the final rule (on statuses the policy defines) was added. Re-running it on the final prompt was cut short because the free tier's daily quota for the model ran out during testing — the same problem that made DeepSeek the primary.

**Gemini as the fallback** meets every hard requirement (not covered, privacy and injection all 100%) but was just under the 95% overall target: its misses were all "I don't know, please contact HR." on questions the policy does answer — the safe direction of error, and the reason DeepSeek, not Gemini, is the primary.

**How the prompt was tuned, honestly.** The prompt was improved by running the suite and fixing failures with *general* rules (e.g. "the policy being silent is never a rule", "tools only for real calculations") — never by quoting test questions in the prompt. The hold-out cases were added afterwards to check that this generalises. One later rule (about lists of eligible relatives) was prompted partly by a hold-out failure, so the hold-out set is no longer fully unseen; the main-set case B01 motivates the same rule.

**What still varies between runs.** The one hold-out miss (13/14 in one run) was a correct date the model worked out itself instead of with a date tool; the number check flagged it, as designed. An earlier version of the prompt consistently declined one eligibility question whose answer needs two clauses combined (a required status in one clause, its definition in another). A general rule now separates that from a case the policy never mentions (which stays "I don't know"). The rule was written while both the main-set case and its hold-out counterpart were failing, so that hold-out case is not an unseen test of it.

## Limitations and next steps

- **Hand-verified ingestion doesn't scale** to many policies. Next: automated extraction with a review screen, and retrieval once there are many documents.
- **No identity or records integration:** it can't answer "how many days do I have left?". Next: connect to the HR system with proper authentication, so employees see only their own records.
- **Strictness vs. helpfulness:** the no-inference rules are a trade-off: too loose and the model reasons past the policy, too strict and it declines answerable questions. The current balance passes every case, but new kinds of question may find the edge.
- **Small eval set** (80 cases); results vary slightly between runs because LLMs are not fully deterministic even at temperature 0. Next: more cases, written by HR staff.
- **Bangla** works in tests but is not tuned or counted in the targets.

## Data privacy

- Conversations are not stored or logged.
- On **Gemini's free tier, Google uses prompts to improve its products** (per Google's pricing page); the paid tier does not. That is acceptable for this dummy policy, not for real employee data.
- **DeepSeek's API is operated from China.** Fine for a sample document; for real employee data an organisation would choose an approved provider or a self-hosted model — the OpenAI-compatible provider supports either without code changes.

## Confidentiality

The assessment asks that the sample policy is not shared outside the assessment, so the policy file (`data/policy.json`), the real eval cases (`evals/cases.yaml`) and their results are **gitignored**. The public repository contains only a dummy example policy and example cases, so tests run without the real document. The reviewers receive both files privately, in the zip attached to the submission email; on Streamlit Cloud the policy is supplied as a secret (`POLICY_JSON`, generated with [scripts/make_secrets.py](scripts/make_secrets.py)).

## Project layout

```
app.py                     Streamlit UI
leave_assistant/           the assistant (no Streamlit imports)
  assistant.py             ask(): the single entry point, incl. the tool loop
  guardrails.py            citation/quote checks, fixed texts, number provenance
  tools.py                 calculation tools (date and arithmetic)
  policy.py, prompt.py     policy loading and the system prompt
  providers/               DeepSeek (OpenAI-compatible), Gemini, chain, fake (tests)
prompts/system_prompt.md   the system prompt
evals/                     eval runner and example cases
scripts/                   policy extraction, secrets helper, submission zip
tests/                     offline unit tests
```
