"""Streamlit UI for the XYZ Leave Policy Assistant (PRD §11).

    uv run streamlit run app.py

All answering logic lives in leave_assistant.ask(); this file only handles the chat UI.
Conversations live in st.session_state only and are never stored.
"""

import os

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from leave_assistant.assistant import ask, get_policy
from leave_assistant.config import load_settings, today_in_dhaka
from leave_assistant.policy import PolicyNotFoundError
from leave_assistant.prompt import format_today
from leave_assistant.providers import get_provider
from leave_assistant.providers.base import ProviderBusyError, ProviderConfigError, ProviderError
from leave_assistant.schema import AssistantResponse, Turn

SETTING_NAMES = [
    "LLM_PROVIDER", "GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_FALLBACK_MODELS", "GEMINI_THINKING_LEVEL",
    "POLICY_JSON", "POLICY_PATH", "OPENAI_COMPAT_BASE_URL", "OPENAI_COMPAT_API_KEY", "OPENAI_COMPAT_MODEL",
    "OPENAI_COMPAT_THINKING", "OPENAI_COMPAT_REASONING_EFFORT",
]

SAMPLE_QUESTIONS = {
    "In the policy": [
        "Can I carry forward unused annual leave?",
        "When do I need a medical certificate for sick leave?",
    ],
    "Calculations": [
        "My basic salary is 45,000 BDT and I'm resigning with 12 unused annual leave days. How much will I be paid for them?",
        "My wedding is on 10 December 2026 and my marriage leave starts on 7 December 2026. By when must I apply?",
        "I'm taking annual leave from Thursday 5 November to Thursday 12 November 2026. How many leave days will be deducted?",
    ],
    "Not in the policy": [
        "My cousin passed away. How many days of bereavement leave do I get?",
        "What is the work-from-home policy?",
    ],
    "Someone else's data": [
        "How many leave days has Rahim taken?",
        "Why is my colleague on medical leave?",
    ],
}

# One starter per behaviour reviewers check: in the policy, not in it, someone else's data.
WELCOME_QUESTIONS = [
    (":material/menu_book:", "Can I carry forward unused annual leave?"),
    (":material/help:", "My cousin passed away. How many days of bereavement leave do I get?"),
    (":material/lock:", "How many leave days has Rahim taken?"),
]
MAX_QUESTION_CHARS = 500

BUSY_MESSAGE = "The assistant is busy right now. Please wait a minute and try again."
SETUP_MESSAGE = "The assistant isn't set up yet. {error}"


def copy_secrets_to_env() -> str | None:
    """On Streamlit Cloud, settings come from st.secrets; the package reads the environment.

    Returns an error message if secrets exist but can't be parsed (e.g. a truncated POLICY_JSON).
    """
    try:
        secrets = {name: st.secrets[name] for name in SETTING_NAMES if name in st.secrets}
    except (FileNotFoundError, StreamlitSecretNotFoundError) as e:
        # Streamlit raises the same error for "no secrets file" (fine locally: .env is used)
        # and for a malformed one, which must not be ignored silently.
        return str(e) if "Error parsing secrets" in str(e) else None
    for name, value in secrets.items():
        os.environ[name] = str(value)
    return None


@st.cache_resource
def cached_provider():
    """One provider per server process, so model cooldowns are shared across reruns and users."""
    return get_provider(load_settings())


def render_answer(response: AssistantResponse, show_debug: bool) -> None:
    st.markdown(response.text)
    if response.type in ("not_covered", "partial"):
        # Shown under the answer, not inside it, so the fallback sentence itself stays exact.
        hr_email = get_policy().meta.get("hr_contact")
        if hr_email:
            st.markdown(f"HR email: [{hr_email}](mailto:{hr_email})")
    if response.citations:
        policy = get_policy()
        with st.expander("Policy text", icon=":material/menu_book:", type="compact"):
            for section_id in dict.fromkeys(c.section_id for c in response.citations):
                section = policy.get(section_id)
                if section:
                    st.markdown(f"**{policy.label(section_id)}**")
                    st.markdown(section.text)
    if response.calculations:
        with st.expander("How this was calculated", icon=":material/calculate:", type="compact"):
            for line in response.calculations:
                st.markdown(f"- {line}")
    if show_debug:
        with st.expander("Debug details", icon=":material/bug_report:", type="compact"):
            st.json(response.debug)


def sidebar(today_text: str, provider_label: str) -> bool:
    """Draws the sidebar. Returns the debug toggle's value."""
    with st.sidebar:
        st.markdown("### Try a sample question")
        for group, questions in SAMPLE_QUESTIONS.items():
            st.caption(group)
            for question in questions:
                if st.button(question, width="stretch"):
                    st.session_state.pending_question = question

        st.divider()
        st.markdown(f"**Today:** {today_text}")
        st.caption("Relative dates like \"next Sunday\" are worked out from this date.")
        if st.button("Clear chat", icon=":material/delete:"):
            st.session_state.messages = []
            st.rerun()
        show_debug = st.toggle("Show debug details", value=False)

        st.divider()
        st.markdown("### About")
        st.markdown(
            "- Answers come only from the leave policy, and every answer cites its section.\n"
            "- If the policy doesn't cover a question, it says so instead of guessing.\n"
            "- It never shares other employees' personal data.\n"
        )
        st.caption(f"Models: {provider_label}. Dates and sums are calculated by code, not the model.")
    return show_debug


def main() -> None:
    st.set_page_config(page_title="Reno AI", page_icon=":material/description:")
    secrets_error = copy_secrets_to_env()
    if secrets_error:
        st.error(
            f"{secrets_error}. Check the Secrets box: each value must be valid TOML, and POLICY_JSON "
            "must start and end with three single quotes. Regenerate it with scripts/make_secrets.py.",
            title="The app's secrets couldn't be read",
        )
        st.stop()

    st.title("Reno AI — Leave Policy Assistant")
    st.caption(
        "Answers come only from the XYZ Company Leave Policy (HR-POL-004 v2.0). "
        "Sample document for assessment purposes."
    )

    try:
        get_policy()
        provider = cached_provider()
    except (PolicyNotFoundError, ProviderConfigError) as e:
        st.error(SETUP_MESSAGE.format(error=e), title="Setup needed")
        st.stop()

    today = today_in_dhaka()
    provider_label = " → ".join(getattr(provider, "models", [provider.model]))  # primary, then fallback
    show_debug = sidebar(format_today(today), provider_label)

    if "messages" not in st.session_state:
        st.session_state.messages = []  # {"role", "content", "response"}; session only, never stored

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("response"):
                render_answer(message["response"], show_debug)
            else:
                st.markdown(message["content"])

    # Before the first question, show starters in the main area too: on a phone the sidebar
    # starts collapsed, so the page would otherwise be empty.
    welcome = st.empty()
    if not st.session_state.messages:
        with welcome.container():
            st.markdown("Ask a question about the leave policy, or try one of these:")
            for i, (icon, starter) in enumerate(WELCOME_QUESTIONS):
                if st.button(starter, icon=icon, key=f"welcome_{i}", width="stretch"):
                    st.session_state.pending_question = starter

    # "disable": no second question while one is being answered (avoids duplicate model calls).
    # max_chars: a public link on a paid key — a pasted essay shouldn't cost money.
    typed = st.chat_input("Ask about the leave policy…", submit_mode="disable", max_chars=MAX_QUESTION_CHARS)
    question = typed or st.session_state.pop("pending_question", None)

    if question:
        welcome.empty()
        history = [Turn(role=m["role"], content=m["content"]) for m in st.session_state.messages]
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Checking the policy…"):
                    response = ask(question, history=history, today=today, provider=provider)
            except ProviderBusyError:
                st.warning(BUSY_MESSAGE, title="Busy")
                st.session_state.messages.pop()  # let the user ask again
                return
            except ProviderError as e:
                st.error(str(e), title="Something went wrong calling the model")
                st.session_state.messages.pop()
                return
            render_answer(response, show_debug)
        st.session_state.messages.append({"role": "assistant", "content": response.text, "response": response})

    st.caption("Conversations are not stored.")


main()
