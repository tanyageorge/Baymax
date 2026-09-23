"""
Milestone 2: turn retrieved chunks into a real, grounded answer.

Written to work with EITHER Anthropic's Claude API or OpenAI's API — set
LLM_PROVIDER in your .env to "anthropic" or "openai" and the rest of the
pipeline doesn't change. Fill in .env once you've picked one and gotten a
key, and this starts working with no other code changes.

The core idea of "grounded generation" (the "AG" in RAG):
Instead of asking the LLM "what should I do about a headache?" (which lets
it answer from its own general training — no sources, no accountability),
we build a prompt that says: "Here are retrieved medical reference
passages. Answer using ONLY this information, and cite which document each
fact came from." That's what makes RAG trustworthy for a healthcare use
case — every claim traces back to a real source instead of the model's
own (possibly wrong, possibly outdated) internal knowledge.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "baymax_system_prompt.md"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").lower()

# Check your provider's docs for the current model ID before running —
# model names change over time:
#   Anthropic: https://docs.claude.com/en/docs/about-claude/models
#   OpenAI:    https://platform.openai.com/docs/models
ANTHROPIC_MODEL = "claude-sonnet-4-5"
OPENAI_MODEL = "gpt-4o-mini"


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text()


def build_user_message(query: str, retrieved_chunks: list[dict]) -> str:
    """
    retrieved_chunks: list of {"source": "01_fever.txt", "text": "..."}
    Packages retrieved context + the question into one message, with clear
    instructions to stay grounded and cite sources.
    """
    context_block = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in retrieved_chunks
    )
    return f"""Here is retrieved reference material relevant to the user's question:

{context_block}

---

User's question: {query}

Instructions: Answer using ONLY the information in the reference material above.
For every claim, cite which source file it came from (e.g. "according to
01_fever.txt"). If the reference material doesn't fully answer the question,
say so honestly rather than filling gaps from general knowledge. If anything
in the question suggests a medical emergency, say so clearly and recommend
calling emergency services, regardless of what the retrieved sources say."""


def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    system_prompt = load_system_prompt()
    user_message = build_user_message(query, retrieved_chunks)

    if LLM_PROVIDER == "anthropic":
        return _generate_anthropic(system_prompt, user_message)
    elif LLM_PROVIDER == "openai":
        return _generate_openai(system_prompt, user_message)
    else:
        raise RuntimeError(
            "Set LLM_PROVIDER to 'anthropic' or 'openai' in backend/.env "
            "(and fill in the matching API key) before running generation."
        )


def _generate_anthropic(system_prompt: str, user_message: str) -> str:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def _generate_openai(system_prompt: str, user_message: str) -> str:
    from openai import OpenAI

    client = OpenAI()  # reads OPENAI_API_KEY from env
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content