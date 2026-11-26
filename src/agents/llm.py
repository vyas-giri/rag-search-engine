"""Chat model factory for the agent (OpenRouter via LangChain)."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
# Cheap, reliable at structured output. Override with AGENT_MODEL in .env.
DEFAULT_AGENT_MODEL = "anthropic/claude-haiku-5.5"


@lru_cache(maxsize=1)
def get_chat_model():
    """Return the shared chat model. Created lazily so imports never need a key."""
    from langchain_openrouter import ChatOpenRouter

    load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    return ChatOpenRouter(
        model=os.environ.get("AGENT_MODEL", DEFAULT_AGENT_MODEL),
        temperature=0,
        max_retries=2,
    )
