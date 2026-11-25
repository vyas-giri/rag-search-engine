"""Lazily constructed OpenRouter client.

Creating the client on first use (instead of at import time) means modules
that merely import LLM helpers, such as hybrid search, tests, or CI, work
without an ``OPENROUTER_API_KEY``. The key is only required when an LLM call
is actually made.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

from lib.constants import OPEN_ROUTER_BASE_URL


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    return OpenAI(base_url=OPEN_ROUTER_BASE_URL, api_key=api_key)
