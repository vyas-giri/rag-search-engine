import os
from dotenv import load_dotenv
from openai import OpenAI
from lib.constants import OPEN_ROUTER_FREE_MODEL, OPEN_ROUTER_BASE_URL


load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url=OPEN_ROUTER_BASE_URL,
    api_key=api_key,
)

messages=[
    {
        "role": "user",
        "content": "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
    }
]

response = client.chat.completions.create(
    model=OPEN_ROUTER_FREE_MODEL,
    messages=messages,
)

print(response.choices[0].message.content)
print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Response tokens: {response.usage.completion_tokens}")
