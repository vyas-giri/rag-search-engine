import mimetypes
import argparse
from openai import OpenAI
from dotenv import load_dotenv
from lib.constants import OPEN_ROUTER_BASE_URL, OPEN_ROUTER_FREE_MODEL
import os
import base64

def main() -> None:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    parser = argparse.ArgumentParser(description="Describe an image file")
    parser.add_argument("--image", type=str, help="Path to the image file")
    parser.add_argument("--query", type=str, help="Query to rewrite based on the image content")
    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    client = OpenAI(
        base_url=OPEN_ROUTER_BASE_URL,
        api_key=api_key,
    )

    img = None
    with open(args.image, "rb") as f:
        img = f.read()

    prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary
    """


    data_url = f"data:{mime};base64,{base64.b64encode(img).decode()}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt.strip()},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": args.query.strip()},
            ],
        }
    ]

    response = client.chat.completions.create(
        model=OPEN_ROUTER_FREE_MODEL,
        messages=messages,
    )

    content = response.choices[0].message.content
    print(f"Rewritten query: {content.strip()}")
    if response.usage is not None:
        print(f"Total tokens:    {response.usage.total_tokens}")

if __name__ == "__main__":
    main()