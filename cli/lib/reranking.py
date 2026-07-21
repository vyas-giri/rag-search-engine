import os
from dotenv import load_dotenv
from openai import OpenAI
from lib.constants import OPEN_ROUTER_FREE_MODEL, DEFAULT_QUERY_LIMIT, OPEN_ROUTER_BASE_URL, CROSS_ENCODER_MODEL
import time
import json
from sentence_transformers import CrossEncoder


load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url=OPEN_ROUTER_BASE_URL,
    api_key=api_key,
)

prompts = {
    "individual": """Rate how well this movie matches the search query.

                Query: {query}
                Movie: {title} - {description}

                Consider:
                - Direct relevance to query
                - User intent (what they're looking for)
                - Content appropriateness

                Rate 0-10 (10 = perfect match).
                Output ONLY the number in your response, no other text or explanation.

                Score:""",
    "batch": """Rank the movies listed below by relevance to the following search query.

            Query: "{query}"

            Movies:
            {doc_list_str}

            Return the movie IDs in order of relevance, best match first.

            Your response must be a raw JSON array of integers.
            Do not wrap the JSON in Markdown. Do not use a ```json code block.
            Do not include any explanatory text.

            For example:
            [75, 12, 34, 2, 1]

            Ranking:"""
}

def rerank_results(results: list[dict], rerank_method: str, query: str, limit: int = DEFAULT_QUERY_LIMIT) -> list[dict]:
    if rerank_method == "individual":
        return individual_rerank(results, query, limit)

    elif rerank_method == "batch":
        return batch_rerank(results, query, limit)

    elif rerank_method == "cross_encoder":
        return cross_encoder_rerank(results, query, limit)
    
    else:
        raise ValueError(f"Unsupported reranking method: {rerank_method}")

def individual_rerank(results: list[dict], query: str, limit: int = DEFAULT_QUERY_LIMIT) -> list[dict]:
    for doc in results:
        prompt = prompts["individual"].format(query=query, title=doc.get("title", ""), description=doc.get("description", ""))
        response = client.chat.completions.create(
            model=OPEN_ROUTER_FREE_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        score = response.choices[0].message.content.strip()
        try:
            doc["individualRerankScore"] = float(score)
        except ValueError:
            doc["individualRerankScore"] = 0.0
        time.sleep(4)  # To avoid hitting rate limits
    
    sorted_docs = sorted(results, key=lambda x: x["individualRerankScore"], reverse=True)
    return sorted_docs[:limit]

def batch_rerank(results: list[dict], query: str, limit: int = DEFAULT_QUERY_LIMIT) -> list[dict]:
    doc_list_str = "\n".join([f"{doc.get('id', '')}: {doc.get('title', '')} - {doc.get('description', '')}" for doc in results])
    prompt = prompts["batch"].format(query=query, doc_list_str=doc_list_str)
    response = client.chat.completions.create(
        model=OPEN_ROUTER_FREE_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_content = response.choices[0].message.content.strip()
    print("RAW MODEL RESPONSE:")
    print(raw_content)
    try:
        ranked_ids = json.loads(response.choices[0].message.content.strip())
    except json.JSONDecodeError:
        ranked_ids = []

    for i, id in enumerate(ranked_ids, start=1):
        for doc in results:
            if doc.get("id") == id:
                doc["batchRerankPosition"] = i

    sorted_docs = sorted(results, key=lambda x: x.get("batchRerankPosition", 0), reverse=False)
    return sorted_docs[:limit]


def cross_encoder_rerank(results: list[dict], query: str, limit: int = DEFAULT_QUERY_LIMIT) -> list[dict]:
    pairs = [(query, f"{doc.get('title', '')} - {doc.get('description', '')}") for doc in results]
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    scores = cross_encoder.predict(pairs)
    for doc, score in zip(results, scores):
        doc["crossEncoderScore"] = score
    
    sorted_docs = sorted(results, key=lambda x: x["crossEncoderScore"], reverse=True)
    return sorted_docs[:limit]