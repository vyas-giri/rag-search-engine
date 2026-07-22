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
    #print(f"Cross encoder results: {[doc['title'] for doc in sorted_docs]}")  # debugging
    return sorted_docs[:limit]

def evaluate_results(results: list[dict], query: str) -> None:
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

            Query: "{query}"

            Results:
            {chr(10).join(result.get('title', '') for result in results)}

            Scale:
            - 3: Highly relevant
            - 2: Relevant
            - 1: Marginally relevant
            - 0: Not relevant

            Do NOT give any numbers other than 0, 1, 2, or 3.

            Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

            [2, 0, 3, 2, 0, 1]"""

    response = client.chat.completions.create(
        model=OPEN_ROUTER_FREE_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_content = response.choices[0].message.content.strip()
    content = json.loads(raw_content)
    for i, (eval, result) in enumerate(zip(content, results), start=1):
        print(f"{i}. {result.get('title', '')}: {eval}/3")