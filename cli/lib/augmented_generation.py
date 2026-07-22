from lib.constants import load_movies, OPEN_ROUTER_BASE_URL, OPEN_ROUTER_FREE_MODEL, DEFAULT_QUERY_LIMIT
from lib.hybrid_search import HybridSearch
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url=OPEN_ROUTER_BASE_URL,
    api_key=api_key,
)

promps: dict[str, str] = {
    "rag": """You are a RAG agent for Webflyx, a movie streaming service.
            Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
            Provide a comprehensive answer that addresses the user's query.

            Query: {query}

            Documents:
            {results}

            Answer:""",
    "summarize": """Provide information useful to the query below by synthesizing data from multiple search results in detail.

                The goal is to provide comprehensive information so that users know what their options are.
                Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

                This should be tailored to Webflyx users. Webflyx is a movie streaming service.

                Query: {query}

                Search results:
                {results}

                Provide a comprehensive 3-4 sentence answer that combines information from multiple sources:""",
    "citations": """Answer the query below and give information based on the provided documents.

                The answer should be tailored to users of Webflyx, a movie streaming service.
                If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

                Query: {query}

                Documents:
                {results}

                Instructions:
                - Provide a comprehensive answer that addresses the query
                - Cite sources in the format [1], [2], etc. when referencing information
                - If sources disagree, mention the different viewpoints
                - If the answer isn't in the provided documents, say "I don't have enough information"
                - Be direct and informative

                Answer:""",
    "question": """Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

                Question: {query}

                Documents:
                {results}

                Instructions:
                - Answer questions directly and concisely
                - Be casual and conversational
                - Don't be cringe or hype-y
                - Talk like a normal person would in a chat conversation

                Answer:"""
}

def augmented_generation(query: str, command: str, limit: int = DEFAULT_QUERY_LIMIT) -> tuple[list[dict], str]:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    results = hybrid_search.rrf_search(query, limit=limit)

    prompt = promps[command].format(query=query, results=results)
    response = client.chat.completions.create(
        model=OPEN_ROUTER_FREE_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.choices[0].message.content.strip()
    return results, answer

def rag_command(query: str) -> None:
    results, answer = augmented_generation(query, "rag")
    print("Search Results:")
    for i, doc in enumerate(results, start=1):
        print(f"{i}. {doc['title']}")
    print(f"\nRAG Response:\n{answer}")

def summarize_command(query: str, limit: int = DEFAULT_QUERY_LIMIT) -> None:
    results, answer = augmented_generation(query, "summarize", limit=limit)
    print("Search Results:")
    for i, doc in enumerate(results, start=1):
        print(f"{i}. {doc['title']}")
    print(f"\nLLM Summary:\n{answer}")

def citations_command(query: str, limit: int = DEFAULT_QUERY_LIMIT) -> None:
    results, answer = augmented_generation(query, "citations", limit=limit)
    print("Search Results:")
    for i, doc in enumerate(results, start=1):
        print(f"{i}. {doc['title']}")
    print(f"\nLLM response with Citations:\n{answer}")

def question_command(query: str, limit: int = DEFAULT_QUERY_LIMIT) -> None:
    results, answer = augmented_generation(query, "question", limit=limit)
    print("Search Results:")
    for i, doc in enumerate(results, start=1):
        print(f"{i}. {doc['title']}")
    print(f"\nLLM response to Question:\n{answer}")
