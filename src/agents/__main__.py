"""Ask the agent a question from the command line.

    uv run python -m src.agents "family-friendly movie with bears"
    uv run python -m src.agents --mermaid      # print the graph diagram
"""

import argparse
import json
import time

from .graph import build_graph, run_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG over the movie catalog.")
    parser.add_argument("question", nargs="?", help="question to ask")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    parser.add_argument("--mermaid", action="store_true", help="print the graph as Mermaid")
    args = parser.parse_args()

    if args.mermaid:
        # Structure only: fakes avoid loading models or needing an API key.
        class _Stub:
            def retrieve(self, query, k=5):
                return []

        print(build_graph(llm=object(), retriever=_Stub()).get_graph().draw_mermaid())
        return
    if not args.question:
        parser.error("a question is required (or use --mermaid)")

    graph = build_graph()
    started = time.perf_counter()
    result = run_agent(graph, args.question)
    elapsed = time.perf_counter() - started

    citations = [{"id": d["id"], "title": d["title"]} for d in result.get("citations", [])]
    if args.json:
        print(json.dumps({
            "answer": result["answer"],
            "citations": citations,
            "path": result["path"],
            "final_query": result.get("query"),
            "rewrites": result.get("rewrites", 0),
            "latency_s": round(elapsed, 2),
        }, indent=2))
        return

    print(result["answer"])
    if citations:
        print("\nSources:")
        for c in citations:
            print(f"  [{c['id']}] {c['title']}")
    print(f"\npath: {' → '.join(result['path'])}")
    if result.get("rewrites"):
        print(f"final query: {result['query']!r} after {result['rewrites']} rewrite(s)")
    print(f"took {elapsed:.1f}s")


if __name__ == "__main__":
    main()
