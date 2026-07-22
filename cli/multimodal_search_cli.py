import argparse
from lib.multimodal_search import verify_image_emmbedding, image_search_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify_image_embedding", help="Verify image embedding")
    verify_parser.add_argument("image_path", help="Path to the image file")

    image_search_parser = subparsers.add_parser("image_search", help="Search for movies using an image")
    image_search_parser.add_argument("image_path", help="Path to the image file")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_emmbedding(args.image_path)

        case "image_search":
            results = image_search_command(args.image_path)
            print("Search Results:")
            for i, doc in enumerate(results, start=1):
                print(f"{i}. {doc['title']} (similarity: {doc['score']:.3f})")
                print(f"     {doc['description'][:100]}\n")


if __name__ == "__main__":
    main()