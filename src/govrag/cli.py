"""Command line: `govrag search "..."` (offline) and `govrag ask "..."` (needs Claude credentials)."""

from __future__ import annotations

import argparse

from govrag.chunk import load_chunks
from govrag.retrieve import MODES, Retriever


def main() -> None:
    parser = argparse.ArgumentParser(prog="govrag")
    sub = parser.add_subparsers(dest="command", required=True)
    search = sub.add_parser("search", help="show the top retrieved passages")
    search.add_argument("question")
    search.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    search.add_argument("-k", type=int, default=5)
    ask = sub.add_parser("ask", help="answer with citations using Claude")
    ask.add_argument("question")
    args = parser.parse_args()

    retriever = Retriever.build(load_chunks())
    if args.command == "search":
        for hit in retriever.search(args.question, k=args.k, mode=args.mode):
            print(f"{hit.score:8.3f}  {hit.chunk.header}\n          {hit.chunk.url}")
        return

    from govrag.generate import answer

    result = answer(args.question, retriever.search(args.question))
    print(result.text)
    for i, url in enumerate(result.cited_urls, start=1):
        print(f"  [{i}] {url}")


if __name__ == "__main__":
    main()
