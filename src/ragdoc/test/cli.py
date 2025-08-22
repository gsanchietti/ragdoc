from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

from ragdoc.retrieval.retriever import Retriever, RetrieverConfig


def handle_query(args: argparse.Namespace) -> int:
    dsn = args.database_url or os.getenv("DATABASE_URL", "")
    if not dsn:
        print("DATABASE_URL not provided.")
        return 2
    text = args.text.strip()
    if not text:
        print("Empty query text.")
        return 2

    retriever = Retriever(RetrieverConfig(dsn=dsn, k=args.k))
    results = retriever.search(text, k=args.k)

    for i, r in enumerate(results, start=1):
        print(f"#{i} score={r.score:.4f} v={r.vscore:.4f} l={r.lscore:.4f} type={r.source_type} path={r.path}")
        if r.source_url:
            print(f"   url: {r.source_url}")
        if r.title:
            print(f"   title: {r.title}")
        print(f"   preview: {r.content_preview!r}")
    if not results:
        print("No results.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragdoc-test", description="Test utilities for ragdoc")
    sub = parser.add_subparsers(dest="action", required=True)

    q = sub.add_parser("query", help="Query the vector DB with a text prompt")
    q.add_argument("text", help="Query text")
    q.add_argument("--k", type=int, default=5, help="Top K results")
    q.add_argument("--database-url", default=os.getenv("DATABASE_URL", ""), help="Postgres DSN")
    q.set_defaults(func=handle_query)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
