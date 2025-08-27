from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from ragdoc.retrieval.retriever import Retriever, RetrieverConfig


def handle_query(args: argparse.Namespace) -> int:
    # Set up debug logging for retrieval operations
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    
    dsn = args.database_url or os.getenv("DATABASE_URL", "")
    if not dsn:
        print("DATABASE_URL not provided.")
        return 2
    text = args.text.strip()
    if not text:
        print("Empty query text.")
        return 2

    # Create retriever config that ensures hybrid search is always executed
    # Handle BM25 enable/disable flags
    use_bm25 = args.use_bm25 and not args.no_bm25
    
    config = RetrieverConfig(
        dsn=dsn, 
        k=args.k,
        alpha=args.alpha,  # Allow configuring the hybrid balance
        use_fts=True,      # Always enable full-text search
        title_boost=args.title_boost,  # Allow configuring title boost
        use_bm25=use_bm25,  # Allow enabling/disabling BM25
        bm25_k1=args.bm25_k1,  # BM25 term frequency saturation
        bm25_b=args.bm25_b,    # BM25 length normalization
        sparse_weight=args.sparse_weight  # Weight for sparse vector component
    )
    
    retriever = Retriever(config)
    
    print(f"Executing hybrid search with query: {text!r}")
    print(f"Configuration: alpha={config.alpha:.2f}, use_fts={config.use_fts}, title_boost={config.title_boost:.1f}")
    print(f"BM25 Configuration: use_bm25={config.use_bm25}, k1={config.bm25_k1:.1f}, b={config.bm25_b:.2f}, sparse_weight={config.sparse_weight:.2f}")
    
    results = retriever.search(text, k=args.k)

    print(f"\nHybrid search results ({len(results)} found):")
    print("=" * 80)
    
    for i, r in enumerate(results, start=1):
        if config.use_bm25:
            print(f"#{i} hybrid_score={r.score:.4f} vector_score={r.vscore:.4f} lexical_score={r.lscore:.4f} sparse_score={r.sscore:.4f}")
        else:
            print(f"#{i} hybrid_score={r.score:.4f} vector_score={r.vscore:.4f} lexical_score={r.lscore:.4f}")
        print(f"   type={r.source_type} path={r.path}")
        if r.source_url:
            print(f"   url: {r.source_url}")
        if r.title:
            print(f"   title: {r.title}")
        print(f"   preview: {r.content_preview!r}")
        print()
        
    if not results:
        print("No results found.")
    else:
        print(f"Hybrid search completed successfully with {len(results)} results.")
    
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragdoc-test", description="Test utilities for ragdoc")
    sub = parser.add_subparsers(dest="action", required=True)

    q = sub.add_parser("query", help="Query the vector DB with a text prompt using hybrid search")
    q.add_argument("text", help="Query text")
    q.add_argument("--k", type=int, default=5, help="Top K results (default: 5)")
    q.add_argument("--database-url", default=os.getenv("DATABASE_URL", ""), help="Postgres DSN")
    q.add_argument("--alpha", type=float, default=0.7, help="Hybrid search balance: vector weight (default: 0.7)")
    q.add_argument("--title-boost", type=float, default=1.5, help="Title match boost factor (default: 1.5)")
    q.add_argument("--use-bm25", action="store_true", default=True, help="Enable BM25 sparse vector search (default: True)")
    q.add_argument("--no-bm25", action="store_true", help="Disable BM25 sparse vector search")
    q.add_argument("--bm25-k1", type=float, default=1.2, help="BM25 term frequency saturation parameter (default: 1.2)")
    q.add_argument("--bm25-b", type=float, default=0.75, help="BM25 length normalization parameter (default: 0.75)")
    q.add_argument("--sparse-weight", type=float, default=0.3, help="Weight for sparse vector component (default: 0.3)")
    q.add_argument("--debug", action="store_true", help="Enable debug logging to see executed queries")
    q.set_defaults(func=handle_query)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
