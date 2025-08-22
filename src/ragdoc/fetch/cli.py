from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

from .http_fetcher import HttpFetcher
from .git_fetcher import GitRepoFetcher
from .models import FetchConfig, GitSource, HttpSource
from .text_extraction import html_to_text, read_markdown, chunk_text
from .indexer import EmbeddingClient, PgVectorIndexer, EmbeddingChunk


def _parse_config(path: Path) -> FetchConfig:
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    http_list = [HttpSource(**item) for item in data.get("http", [])]
    git_list = [GitSource(**item) for item in data.get("git", [])]
    return FetchConfig(http=http_list, git=git_list)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ragdoc fetch job")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config file")
    parser.add_argument("--out-root", default=Path("."), type=Path, help="Project root for outputs")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", ""), help="Postgres DSN (or set DATABASE_URL)")
    parser.add_argument("--md-glob", default="**/*.md", help="Glob for Markdown files to index")
    parser.add_argument("--md-root", action="append", default=[], help="Root directory to search for Markdown (can repeat)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    cfg = _parse_config(args.config)

    out_root = args.out_root

    http = HttpFetcher()
    git = GitRepoFetcher()

    try:
        # 1) Fetch HTTP and index
        http_outputs: list[tuple[Path, str]] = []
        if cfg.http:
            http_outputs = http.fetch_many(cfg.http, out_root)

        # 2) Update/pull Git repos (content used for Markdown indexing)
        repo_roots: list[Path] = []
        if cfg.git:
            for src in cfg.git:
                repo_dir = git.fetch_one(src, out_root)
                repo_roots.append(repo_dir)

        # Prepare DB and embedding clients
        db_url = args.database_url or os.getenv("DATABASE_URL", "")
        if not db_url:
            logging.error("DATABASE_URL not provided. Set --database-url or env DATABASE_URL.")
            return 2

        indexer = PgVectorIndexer(db_url)
        embedder = EmbeddingClient()

        # Index fetched HTTP pages
        http_chunks: list[EmbeddingChunk] = []
        for path, url in http_outputs:
            content = path.read_bytes()
            text = html_to_text(content) if path.suffix.lower() in {".html", ".htm"} else read_markdown(path)
            for chunk in chunk_text(text):
                http_chunks.append(EmbeddingChunk(text=chunk, source_type="http", source_url=url, path=str(path)))

        _index_chunks_batched(http_chunks, embedder, indexer)

        # Index Markdown files (configurable glob and roots)
        md_glob = args.md_glob
        md_roots: list[Path] = [Path(p) for p in args.md_root] if args.md_root else []
        if not md_roots:
            # Default to repos out_dir if present, else project root
            default_repo_dirs = { (out_root / (s.out_dir or "data/repos")) for s in cfg.git }
            md_roots = list(default_repo_dirs) if default_repo_dirs else [out_root]

        md_chunks: list[EmbeddingChunk] = []
        for root in md_roots:
            for md_path in root.rglob(md_glob):
                if not md_path.is_file():
                    continue
                text = read_markdown(md_path)
                for chunk in chunk_text(text):
                    md_chunks.append(EmbeddingChunk(text=chunk, source_type="md", path=str(md_path)))

        _index_chunks_batched(md_chunks, embedder, indexer)
    finally:
        http.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def _index_chunks_batched(
    chunks: list[EmbeddingChunk], embedder: EmbeddingClient, indexer: PgVectorIndexer, batch_size: int = 128
) -> None:
    if not chunks:
        return
    logging.info("Indexing %d chunks", len(chunks))
    i = 0
    while i < len(chunks):
        batch = chunks[i : i + batch_size]
        vectors = embedder.embed_batch([c.text for c in batch])
        indexer.upsert(batch, vectors)
        i += batch_size
