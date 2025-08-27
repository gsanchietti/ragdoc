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
from .text_extraction import html_to_text, html_to_text_with_title, chunk_text, read_by_suffix, read_markdown_with_title, extract_title_from_markdown
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
    parser.add_argument(
        "--glob",
        action="append",
        default=["**/*.md", "**/*.txt", "**/*.rst"],
        help="File glob(s) to index; can repeat. Defaults to **/*.md, **/*.txt, **/*.rst",
    )
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="Root directory(ies) to search for files; can repeat. Defaults to repo dirs or project root",
    )
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
            if path.suffix.lower() in {".html", ".htm"}:
                text, title = html_to_text_with_title(content)
            else:
                text = read_by_suffix(path)
                title = None
            for chunk in chunk_text(text):
                http_chunks.append(EmbeddingChunk(text=chunk, source_type="http", source_url=url, path=str(path), title=title))

        _index_chunks_batched(http_chunks, embedder, indexer)

        # Index content files (txt/md/rst) with configurable globs and roots
        globs: list[str] = args.glob
        roots: list[Path] = [Path(p) for p in (args.root or [])]
        if not roots:
            # Default to repos out_dir if present, else project root
            default_repo_dirs = {(out_root / (s.out_dir or "data/repos")) for s in cfg.git}
            roots = list(default_repo_dirs) if default_repo_dirs else [out_root]

        content_chunks: list[EmbeddingChunk] = []
        for root in roots:
            for pattern in globs:
                for p in root.rglob(pattern):
                    if not p.is_file():
                        continue
                    logging.info("Reading %s", p)
                    
                    # Extract title based on file type
                    title = None
                    if p.suffix.lower() in {".md", ".markdown"}:
                        text, title = read_markdown_with_title(p)
                    else:
                        text = read_by_suffix(p)
                    
                    stype = {
                        ".md": "md",
                        ".markdown": "md",
                        ".rst": "rst",
                    }.get(p.suffix.lower(), "txt")
                    for chunk in chunk_text(text):
                        content_chunks.append(EmbeddingChunk(text=chunk, source_type=stype, path=str(p), title=title))

        _index_chunks_batched(content_chunks, embedder, indexer)
    finally:
        http.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def _index_chunks_batched(
    chunks: list[EmbeddingChunk], embedder: EmbeddingClient, indexer: PgVectorIndexer, batch_size: int = 32
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
