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
    # Load .env file early to ensure environment variables are available for defaults
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not available, environment variables must be set manually
    
    parser = argparse.ArgumentParser(description="ragdoc fetch job")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config file")
    parser.add_argument("--out-root", default=Path("."), type=Path, help="Project root for outputs")
    parser.add_argument("--log-level", default=os.getenv("RAGDOC_LOG_LEVEL", "INFO"), help="Logging level")
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

        # Index fetched HTTP pages - one document at a time
        total_http_chunks = 0
        for path, url in http_outputs:
            content = path.read_bytes()
            if path.suffix.lower() in {".html", ".htm"}:
                text, title = html_to_text_with_title(content)
                source_type = "http"
            else:
                text = read_by_suffix(path)
                title = None
                source_type = "http"
            
            # Index this document individually
            chunks_indexed = indexer.index_document(
                source_type=source_type,
                source_url=url,
                path=str(path),
                title=title,
                text=text,
                embedder=embedder
            )
            total_http_chunks += chunks_indexed

        logging.info("HTTP indexing completed: %d total chunks indexed from %d documents", 
                    total_http_chunks, len(http_outputs))

        # Index content files (txt/md/rst) with configurable globs and roots - one document at a time
        globs: list[str] = args.glob
        roots: list[Path] = [Path(p) for p in (args.root or [])]
        if not roots:
            # Default to repos out_dir if present, else project root
            default_repo_dirs = {(out_root / (s.out_dir or "data/repos")) for s in cfg.git}
            roots = list(default_repo_dirs) if default_repo_dirs else [out_root]

        total_content_chunks = 0
        documents_processed = 0
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
                        source_type = "md"
                    else:
                        text = read_by_suffix(p)
                        source_type = {
                            ".rst": "rst",
                        }.get(p.suffix.lower(), "txt")
                    
                    # Index this document individually
                    chunks_indexed = indexer.index_document(
                        source_type=source_type,
                        source_url="",  # No source URL for local files
                        path=str(p),
                        title=title,
                        text=text,
                        embedder=embedder
                    )
                    total_content_chunks += chunks_indexed
                    documents_processed += 1

        logging.info("Content indexing completed: %d total chunks indexed from %d documents", 
                    total_content_chunks, documents_processed)
    finally:
        http.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
