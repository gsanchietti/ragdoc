from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from .http_fetcher import HttpFetcher
from .git_fetcher import GitRepoFetcher
from .models import FetchConfig, GitSource, HttpSource


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
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    cfg = _parse_config(args.config)

    out_root = args.out_root

    http = HttpFetcher()
    git = GitRepoFetcher()

    try:
        if cfg.http:
            http.fetch_many(cfg.http, out_root)
        if cfg.git:
            for src in cfg.git:
                git.fetch_one(src, out_root)
    finally:
        http.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
