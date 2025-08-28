from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _load_env_config():
    """Load environment variables from .env file if available."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not available, environment variables must be set manually


# Load .env configuration at module import
_load_env_config()


@dataclass(frozen=True)
class HttpSource:
    url: str
    out_dir: str
    # Maximum depth for recursive crawling (default: 2)
    max_depth: int = 2
    # Custom link regex filter
    link_regex: Optional[str] = None
    # Directories to exclude
    exclude_dirs: tuple[str, ...] = ()
    # Request timeout
    timeout: int = 10


@dataclass(frozen=True)
class GitSource:
    repo: str
    ref: Optional[str] = None
    shallow: bool = True
    out_dir: str = "data/repos"


@dataclass(frozen=True)
class FetchConfig:
    http: list[HttpSource]
    git: list[GitSource]
