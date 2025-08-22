from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HttpSource:
    url: str
    out_dir: str


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
