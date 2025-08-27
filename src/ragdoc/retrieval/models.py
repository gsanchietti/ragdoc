from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievalResult:
    id: str
    source_type: str
    source_url: str
    path: str
    title: str | None
    vscore: float
    lscore: float
    score: float
    content_preview: str
    metadata: dict[str, Any] | None
    sscore: float = 0.0  # BM25 sparse vector score
