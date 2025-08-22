from __future__ import annotations

import hashlib
from pathlib import Path


def safe_name(s: str, max_len: int = 64) -> str:
    """Generate a filesystem-safe directory/file name from an arbitrary string.

    Uses a short hash suffix to avoid collisions and keeps the last path/token
    visible when possible.
    """
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
    base = s.rstrip("/").split("/")[-1]
    base = base or "root"
    safe = "".join(c for c in base if c.isalnum() or c in ("-", "_", "."))
    candidate = f"{safe}-{h}"
    return candidate[:max_len]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
