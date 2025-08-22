from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import httpx

from .models import HttpSource
from .state import get_http_meta, set_http_meta
from .utils import ensure_dir, safe_name

logger = logging.getLogger(__name__)


class HttpFetcher:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(follow_redirects=True, timeout=30.0)

    def close(self) -> None:
        self._client.close()

    def fetch_many(self, sources: Iterable[HttpSource], out_root: Path) -> list[Path]:
        outputs: list[Path] = []
        for src in sources:
            outputs.extend(self.fetch_one(src, out_root))
        return outputs

    def fetch_one(self, src: HttpSource, out_root: Path) -> list[Path]:
        from .state import load_state, save_state

        state = load_state()
        etag, last_mod = get_http_meta(state, src.url)

        headers = {}
        if etag:
            headers["If-None-Match"] = etag
        if last_mod:
            headers["If-Modified-Since"] = last_mod

        logger.info("HTTP GET %s", src.url)
        resp = self._client.get(src.url, headers=headers)
        if resp.status_code == 304:
            logger.info("Not modified: %s", src.url)
            return []
        resp.raise_for_status()

        # Update cache metadata if present
        new_etag = resp.headers.get("ETag")
        new_last_mod = resp.headers.get("Last-Modified")

        # Decide output file path
        out_dir = out_root / src.out_dir
        ensure_dir(out_dir)
        fname = safe_name(src.url)
        # Try to infer extension from content-type
        ctype = resp.headers.get("Content-Type", "").lower()
        ext = ".html" if "text/html" in ctype else (".md" if "markdown" in ctype or "text/plain" in ctype else "")
        out_file = out_dir / f"{fname}{ext}"

        out_file.write_bytes(resp.content)
        logger.info("Saved %s (%d bytes)", out_file, len(resp.content))

        set_http_meta(state, src.url, new_etag, new_last_mod)
        save_state(state)
        return [out_file]
