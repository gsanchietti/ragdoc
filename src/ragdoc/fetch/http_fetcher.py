from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup

from .models import HttpSource
from .utils import ensure_dir, safe_name

logger = logging.getLogger(__name__)


def bs4_extractor(html: str) -> str:
    """Extract clean text from HTML using BeautifulSoup."""
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def extract_title_from_html(html: str) -> str | None:
    """Extract title from HTML using BeautifulSoup."""
    try:
        soup = BeautifulSoup(html, "lxml")
        
        # Try to find title tag first
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            title = title_tag.string.strip()
            if title:
                return title
        
        # If no title tag, try to find the first h1
        h1_tag = soup.find("h1")
        if h1_tag:
            title = h1_tag.get_text(strip=True)
            if title:
                return title
        
        # Try h2 as fallback
        h2_tag = soup.find("h2")
        if h2_tag:
            title = h2_tag.get_text(strip=True)
            if title:
                return title
                
        return None
    except Exception:
        return None


class HttpFetcher:
    def __init__(self) -> None:
        # RecursiveUrlLoader manages its own HTTP client
        pass

    def close(self) -> None:
        # Nothing to close with RecursiveUrlLoader
        pass

    def fetch_many(self, sources: Iterable[HttpSource], out_root: Path) -> list[tuple[Path, str]]:
        outputs: list[tuple[Path, str]] = []
        for src in sources:
            outputs.extend(self.fetch_one(src, out_root))
        return outputs

    def fetch_one(self, src: HttpSource, out_root: Path) -> list[tuple[Path, str]]:
        try:
            from langchain_community.document_loaders import RecursiveUrlLoader
        except ImportError as e:
            logger.error("RecursiveUrlLoader requires langchain-community: %s", e)
            return []

        logger.info("Recursive crawl starting from: %s", src.url)
        
        # Set up the recursive loader with parameters from HttpSource
        loader = RecursiveUrlLoader(
            src.url,
            max_depth=getattr(src, "max_depth", 2),
            extractor=lambda x: x,  # Keep raw HTML for title extraction
            link_regex=getattr(src, "link_regex", None),
            exclude_dirs=getattr(src, "exclude_dirs", ()),
            timeout=getattr(src, "timeout", 10),
            check_response_status=False,
            continue_on_failure=True,
            prevent_outside=True,
        )

        # Load documents
        try:
            docs = loader.load()
        except Exception as e:
            logger.error("Failed to load documents from %s: %s", src.url, e)
            return []

        # Save documents to files
        outputs: list[tuple[Path, str]] = []
        out_dir = out_root / src.out_dir
        ensure_dir(out_dir)
        
        for doc in docs:
            url = doc.metadata.get("source", src.url)
            fname = safe_name(url)
            out_file = out_dir / f"{fname}.html"
            
            try:
                # Extract title and clean content from raw HTML
                raw_html = doc.page_content or ""
                title = extract_title_from_html(raw_html)
                clean_content = bs4_extractor(raw_html)
                
                # Save the original HTML content for proper processing later
                out_file.write_text(raw_html, encoding="utf-8")
                outputs.append((out_file, url))
                logger.debug("Saved %s (%d chars) with title: %s", out_file, len(clean_content), title)
            except Exception as ex:
                logger.warning("Failed to save %s: %s", url, ex)

        logger.info("Recursive crawl completed: %d pages from %s", len(outputs), src.url)
        return outputs
