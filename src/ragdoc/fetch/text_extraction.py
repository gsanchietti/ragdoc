from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup
from docutils.core import publish_parts


def html_to_text(content: bytes) -> str:
    """Extract readable text from HTML bytes using BeautifulSoup."""
    soup = BeautifulSoup(content, "html.parser")
    # Remove script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return text


def read_markdown(path: Path, encoding: str = "utf-8") -> str:
    return path.read_text(encoding=encoding)


def chunk_text(text: str, max_chars: int = 8000) -> list[str]:
    """Simple paragraph-aware chunking by character count."""
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for para in text.split("\n\n"):
        p = para.strip()
        if not p:
            continue
        if size + len(p) + 2 > max_chars and buf:
            chunks.append("\n\n".join(buf))
            buf = []
            size = 0
        buf.append(p)
        size += len(p) + 2
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def read_text(path: Path, encoding: str = "utf-8") -> str:
    return path.read_text(encoding=encoding)


def rst_to_text(content: str) -> str:
    """Convert reStructuredText to plain text using docutils; fallback to raw on errors."""
    try:
        parts = publish_parts(source=content, writer_name="plaintext")
        return parts.get("whole", content)
    except Exception:
        return content


def read_rst(path: Path, encoding: str = "utf-8") -> str:
    return rst_to_text(path.read_text(encoding=encoding))


def read_by_suffix(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".md", ".markdown"}:
        return read_markdown(path)
    if ext in {".rst"}:
        return read_rst(path)
    # default to plain text
    return read_text(path)
