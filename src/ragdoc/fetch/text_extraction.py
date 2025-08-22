from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup


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
