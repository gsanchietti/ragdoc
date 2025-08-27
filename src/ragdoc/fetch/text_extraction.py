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


def html_to_text_with_title(content: bytes) -> tuple[str, str | None]:
    """Extract readable text and title from HTML bytes using BeautifulSoup."""
    soup = BeautifulSoup(content, "html.parser")
    
    # Extract title
    title = None
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        title = title_tag.string.strip()
    
    # If no title tag, try to find the first h1
    if not title:
        h1_tag = soup.find("h1")
        if h1_tag:
            title = h1_tag.get_text(strip=True)
    
    # Try h2 as fallback
    if not title:
        h2_tag = soup.find("h2")
        if h2_tag:
            title = h2_tag.get_text(strip=True)
    
    # Remove script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return text, title if title else None


def read_markdown(path: Path, encoding: str = "utf-8") -> str:
    return path.read_text(encoding=encoding)


def extract_title_from_markdown(content: str) -> str | None:
    """Extract title from Markdown content."""
    lines = content.split('\n')
    
    # Check for YAML front matter title
    if lines and lines[0].strip() == '---':
        in_frontmatter = True
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == '---':
                break
            if line.strip().startswith('title:'):
                title = line.split('title:', 1)[1].strip()
                # Remove quotes if present
                title = title.strip('"\'')
                if title:
                    return title
    
    # Look for first heading (# Title)
    for line in lines:
        line = line.strip()
        if line.startswith('# '):
            title = line[2:].strip()
            if title:
                return title
    
    return None


def read_markdown_with_title(path: Path, encoding: str = "utf-8") -> tuple[str, str | None]:
    """Read markdown file and extract title."""
    content = path.read_text(encoding=encoding)
    title = extract_title_from_markdown(content)
    return content, title


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
