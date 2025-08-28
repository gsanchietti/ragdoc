from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup
from docutils.core import publish_parts


def html_to_text(content: bytes) -> str:
    """Extract readable text from HTML bytes using BeautifulSoup, excluding sidebars and navigation."""
    soup = BeautifulSoup(content, "html.parser")
    
    # Remove common sidebar and navigation elements
    _remove_sidebar_elements(soup)
    
    # Remove script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return text


def html_to_text_with_title(content: bytes) -> tuple[str, str | None]:
    """Extract readable text and title from HTML bytes using BeautifulSoup, excluding sidebars and navigation."""
    soup = BeautifulSoup(content, "html.parser")
    
    # Extract title before removing elements
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
    
    # Remove common sidebar and navigation elements
    _remove_sidebar_elements(soup)
    
    # Remove script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return text, title if title else None


def _remove_sidebar_elements(soup: BeautifulSoup) -> None:
    """Remove common sidebar, navigation, and non-content elements from HTML soup."""
    # Remove elements by common class names and IDs
    sidebar_selectors = [
        # Common sidebar classes and IDs
        'aside', 'nav', '.sidebar', '.navigation', '.nav', '.menu',
        '#sidebar', '#navigation', '#nav', '#menu',
        '.side-nav', '.side-menu', '.side-bar', '.left-nav', '.right-nav',
        '.navbar', '.nav-bar', '.header-nav', '.footer-nav',
        
        # Common content-filtering classes
        '.advertisement', '.ads', '.banner', '.promotion',
        '.related', '.related-posts', '.related-articles',
        '.breadcrumb', '.breadcrumbs', '.pagination',
        '.share', '.social', '.social-share', '.social-media',
        '.comments', '.comment-section', '.comment-form',
        '.footer', '.site-footer', '.page-footer',
        '.header', '.site-header', '.page-header',
        
        # Common utility classes
        '.skip-link', '.sr-only', '.screen-reader-text',
        '.hidden', '.invisible', '.collapse',
        
        # Documentation-specific elements
        '.toc', '.table-of-contents', '.page-toc',
        '.edit-page', '.edit-link', '.last-modified',
        '.version-selector', '.lang-selector',
        '.search-form', '.search-box', '.search-input',
        
        # CMS and framework specific
        '.widget', '.module', '.block',
        '.drupal-sidebar', '.wp-sidebar',
        '.mkdocs-nav', '.sphinx-sidebar',
    ]
    
    # Remove elements by tag name
    for tag_name in ['aside', 'nav']:
        for element in soup.find_all(tag_name):
            element.decompose()
    
    # Remove elements by selector
    for selector in sidebar_selectors:
        try:
            for element in soup.select(selector):
                element.decompose()
        except Exception:
            # Ignore CSS selector parsing errors
            continue
    
    # Remove elements with role attributes that indicate navigation
    nav_roles = ['navigation', 'banner', 'complementary', 'contentinfo']
    for role in nav_roles:
        for element in soup.find_all(attrs={'role': role}):
            element.decompose()
    
    # Remove elements that are likely to be sidebars based on their position/structure
    # Find elements with common sidebar patterns
    for element in soup.find_all(['div', 'section']):
        # Check if element has sidebar-like characteristics
        classes = element.get('class', [])
        element_id = element.get('id', '')
        
        if isinstance(classes, list):
            class_str = ' '.join(classes).lower()
        else:
            class_str = str(classes).lower()
        
        # Check for sidebar indicators in class names or IDs
        sidebar_indicators = [
            'sidebar', 'nav', 'menu', 'aside', 'widget', 'toc',
            'breadcrumb', 'related', 'share', 'social', 'comment',
            'footer', 'header', 'banner', 'advertisement', 'ad'
        ]
        
        is_sidebar = any(indicator in class_str or indicator in element_id.lower() 
                        for indicator in sidebar_indicators)
        
        if is_sidebar:
            element.decompose()


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
