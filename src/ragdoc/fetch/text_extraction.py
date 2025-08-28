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
        # Common sidebar classes and IDs - be specific to avoid removing main content
        '.sidebar', '.side-nav', '.side-menu', '.side-bar', '.left-nav', '.right-nav',
        '#sidebar', '#side-nav', '#side-menu', '#side-bar', '#left-nav', '#right-nav',
        
        # Specific navigation that's typically not main content
        '.navbar', '.nav-bar', '.header-nav', '.footer-nav', '.top-nav',
        '.wy-nav-side', '.wy-side-scroll', '.wy-side-nav-search', '.wy-menu',
        
        # Common content-filtering classes
        '.advertisement', '.ads', '.banner', '.promotion',
        '.related', '.related-posts', '.related-articles',
        '.breadcrumb', '.breadcrumbs', '.pagination',
        '.share', '.social', '.social-share', '.social-media',
        '.comments', '.comment-section', '.comment-form',
        '.site-footer', '.page-footer', '.footer',
        '.site-header', '.page-header',
        
        # Common utility classes
        '.skip-link', '.sr-only', '.screen-reader-text',
        '.hidden', '.invisible', '.collapse',
        
        # Documentation-specific elements that are typically navigation
        '.toc', '.table-of-contents', '.page-toc',
        '.edit-page', '.edit-link', '.last-modified',
        '.version-selector', '.lang-selector',
        '.search-form', '.search-box', '.search-input',
        
        # CMS and framework specific
        '.widget', '.module', '.block',
        '.drupal-sidebar', '.wp-sidebar',
        '.mkdocs-nav', '.sphinx-sidebar',
    ]
    
    # Remove specific navigation elements (but not all nav tags)
    nav_specific_selectors = [
        'nav.navbar', 'nav.nav-bar', 'nav.header-nav', 'nav.footer-nav',
        'nav.wy-nav-top', 'nav.breadcrumb', 'nav.pagination'
    ]
    
    # Remove aside elements (typically sidebars)
    for element in soup.find_all('aside'):
        element.decompose()
    
    # Remove specific navigation elements
    for selector in nav_specific_selectors:
        try:
            for element in soup.select(selector):
                element.decompose()
        except Exception:
            continue
    
    # Remove elements by selector
    for selector in sidebar_selectors:
        try:
            for element in soup.select(selector):
                element.decompose()
        except Exception:
            # Ignore CSS selector parsing errors
            continue
    
    # Remove elements with role attributes that indicate navigation/supplementary content
    nav_roles = ['navigation', 'banner', 'complementary', 'contentinfo']
    for role in nav_roles:
        for element in soup.find_all(attrs={'role': role}):
            element.decompose()
    
    # Remove elements that are likely to be sidebars based on their position/structure
    # Be more selective here - only remove divs/sections with clear sidebar indicators
    for element in soup.find_all(['div', 'section']):
        # Skip if element doesn't have the expected methods/attributes
        if not hasattr(element, 'get') or not hasattr(element, 'decompose'):
            continue
            
        # Check if element has sidebar-like characteristics
        # Handle cases where element.attrs might be None or missing
        try:
            classes = element.get('class', []) if hasattr(element, 'attrs') and element.attrs else []
            element_id = element.get('id', '') if hasattr(element, 'attrs') and element.attrs else ''
        except (AttributeError, TypeError):
            # Skip elements that don't have proper attributes
            continue
        
        # Ensure we have valid data types
        if not isinstance(classes, (list, str)):
            classes = []
        if not isinstance(element_id, str):
            element_id = ''
        
        if isinstance(classes, list):
            class_str = ' '.join(str(c) for c in classes).lower()
        else:
            class_str = str(classes).lower()
        
        # Be more specific about sidebar indicators to avoid removing main content
        # Only remove if it's clearly a sidebar/navigation element
        sidebar_indicators = [
            'sidebar', 'side-nav', 'side-menu', 'side-bar',
            'widget', 'advertisement', 'ads', 'banner',
            'breadcrumb', 'pagination', 'related-posts', 'related-articles',
            'social-share', 'social-media', 'comment-section', 'comment-form'
        ]
        
        try:
            # Check for strong sidebar indicators
            is_sidebar = any(indicator in class_str for indicator in sidebar_indicators) or \
                        any(indicator in element_id.lower() for indicator in sidebar_indicators)
            
            # Additional check: if it contains "nav" or "menu" but not "content" or "main"
            has_nav_indicator = ('nav' in class_str or 'menu' in class_str) and \
                              ('content' not in class_str and 'main' not in class_str and 'document' not in class_str)
            
            if is_sidebar or has_nav_indicator:
                element.decompose()
        except (AttributeError, TypeError):
            # Skip if we can't process the element safely
            continue


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
