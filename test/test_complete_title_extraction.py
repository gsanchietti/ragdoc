#!/usr/bin/env python3

"""Test script to validate comprehensive title extraction functionality."""

from src.ragdoc.fetch.text_extraction import (
    html_to_text_with_title, 
    extract_title_from_markdown,
    read_markdown_with_title
)
from src.ragdoc.fetch.http_fetcher import extract_title_from_html
from tempfile import NamedTemporaryFile
import os

def test_html_title_extraction():
    """Test HTML title extraction."""
    print("Testing HTML title extraction...")
    
    test_html = """<!DOCTYPE html>
<html>
<head>
    <title>My Awesome Page</title>
</head>
<body>
    <h1>Welcome</h1>
    <p>This is content.</p>
</body>
</html>"""
    
    # Test html_to_text_with_title
    text, title = html_to_text_with_title(test_html.encode('utf-8'))
    assert title == "My Awesome Page", f"Expected 'My Awesome Page', got {repr(title)}"
    assert "Welcome" in text, "Content extraction failed"
    
    # Test extract_title_from_html
    title_direct = extract_title_from_html(test_html)
    assert title_direct == "My Awesome Page", f"Direct extraction failed: {repr(title_direct)}"
    
    print("✓ HTML title extraction passed")

def test_markdown_title_extraction():
    """Test Markdown title extraction."""
    print("Testing Markdown title extraction...")
    
    # Test YAML front matter
    markdown_yaml = """---
title: "My Markdown Document"
date: 2023-01-01
---

# Introduction

This is my document content.
"""
    
    title = extract_title_from_markdown(markdown_yaml)
    assert title == "My Markdown Document", f"YAML front matter failed: {repr(title)}"
    
    # Test heading-based title
    markdown_heading = """# Main Document Title

This is the content of the document.

## Section 2

More content here.
"""
    
    title = extract_title_from_markdown(markdown_heading)
    assert title == "Main Document Title", f"Heading extraction failed: {repr(title)}"
    
    # Test file-based reading
    with NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(markdown_yaml)
        temp_path = f.name
    
    try:
        from pathlib import Path
        content, title = read_markdown_with_title(Path(temp_path))
        assert title == "My Markdown Document", f"File reading failed: {repr(title)}"
        assert "Introduction" in content, "Content reading failed"
    finally:
        os.unlink(temp_path)
    
    print("✓ Markdown title extraction passed")

def test_edge_cases():
    """Test edge cases for title extraction."""
    print("Testing edge cases...")
    
    # HTML with no title
    html_no_title = "<html><body><p>No title here</p></body></html>"
    title = extract_title_from_html(html_no_title)
    assert title is None, f"Expected None for no title, got {repr(title)}"
    
    # Markdown with no title
    markdown_no_title = "Just some content without any title structure."
    title = extract_title_from_markdown(markdown_no_title)
    assert title is None, f"Expected None for no title, got {repr(title)}"
    
    # HTML with h1 fallback
    html_h1 = "<html><body><h1>H1 Title</h1><p>Content</p></body></html>"
    title = extract_title_from_html(html_h1)
    assert title == "H1 Title", f"H1 fallback failed: {repr(title)}"
    
    # HTML with h2 fallback
    html_h2 = "<html><body><h2>H2 Title</h2><p>Content</p></body></html>"
    title = extract_title_from_html(html_h2)
    assert title == "H2 Title", f"H2 fallback failed: {repr(title)}"
    
    print("✓ Edge cases passed")

def main():
    """Run all tests."""
    print("🔧 Testing title extraction functionality...\n")
    
    test_html_title_extraction()
    test_markdown_title_extraction()
    test_edge_cases()
    
    print("\n✅ All title extraction tests passed!")
    print("\n📋 Summary of fixes:")
    print("  • HTTP fetcher now preserves raw HTML for title extraction")
    print("  • Title extraction supports <title>, <h1>, and <h2> fallbacks")
    print("  • Markdown title extraction supports YAML front matter and # headings")
    print("  • Both HTML and Markdown files now have title information in the database")

if __name__ == "__main__":
    main()
