#!/usr/bin/env python3

"""Test script to validate title extraction from HTML."""

from src.ragdoc.fetch.text_extraction import html_to_text_with_title
from src.ragdoc.fetch.http_fetcher import extract_title_from_html

# Test HTML content with different title scenarios
test_cases = [
    {
        "name": "HTML with title tag",
        "html": b"""<!DOCTYPE html>
<html>
<head>
    <title>Test Page Title</title>
</head>
<body>
    <h1>Main Heading</h1>
    <p>Some content here.</p>
</body>
</html>""",
        "expected_title": "Test Page Title"
    },
    {
        "name": "HTML with only h1 (no title tag)",
        "html": b"""<!DOCTYPE html>
<html>
<head>
</head>
<body>
    <h1>Main Heading as Title</h1>
    <p>Some content here.</p>
</body>
</html>""",
        "expected_title": "Main Heading as Title"
    },
    {
        "name": "HTML with empty title tag (should fallback to h1)",
        "html": b"""<!DOCTYPE html>
<html>
<head>
    <title></title>
</head>
<body>
    <h1>H1 Fallback Title</h1>
    <p>Some content here.</p>
</body>
</html>""",
        "expected_title": "H1 Fallback Title"
    },
    {
        "name": "HTML with no title or h1",
        "html": b"""<!DOCTYPE html>
<html>
<head>
</head>
<body>
    <p>Some content here with no title.</p>
</body>
</html>""",
        "expected_title": None
    }
]

def test_title_extraction():
    print("Testing title extraction...")
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        
        # Test html_to_text_with_title function
        text, title = html_to_text_with_title(test_case['html'])
        print(f"Extracted title: {repr(title)}")
        print(f"Expected title: {repr(test_case['expected_title'])}")
        print(f"Text preview: {text[:100]}...")
        
        # Test extract_title_from_html function
        html_str = test_case['html'].decode('utf-8')
        title_direct = extract_title_from_html(html_str)
        print(f"Direct extraction: {repr(title_direct)}")
        
        # Validate
        assert title == test_case['expected_title'], f"Title mismatch: got {repr(title)}, expected {repr(test_case['expected_title'])}"
        assert title_direct == test_case['expected_title'], f"Direct extraction mismatch: got {repr(title_direct)}, expected {repr(test_case['expected_title'])}"
        
        print("✓ Test passed")
    
    print("\n✅ All title extraction tests passed!")

if __name__ == "__main__":
    test_title_extraction()
