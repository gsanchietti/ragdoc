#!/usr/bin/env python3
"""
Simple test script for the new RecursiveUrlLoader-based HTTP fetcher
"""

from pathlib import Path
from ragdoc.fetch.http_fetcher import HttpFetcher
from ragdoc.fetch.models import HttpSource

def test_recursive_fetcher():
    # Create a test source (using a simple site that should work)
    source = HttpSource(
        url="https://httpbin.org/",
        out_dir="test_output",
        max_depth=1,  # Keep it small for testing
        timeout=10
    )
    
    fetcher = HttpFetcher()
    output_root = Path(".")
    
    try:
        print(f"Testing recursive fetch from: {source.url}")
        results = fetcher.fetch_one(source, output_root)
        
        print(f"✅ Fetch completed successfully!")
        print(f"📄 Downloaded {len(results)} pages:")
        
        for file_path, url in results[:5]:  # Show first 5 results
            print(f"  - {file_path} ({url})")
            if file_path.exists():
                size = file_path.stat().st_size
                print(f"    Size: {size} bytes")
        
        if len(results) > 5:
            print(f"  ... and {len(results) - 5} more pages")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        fetcher.close()

if __name__ == "__main__":
    print("🧪 Testing RecursiveUrlLoader-based HTTP fetcher...")
    success = test_recursive_fetcher()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")
