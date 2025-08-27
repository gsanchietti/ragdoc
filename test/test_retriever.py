#!/usr/bin/env python3
"""Test script to verify the improved retriever works correctly."""

import os
import logging
from src.ragdoc.retrieval.retriever import Retriever, RetrieverConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set mock API key for testing
os.environ["OPENAI_API_KEY"] = "mock-key-for-testing"

def test_retriever_improvements():
    """Test the improved text search capabilities."""
    
    # Mock database URL for testing - you'd need a real DB with data for actual testing
    test_dsn = os.getenv("DATABASE_URL", "postgresql://ragdoc:ragdoc@127.0.0.1:5432/ragdoc")
    
    # Test different configurations
    configs = [
        RetrieverConfig(dsn=test_dsn, use_fts=True, title_boost=1.5, alpha=0.7),
        RetrieverConfig(dsn=test_dsn, use_fts=False, title_boost=2.0, alpha=0.5),  # Pattern matching only
        RetrieverConfig(dsn=test_dsn, use_fts=True, title_boost=1.0, alpha=0.3),  # More weight on text
    ]
    
    test_queries = [
        "PostgreSQL vector search",
        "how to configure database",
        "embedding model",
        "ragdoc installation"
    ]
    
    for i, config in enumerate(configs):
        print(f"\n=== Configuration {i+1} ===")
        print(f"FTS enabled: {config.use_fts}")
        print(f"Title boost: {config.title_boost}")
        print(f"Alpha (vector weight): {config.alpha}")
        
        retriever = Retriever(config)
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            try:
                # This would fail without actual database data, but tests the logic
                results = retriever.search(query, k=3)
                print(f"  Found {len(results)} results")
                for j, result in enumerate(results[:2]):
                    print(f"  {j+1}. Score: {result.score:.3f} (v:{result.vscore:.3f}, l:{result.lscore:.3f})")
                    print(f"     Title: {result.title}")
                    print(f"     Preview: {result.content_preview[:100]}...")
            except Exception as e:
                print(f"  Error (expected without DB): {type(e).__name__}")

def test_fallback_search():
    """Test the fallback pattern search method."""
    from unittest.mock import Mock, MagicMock
    
    # Create a mock retriever to test the fallback logic
    config = RetrieverConfig(dsn="mock://", use_fts=False)
    retriever = Retriever(config)
    
    # Mock connection and cursor
    mock_conn = Mock()
    mock_cursor = Mock()
    
    # Set up the context manager properly
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.__enter__ = Mock(return_value=mock_cursor)
    mock_cursor.__exit__ = Mock(return_value=None)
    
    # Mock query results
    mock_cursor.fetchall.return_value = [
        ("id1", 1.5),  # title match with boost
        ("id2", 0.8),  # content match
        ("id3", 0.3),  # frequency match
    ]
    
    scores = {}
    retriever._fallback_pattern_search(mock_conn, "test query", 5, scores)
    
    print(f"\nFallback search test:")
    print(f"Generated scores: {scores}")
    assert len(scores) == 3
    assert scores["id1"] == 1.5  # Should have title boost
    assert scores["id2"] == 0.8  # Content match
    assert scores["id3"] == 0.3  # Frequency match
    print("✓ Fallback search working correctly")

if __name__ == "__main__":
    print("Testing retriever improvements...")
    test_fallback_search()
    print("\nTesting with different configurations...")
    test_retriever_improvements()
    print("\nTest completed!")
