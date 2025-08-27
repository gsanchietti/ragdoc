#!/usr/bin/env python3

"""Test script to validate BM25 sparse vector search implementation."""

import os
# Set dummy API key for testing
os.environ["OPENAI_API_KEY"] = "dummy-key-for-testing"

from src.ragdoc.retrieval.retriever import Retriever, RetrieverConfig
from src.ragdoc.retrieval.models import RetrievalResult
import logging

def test_bm25_tokenization():
    """Test the tokenization function for BM25."""
    print("Testing BM25 tokenization...")
    
    config = RetrieverConfig(dsn="dummy://")
    retriever = Retriever(config)
    
    # Mock the OpenAI client to avoid API key requirement
    class MockOpenAI:
        def __init__(self):
            pass
    
    retriever.client = MockOpenAI()
    
    test_cases = [
        ("Hello world!", ["hello", "world"]),
        ("Python programming 2024", ["python", "programming", "2024"]),
        ("Natural Language Processing, AI & ML", ["natural", "language", "processing", "ai", "ml"]),
        ("test@example.com", ["test", "example", "com"]),
        ("BM25 is better than TF-IDF", ["bm25", "is", "better", "than", "tf", "idf"]),
    ]
    
    for text, expected in test_cases:
        tokens = retriever._tokenize(text)
        print(f"'{text}' -> {tokens}")
        assert tokens == expected, f"Expected {expected}, got {tokens}"
    
    print("✓ BM25 tokenization tests passed")

def test_bm25_computation():
    """Test BM25 score computation with mock documents."""
    print("Testing BM25 score computation...")
    
    config = RetrieverConfig(dsn="dummy://", bm25_k1=1.2, bm25_b=0.75)
    retriever = Retriever(config)
    
    # Mock the OpenAI client
    class MockOpenAI:
        def __init__(self):
            pass
    
    retriever.client = MockOpenAI()
    
    # Mock documents
    documents = [
        {
            'id': 'doc1',
            'title': 'Machine Learning Basics',
            'content': 'Machine learning is a subset of artificial intelligence. It involves training algorithms to learn patterns from data.'
        },
        {
            'id': 'doc2', 
            'title': 'Deep Learning Guide',
            'content': 'Deep learning uses neural networks with multiple layers. It is particularly effective for image and speech recognition.'
        },
        {
            'id': 'doc3',
            'title': 'Data Science Introduction',
            'content': 'Data science combines statistics, programming, and domain expertise to extract insights from data.'
        }
    ]
    
    # Test query
    query_tokens = ['machine', 'learning']
    
    scores = retriever._compute_bm25_scores(query_tokens, documents)
    
    print(f"BM25 scores for query {query_tokens}:")
    for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {doc_id}: {score:.4f}")
    
    # Validate that doc1 has the highest score (contains both 'machine' and 'learning')
    assert scores['doc1'] > scores['doc2'], "doc1 should score higher than doc2"
    assert scores['doc1'] > scores['doc3'], "doc1 should score higher than doc3"
    
    # Test with a different query
    query_tokens_2 = ['deep', 'neural']
    scores_2 = retriever._compute_bm25_scores(query_tokens_2, documents)
    
    print(f"BM25 scores for query {query_tokens_2}:")
    for doc_id, score in sorted(scores_2.items(), key=lambda x: x[1], reverse=True):
        print(f"  {doc_id}: {score:.4f}")
    
    # Validate that doc2 has the highest score for 'deep neural'
    assert scores_2['doc2'] > scores_2['doc1'], "doc2 should score higher than doc1 for 'deep neural'"
    
    print("✓ BM25 computation tests passed")

def test_configuration_parameters():
    """Test that BM25 configuration parameters are properly handled."""
    print("Testing BM25 configuration parameters...")
    
    # Test default configuration
    config_default = RetrieverConfig(dsn="dummy://")
    assert config_default.use_bm25 == True, "BM25 should be enabled by default"
    assert config_default.bm25_k1 == 1.2, f"Default k1 should be 1.2, got {config_default.bm25_k1}"
    assert config_default.bm25_b == 0.75, f"Default b should be 0.75, got {config_default.bm25_b}"
    assert config_default.sparse_weight == 0.3, f"Default sparse_weight should be 0.3, got {config_default.sparse_weight}"
    
    # Test custom configuration
    config_custom = RetrieverConfig(
        dsn="dummy://",
        use_bm25=False,
        bm25_k1=1.5,
        bm25_b=0.8,
        sparse_weight=0.4
    )
    assert config_custom.use_bm25 == False, "BM25 should be disabled when set to False"
    assert config_custom.bm25_k1 == 1.5, f"Custom k1 should be 1.5, got {config_custom.bm25_k1}"
    assert config_custom.bm25_b == 0.8, f"Custom b should be 0.8, got {config_custom.bm25_b}"
    assert config_custom.sparse_weight == 0.4, f"Custom sparse_weight should be 0.4, got {config_custom.sparse_weight}"
    
    print("✓ BM25 configuration tests passed")

def test_retrieval_result_model():
    """Test that RetrievalResult model includes sparse score."""
    print("Testing RetrievalResult model with sparse score...")
    
    result = RetrievalResult(
        id="test_id",
        source_type="md",
        source_url="http://example.com",
        path="/test/path.md",
        title="Test Document",
        vscore=0.8,
        lscore=0.6,
        sscore=0.9,  # Sparse score
        score=0.75,
        content_preview="This is a test document content...",
        metadata={"test": "metadata"}
    )
    
    assert hasattr(result, 'sscore'), "RetrievalResult should have sscore attribute"
    assert result.sscore == 0.9, f"Sparse score should be 0.9, got {result.sscore}"
    
    # Test default value
    result_default = RetrievalResult(
        id="test_id",
        source_type="md",
        source_url="",
        path="/test/path.md", 
        title="Test Document",
        vscore=0.8,
        lscore=0.6,
        score=0.75,
        content_preview="This is a test document content...",
        metadata=None
    )
    
    assert result_default.sscore == 0.0, f"Default sparse score should be 0.0, got {result_default.sscore}"
    
    print("✓ RetrievalResult model tests passed")

def main():
    """Run all BM25 tests."""
    print("🔧 Testing BM25 Sparse Vector Search Implementation...\n")
    
    # Configure logging to see debug output
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s - %(message)s")
    
    test_bm25_tokenization()
    test_bm25_computation()
    test_configuration_parameters()
    test_retrieval_result_model()
    
    print("\n✅ All BM25 sparse vector search tests passed!")
    print("\n📋 Summary of BM25 Implementation:")
    print("  • BM25 algorithm with configurable k1 and b parameters")
    print("  • Sparse vector generation based on TF-IDF with BM25 weighting")
    print("  • Three-way hybrid scoring: dense vectors + lexical + sparse vectors")
    print("  • Configurable sparse vector weight in hybrid search")
    print("  • Enhanced CLI with BM25 configuration options")
    print("  • Comprehensive tokenization and normalization")
    print("  • Support for title and content combined scoring")

if __name__ == "__main__":
    main()
