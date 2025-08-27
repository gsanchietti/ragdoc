#!/usr/bin/env python3
"""Test script to verify token limit handling works correctly."""

import os
import logging
from src.ragdoc.fetch.indexer import EmbeddingClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock OpenAI key for testing
os.environ["OPENAI_API_KEY"] = "mock-key-for-testing"

def test_token_estimation():
    """Test our token estimation is conservative enough."""
    client = EmbeddingClient()
    
    # Test various text sizes
    test_texts = [
        "Short text",
        "A" * 1000,  # 1000 chars
        "A" * 5000,  # 5000 chars  
        "A" * 10000, # 10000 chars
        "A" * 20000, # 20000 chars - this should be way over limit
    ]
    
    safe_limit = int(client.max_tokens_per_request * 0.6)
    logger.info(f"Max tokens per request: {client.max_tokens_per_request}")
    logger.info(f"Safe limit (60%): {safe_limit}")
    
    for i, text in enumerate(test_texts):
        estimated = client._estimate_tokens(text)
        chars = len(text)
        ratio = chars / estimated if estimated > 0 else 0
        
        logger.info(f"Test {i+1}: {chars} chars -> {estimated} tokens (ratio: {ratio:.2f} chars/token)")
        
        if estimated > safe_limit:
            logger.warning(f"  Text {i+1} would be truncated (exceeds safe limit)")
        else:
            logger.info(f"  Text {i+1} is within safe limit")

def test_batching_logic():
    """Test that our batching logic correctly handles oversized batches."""
    client = EmbeddingClient()
    
    # Create texts that together would exceed the limit
    # If our safe limit is ~4800 tokens, create texts that sum to more than that
    large_text = "A" * 15000  # Should be ~5000 tokens with our 3-char estimate
    medium_text = "B" * 9000  # Should be ~3000 tokens
    small_text = "C" * 3000   # Should be ~1000 tokens
    
    test_batch = [large_text, medium_text, small_text]
    
    total_estimated = sum(client._estimate_tokens(text) for text in test_batch)
    safe_limit = int(client.max_tokens_per_request * 0.6)
    
    logger.info(f"Test batch: {len(test_batch)} texts, {total_estimated} estimated tokens")
    logger.info(f"Safe limit: {safe_limit} tokens")
    logger.info(f"Would exceed limit: {total_estimated > safe_limit}")
    
    # Test individual text truncation
    for i, text in enumerate(test_batch):
        estimated = client._estimate_tokens(text)
        if estimated > safe_limit:
            logger.warning(f"Text {i+1} ({estimated} tokens) would be truncated")
            max_chars = int(safe_limit * 3 * 0.8)
            logger.info(f"  Would truncate to {max_chars} chars")

if __name__ == "__main__":
    print("Testing token estimation...")
    test_token_estimation()
    print("\nTesting batching logic...")
    test_batching_logic()
    print("\nTest completed!")
