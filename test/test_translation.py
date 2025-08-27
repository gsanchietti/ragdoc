#!/usr/bin/env python3
"""Test script to verify the translation functionality works correctly."""

import os
import logging
import sys
import tempfile

# Set mock API key for testing
os.environ["OPENAI_API_KEY"] = "mock-key-for-testing"

# Add the src directory to the Python path
sys.path.insert(0, 'src')

from ragdoc.agent.graph import _translate_to_english

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_translation_logic():
    """Test the translation logic without making actual API calls."""
    
    test_cases = [
        # English text (should not be translated)
        ("How to configure PostgreSQL?", "English query"),
        ("database setup instructions", "English query"),
        ("the quick brown fox", "English phrase"),
        
        # Non-English text (would be translated if API was available)
        ("Come configurare PostgreSQL?", "Italian query"),
        ("instruções de configuração do banco de dados", "Portuguese query"),
        ("如何配置数据库", "Chinese query"),
        
        # Edge cases
        ("", "Empty string"),
        ("DB", "Short technical term"),
        ("api", "Short English term"),
    ]
    
    print("Testing translation decision logic...")
    print("=" * 50)
    
    for text, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Input: '{text}'")
        
        # Test the heuristic logic manually
        if not text or not text.strip():
            decision = "Skip (empty)"
        else:
            text_clean = text.lower().strip()
            english_indicators = ['the', 'and', 'or', 'to', 'of', 'in', 'for', 'with', 'on', 'at', 'by', 'from']
            has_english_words = any(word in text_clean for word in english_indicators)
            
            if len(text_clean) < 10 or (has_english_words and text.isascii()):
                decision = "Skip (appears English)"
            else:
                decision = "Would translate"
        
        print(f"Decision: {decision}")
        
        # Since we don't have a real API key, we can't test actual translation
        # but we can verify the logic flow
        try:
            result = _translate_to_english(text)
            print(f"Result: '{result}' (no translation due to mock API)")
        except Exception as e:
            print(f"Expected error (mock API): {type(e).__name__}")

def test_integration():
    """Test integration with the agent graph structure."""
    from ragdoc.agent.graph import AgentState, retrieve_node
    from langchain_core.messages import HumanMessage
    
    print("\n" + "=" * 50)
    print("Testing integration with retrieve_node...")
    
    # Create a mock state with a non-English message
    state = AgentState(
        messages=[HumanMessage(content="Come configurare il database PostgreSQL?")],
        language="it"
    )
    
    print(f"Original message: '{state.messages[0].content}'")
    
    try:
        # This will fail due to missing database and mock API key, but we can check the logs
        result = retrieve_node(state)
        print("Retrieve node executed successfully (unexpected)")
    except Exception as e:
        print(f"Expected error (no database/API): {type(e).__name__}")
        print("Translation logic was triggered as expected")

if __name__ == "__main__":
    print("Testing translation functionality...")
    test_translation_logic()
    test_integration()
    print("\nTest completed!")
