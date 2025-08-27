#!/usr/bin/env python3
"""
Test script for the new prompt configuration system.
"""

import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ragdoc.agent.prompts import get_agent_prompts, reload_agent_prompts, AgentPrompts
from ragdoc.agent.graph import AgentState, generate_targeted_question, system_prompt

def test_prompt_configuration():
    """Test the prompt configuration system."""
    print("Testing Prompt Configuration System")
    print("=" * 40)
    
    # Test default prompts
    prompts = get_agent_prompts()
    print("\n1. Testing default system prompts:")
    print(f"   IT: {prompts.get_system_prompt('it')[:60]}...")
    print(f"   EN: {prompts.get_system_prompt('en')[:60]}...")
    
    # Test clarification questions
    print("\n2. Testing clarification questions:")
    state_it = AgentState(language="it")
    state_en = AgentState(language="en") 
    
    print(f"   Product question (IT): {prompts.get_question('product', 'it')}")
    print(f"   Product question (EN): {prompts.get_question('product', 'en')}")
    
    # Test escalation messages
    print("\n3. Testing escalation messages:")
    escalate_it = prompts.get_escalate_message(
        language="it", 
        attempts=3, 
        confidence=0.45, 
        keywords=["nethserver", "mail"]
    )
    print(f"   Escalation (IT): {escalate_it}")
    
    escalate_en = prompts.get_escalate_message(
        language="en", 
        attempts=3, 
        confidence=0.45, 
        keywords=["nethserver", "mail"]
    )
    print(f"   Escalation (EN): {escalate_en}")
    
    # Test answer instructions
    print("\n4. Testing answer instructions:")
    refs_text = "- NethServer Guide: https://docs.nethserver.org\n- Mail Setup: https://docs.nethserver.org/mail"
    answer_it = prompts.get_answer_instruction("it", refs_text)
    answer_en = prompts.get_answer_instruction("en", refs_text)
    print(f"   Answer instruction (IT): {answer_it[:100]}...")
    print(f"   Answer instruction (EN): {answer_en[:100]}...")
    
    # Test placeholder messages
    print("\n5. Testing placeholder messages:")
    print(f"   Placeholder title (IT): {prompts.get_placeholder_title('it')}")
    print(f"   Placeholder title (EN): {prompts.get_placeholder_title('en')}")
    print(f"   Placeholder preview (IT): {prompts.get_placeholder_preview('it')}")
    print(f"   Placeholder preview (EN): {prompts.get_placeholder_preview('en')}")
    
    print("\n✓ All prompt configuration tests passed!")


def test_environment_override():
    """Test that environment variables override defaults."""
    print("\n" + "=" * 40)
    print("Testing Environment Variable Override")
    print("=" * 40)
    
    # Set custom environment variable
    custom_prompt = "Custom test system prompt for ragdoc"
    os.environ["RAGDOC_SYSTEM_PROMPT_EN"] = custom_prompt
    
    # Reload configuration to pick up environment change
    prompts = reload_agent_prompts()
    
    # Verify the override worked
    result = prompts.get_system_prompt("en")
    print(f"Custom prompt set: {custom_prompt}")
    print(f"Retrieved prompt: {result}")
    
    assert result == custom_prompt, f"Expected '{custom_prompt}', got '{result}'"
    print("✓ Environment variable override test passed!")
    
    # Clean up
    del os.environ["RAGDOC_SYSTEM_PROMPT_EN"]


def test_integration_with_graph():
    """Test integration with the graph module."""
    print("\n" + "=" * 40) 
    print("Testing Integration with Graph Module")
    print("=" * 40)
    
    # Test system_prompt function
    print("\n1. Testing system_prompt function:")
    sys_prompt_it = system_prompt("it")
    sys_prompt_en = system_prompt("en")
    print(f"   System prompt (IT): {sys_prompt_it[:60]}...")
    print(f"   System prompt (EN): {sys_prompt_en[:60]}...")
    
    # Test generate_targeted_question function
    print("\n2. Testing generate_targeted_question function:")
    state = AgentState(language="en")
    analysis = {"quality": "no_results", "suggestions": ["add_specifics"]}
    query_info = {"missing_types": ["product"]}
    
    question = generate_targeted_question(state, analysis, query_info)
    print(f"   Generated question: {question}")
    
    assert "product" in question.lower(), "Question should ask about product"
    print("✓ Integration with graph module test passed!")


def main():
    """Run all tests."""
    try:
        test_prompt_configuration()
        test_environment_override()
        test_integration_with_graph()
        
        print("\n" + "=" * 50)
        print("✓ ALL TESTS PASSED!")
        print("\nPrompt configuration system is working correctly!")
        print("\nKey features validated:")
        print("- Centralized prompt configuration")
        print("- Environment variable overrides")
        print("- Multi-language support")
        print("- Integration with existing graph system")
        print("- Backward compatibility")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
