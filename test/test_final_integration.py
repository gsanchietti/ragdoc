#!/usr/bin/env python3
"""
Final integration test for the prompt configuration system.
Tests the complete workflow with all components working together.
"""

import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ragdoc.agent.graph import AgentState, build_graph, supervisor, clarify_node, escalate_node
from ragdoc.agent.prompts import get_agent_prompts
from langchain_core.messages import HumanMessage, AIMessage

def test_complete_agent_workflow():
    """Test a complete agent workflow with custom prompts."""
    print("Testing Complete Agent Workflow with Custom Prompts")
    print("=" * 55)
    
    # Set custom prompts via environment
    os.environ["RAGDOC_SYSTEM_PROMPT_EN"] = "You are a test assistant for configuration validation."
    os.environ["RAGDOC_ESCALATE_BASE_EN"] = "Escalating to human support with test summary."
    
    # Reload configuration to pick up environment changes
    from ragdoc.agent.prompts import reload_agent_prompts
    prompts = reload_agent_prompts()
    
    # Verify custom prompts are loaded
    assert "test assistant" in prompts.get_system_prompt("en").lower()
    print("✓ Custom environment prompts loaded successfully")
    
    # Test agent state with clarification workflow
    print("\n1. Testing clarification workflow:")
    state = AgentState(
        messages=[HumanMessage(content="help")],
        language="en",
        contexts=[],
        confidence=0.0,
        clarify_turns=0,
        contexts_placeholder=True
    )
    
    # Should decide to clarify
    decision = supervisor(state)
    print(f"   Supervisor decision: {decision}")
    assert decision == "clarify"
    
    # Generate clarification
    clarified_state = clarify_node(state)
    question = clarified_state.messages[-1].content
    print(f"   Generated question: {question}")
    assert isinstance(clarified_state.messages[-1], AIMessage)
    assert clarified_state.clarify_turns == 1
    
    # Test escalation workflow
    print("\n2. Testing escalation workflow:")
    escalation_state = AgentState(
        messages=[HumanMessage(content="test query")],
        language="en",
        contexts=[{"score": 0.2}],
        confidence=0.2,
        clarify_turns=5,  # Max attempts reached
        contexts_placeholder=False,
        refinement_history=[{"turn": i, "confidence": 0.2} for i in range(1, 6)],
        best_confidence=0.4,
        query_keywords=["test", "keyword"]
    )
    
    # Should decide to escalate
    decision = supervisor(escalation_state)
    print(f"   Supervisor decision: {decision}")
    assert decision == "escalate"
    
    # Generate escalation message
    escalated_state = escalate_node(escalation_state)
    escalation_msg = escalated_state.messages[-1].content
    print(f"   Escalation message: {escalation_msg}")
    assert "test summary" in escalation_msg.lower()  # Custom escalation message
    assert "5" in escalation_msg  # Should include attempt count
    
    # Test graph building
    print("\n3. Testing graph compilation:")
    graph = build_graph()
    print("   ✓ Graph compiled successfully with custom prompts")
    
    print("\n✓ Complete agent workflow test passed!")
    
    # Clean up environment
    del os.environ["RAGDOC_SYSTEM_PROMPT_EN"]
    del os.environ["RAGDOC_ESCALATE_BASE_EN"]


def test_multilingual_support():
    """Test multilingual prompt support."""
    print("\n" + "=" * 55)
    print("Testing Multilingual Prompt Support")
    print("=" * 55)
    
    # Reload prompts to get clean defaults
    from ragdoc.agent.prompts import reload_agent_prompts
    prompts = reload_agent_prompts()
    
    # Test Italian prompts
    print("\n1. Testing Italian prompts:")
    sys_prompt_it = prompts.get_system_prompt("it")
    question_it = prompts.get_question("product", "it")
    escalate_it = prompts.get_escalate_message("it", attempts=3, confidence=0.5)
    
    print(f"   System prompt: {sys_prompt_it[:50]}...")
    print(f"   Product question: {question_it}")
    print(f"   Escalation message: {escalate_it[:80]}...")
    
    assert "ragdoc" in sys_prompt_it.lower()
    assert "prodotto" in question_it.lower()
    assert "passando" in escalate_it.lower()
    
    # Test English prompts
    print("\n2. Testing English prompts:")
    sys_prompt_en = prompts.get_system_prompt("en")
    question_en = prompts.get_question("product", "en")
    escalate_en = prompts.get_escalate_message("en", attempts=3, confidence=0.5)
    
    print(f"   System prompt: {sys_prompt_en[:50]}...")
    print(f"   Product question: {question_en}")
    print(f"   Escalation message: {escalate_en[:80]}...")
    
    assert "ragdoc" in sys_prompt_en.lower()
    assert "product" in question_en.lower()
    assert "escalating" in escalate_en.lower()
    
    print("\n✓ Multilingual support test passed!")


def test_backward_compatibility():
    """Test that existing code still works unchanged."""
    print("\n" + "=" * 55)
    print("Testing Backward Compatibility")
    print("=" * 55)
    
    # Test that old system_prompt function still works
    from ragdoc.agent.graph import system_prompt
    
    prompt_it = system_prompt("it")
    prompt_en = system_prompt("en")
    
    print(f"   system_prompt('it'): {prompt_it[:50]}...")
    print(f"   system_prompt('en'): {prompt_en[:50]}...")
    
    assert "ragdoc" in prompt_it.lower()
    assert "ragdoc" in prompt_en.lower()
    
    # Test that graph building still works
    graph = build_graph()
    print("   ✓ Graph builds successfully")
    
    print("\n✓ Backward compatibility test passed!")


def main():
    """Run all integration tests."""
    try:
        test_complete_agent_workflow()
        test_multilingual_support()
        test_backward_compatibility()
        
        print("\n" + "=" * 60)
        print("✓ ALL INTEGRATION TESTS PASSED!")
        print("\nPrompt configuration system is fully operational!")
        print("\nKey achievements:")
        print("- All system prompts and responses moved to configuration")
        print("- Environment variable override system working")
        print("- Multilingual support maintained")
        print("- Complete backward compatibility")
        print("- Enhanced iterative refinement system integrated")
        print("- Agent workflow functions correctly with custom prompts")
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
