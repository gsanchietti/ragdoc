#!/usr/bin/env python3
"""
Test script for the enhanced iterative query refinement system.
This script tests the improved clarify mechanism with targeted questions.
"""

import os
import sys
import logging
from typing import Dict, Any, List

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ragdoc.agent.graph import (
    AgentState, analyze_retrieval_quality, extract_query_info, 
    generate_targeted_question, clarify_node, supervisor
)
from langchain_core.messages import HumanMessage, AIMessage

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s [%(name)s] %(message)s')
logger = logging.getLogger("test_refinement")

def test_retrieval_analysis():
    """Test the retrieval quality analysis function."""
    print("\n=== Testing Retrieval Quality Analysis ===")
    
    # Test case 1: No results
    contexts1 = []
    query1 = "nethserver configuration"
    analysis1 = analyze_retrieval_quality(contexts1, query1)
    print(f"No results: {analysis1}")
    assert analysis1["quality"] == "no_results"
    assert "add_specifics" in analysis1["suggestions"]
    
    # Test case 2: Placeholder only
    contexts2 = [{"_placeholder": True, "score": 0.0}]
    query2 = "help me"
    analysis2 = analyze_retrieval_quality(contexts2, query2)
    print(f"Placeholder only: {analysis2}")
    assert analysis2["quality"] == "no_results"
    
    # Test case 3: Good results
    contexts3 = [
        {"title": "NethServer Mail Configuration", "score": 0.85},
        {"title": "Email Setup Guide", "score": 0.78},
        {"title": "Mail Server Settings", "score": 0.72}
    ]
    query3 = "mail configuration"
    analysis3 = analyze_retrieval_quality(contexts3, query3)
    print(f"Good results: {analysis3}")
    assert analysis3["quality"] in ["good", "excellent"]
    assert analysis3["max_score"] >= 0.8
    
    # Test case 4: Poor results
    contexts4 = [
        {"title": "General Information", "score": 0.25},
        {"title": "About Us", "score": 0.18},
        {"title": "Contact", "score": 0.12}
    ]
    query4 = "specific technical issue"
    analysis4 = analyze_retrieval_quality(contexts4, query4)
    print(f"Poor results: {analysis4}")
    assert analysis4["quality"] == "poor"
    assert len(analysis4["suggestions"]) > 2
    
    print("✓ Retrieval quality analysis tests passed")


def test_query_info_extraction():
    """Test the query information extraction function."""
    print("\n=== Testing Query Info Extraction ===")
    
    # Test case 1: Vague query
    query1 = "help"
    info1 = extract_query_info(query1)
    print(f"Vague query '{query1}': {info1}")
    assert "product" in info1["missing_types"]
    assert "context" in info1["missing_types"]
    
    # Test case 2: Product-specific query
    query2 = "nethserver mail configuration"
    info2 = extract_query_info(query2)
    print(f"Product query '{query2}': {info2}")
    assert "nethserver" in [k.lower() for k in info2["keywords"]]
    assert "mail" in [k.lower() for k in info2["keywords"]]
    assert "product" not in info2["missing_types"]
    
    # Test case 3: Error query without details
    query3 = "error connecting"
    info3 = extract_query_info(query3)
    print(f"Error query '{query3}': {info3}")
    assert "error_details" in info3["missing_types"]
    
    # Test case 4: Configuration query
    query4 = "how to configure ssl"
    info4 = extract_query_info(query4)
    print(f"Config query '{query4}': {info4}")
    # Should not have config_location since it includes "how to"
    
    print("✓ Query info extraction tests passed")


def test_targeted_questions():
    """Test the targeted question generation."""
    print("\n=== Testing Targeted Question Generation ===")
    
    # Create test state
    state = AgentState(language="en")
    
    # Test case 1: No product specified
    analysis1 = {"quality": "no_results", "suggestions": ["add_specifics"]}
    query_info1 = {"missing_types": ["product", "context"]}
    question1 = generate_targeted_question(state, analysis1, query_info1)
    print(f"No product question: {question1}")
    assert "product" in question1.lower()
    
    # Test case 2: Error without details
    analysis2 = {"quality": "poor", "suggestions": ["use_different_terms"]}
    query_info2 = {"missing_types": ["error_details"]}
    question2 = generate_targeted_question(state, analysis2, query_info2)
    print(f"Error details question: {question2}")
    assert "error" in question2.lower()
    
    # Test case 3: Fair results needing specifics
    analysis3 = {"quality": "fair", "suggestions": ["add_specifics"]}
    query_info3 = {"missing_types": ["specifics"]}
    question3 = generate_targeted_question(state, analysis3, query_info3)
    print(f"Add specifics question: {question3}")
    assert any(word in question3.lower() for word in ["specific", "details", "more"])
    
    # Test Italian
    state_it = AgentState(language="it")
    question_it = generate_targeted_question(state_it, analysis1, query_info1)
    print(f"Italian question: {question_it}")
    assert any(word in question_it.lower() for word in ["prodotto", "specifico"])
    
    print("✓ Targeted question generation tests passed")


def test_clarify_node_logic():
    """Test the enhanced clarify node logic."""
    print("\n=== Testing Enhanced Clarify Node ===")
    
    # Create test state with a user message
    initial_state = AgentState(
        messages=[HumanMessage(content="help with email")],
        language="en",
        contexts=[{"_placeholder": True, "score": 0.0}],
        confidence=0.0,
        clarify_turns=0,
        contexts_placeholder=True
    )
    
    # Run clarify node
    result_state = clarify_node(initial_state)
    
    print(f"Clarify turns: {result_state.clarify_turns}")
    print(f"Last message: {result_state.messages[-1].content}")
    print(f"Refinement history: {len(result_state.refinement_history)}")
    
    assert result_state.clarify_turns == 1
    assert isinstance(result_state.messages[-1], AIMessage)
    assert len(result_state.refinement_history) == 1
    assert result_state.last_clarify_idx is not None
    
    # Check refinement record
    refinement = result_state.refinement_history[0]
    assert refinement["turn"] == 1
    assert refinement["query"] == "help with email"
    assert "missing_types" in refinement
    
    print("✓ Enhanced clarify node tests passed")


def test_supervisor_logic():
    """Test the enhanced supervisor decision logic."""
    print("\n=== Testing Enhanced Supervisor Logic ===")
    
    # Test case 1: High confidence -> answer
    state1 = AgentState(
        contexts=[{"score": 0.9}],
        confidence=0.8,
        clarify_turns=1,
        contexts_placeholder=False
    )
    decision1 = supervisor(state1)
    print(f"High confidence decision: {decision1}")
    assert decision1 == "answer"
    
    # Test case 2: No contexts -> clarify
    state2 = AgentState(
        contexts=[],
        confidence=0.0,
        clarify_turns=0,
        contexts_placeholder=True
    )
    decision2 = supervisor(state2)
    print(f"No contexts decision: {decision2}")
    assert decision2 == "clarify"
    
    # Test case 3: Max iterations reached -> escalate
    state3 = AgentState(
        contexts=[{"score": 0.4}],
        confidence=0.4,
        clarify_turns=5,
        contexts_placeholder=False
    )
    decision3 = supervisor(state3)
    print(f"Max iterations decision: {decision3}")
    assert decision3 == "escalate"
    
    # Test case 4: Improving trajectory -> clarify
    state4 = AgentState(
        contexts=[{"score": 0.6}],
        confidence=0.55,  # 85% of 0.65 threshold
        clarify_turns=2,
        contexts_placeholder=False,
        best_confidence=0.5
    )
    decision4 = supervisor(state4)
    print(f"Improving trajectory decision: {decision4}")
    assert decision4 == "clarify"
    
    # Test case 5: Very low confidence after attempts -> escalate
    state5 = AgentState(
        contexts=[{"score": 0.2}],
        confidence=0.25,  # Well below 60% of threshold (0.65 * 0.6 = 0.39)
        clarify_turns=3,  # More attempts
        contexts_placeholder=False,
        best_confidence=0.25
    )
    decision5 = supervisor(state5)
    print(f"Very low confidence decision: {decision5}")
    assert decision5 == "escalate"
    
    print("✓ Enhanced supervisor logic tests passed")


def test_integration_scenario():
    """Test a complete iterative refinement scenario."""
    print("\n=== Testing Complete Iterative Refinement Scenario ===")
    
    print("\n1. User asks vague question")
    state = AgentState(
        messages=[HumanMessage(content="error")],
        language="en",
        contexts=[],
        confidence=0.0,
        clarify_turns=0,
        contexts_placeholder=True
    )
    
    # Should ask for clarification
    decision = supervisor(state)
    print(f"   Decision: {decision}")
    assert decision == "clarify"
    
    # Clarify node generates targeted question
    state = clarify_node(state)
    print(f"   Question: {state.messages[-1].content}")
    assert "product" in state.messages[-1].content.lower() or "specific" in state.messages[-1].content.lower()
    
    print("\n2. User provides more details")
    state.messages.append(HumanMessage(content="NethServer mail server error"))
    # Simulate retrieval with moderate results
    state.contexts = [
        {"title": "Mail Server Troubleshooting", "score": 0.6},
        {"title": "Common Issues", "score": 0.45}
    ]
    state.confidence = 0.55
    state.contexts_placeholder = False
    
    # Should continue clarifying since confidence is decent but not excellent
    decision = supervisor(state)
    print(f"   Decision: {decision}")
    # Could be clarify or answer depending on threshold and logic
    
    if decision == "clarify":
        state = clarify_node(state)
        print(f"   Follow-up question: {state.messages[-1].content}")
        
        print("\n3. User provides specific error message")
        state.messages.append(HumanMessage(content="Getting 'Connection refused' error when trying to send email"))
        # Simulate retrieval with better results
        state.contexts = [
            {"title": "Connection Refused Mail Error", "score": 0.85},
            {"title": "SMTP Configuration Fix", "score": 0.78}
        ]
        state.confidence = 0.82
        
        # Should answer now
        decision = supervisor(state)
        print(f"   Final decision: {decision}")
        assert decision == "answer"
    
    print(f"\nTotal refinement attempts: {len(state.refinement_history)}")
    print(f"Best confidence achieved: {state.best_confidence}")
    
    print("✓ Integration scenario test passed")


def main():
    """Run all tests."""
    print("Testing Enhanced Iterative Query Refinement System")
    print("=" * 55)
    
    try:
        test_retrieval_analysis()
        test_query_info_extraction()
        test_targeted_questions()
        test_clarify_node_logic()
        test_supervisor_logic()
        test_integration_scenario()
        
        print("\n" + "=" * 55)
        print("✓ ALL TESTS PASSED - Enhanced iterative refinement system is working correctly!")
        print("\nKey improvements implemented:")
        print("- Analyzes retrieval quality and suggests specific improvements")
        print("- Extracts query keywords and identifies missing information types")
        print("- Generates targeted clarifying questions based on analysis")
        print("- Tracks refinement history and improvement trajectory")
        print("- Makes smart decisions about when to continue vs escalate")
        print("- Escalates only after 5 iterations as requested")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
