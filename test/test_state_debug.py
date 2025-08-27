#!/usr/bin/env python3
"""
Test script to examine conversation state management without API calls.
"""

import os
import sys
sys.path.insert(0, 'src')

from langchain_core.messages import HumanMessage, AIMessage
from ragdoc.agent.graph import AgentState, answer_router

def test_answer_router_logic():
    """Test the answer router logic for conversation continuation."""
    print("🧪 Testing answer router logic...")
    
    # Test 1: State with assistant answer only - should wait
    state1 = AgentState(
        messages=[
            HumanMessage(content="How do I configure email?"),
            AIMessage(content="To configure email, follow these steps...")
        ],
        language="en",
        contexts=[],
        confidence=None,
        clarify_turns=0,
        contexts_placeholder=False,
        last_clarify_idx=None,
        refinement_history=[],
        best_confidence=0.0,
        query_keywords=[],
        missing_info_types=[]
    )
    
    result1 = answer_router(state1)
    print(f"Test 1 - Assistant answer only: {result1} (expected: wait)")
    
    # Test 2: State with human follow-up after assistant answer - should retrieve
    state2 = AgentState(
        messages=[
            HumanMessage(content="How do I configure email?"),
            AIMessage(content="To configure email, follow these steps..."),
            HumanMessage(content="What about SMTP settings?")
        ],
        language="en",
        contexts=[],
        confidence=None,
        clarify_turns=0,
        contexts_placeholder=False,
        last_clarify_idx=None,
        refinement_history=[],
        best_confidence=0.0,
        query_keywords=[],
        missing_info_types=[]
    )
    
    result2 = answer_router(state2)
    print(f"Test 2 - Human follow-up after answer: {result2} (expected: retrieve)")
    
    # Test 3: Multiple turns with latest being human - should retrieve
    state3 = AgentState(
        messages=[
            HumanMessage(content="How do I configure email?"),
            AIMessage(content="To configure email, follow these steps..."),
            HumanMessage(content="What about SMTP settings?"),
            AIMessage(content="For SMTP settings, use these configurations..."),
            HumanMessage(content="Can you explain port 587 vs 465?")
        ],
        language="en",
        contexts=[],
        confidence=None,
        clarify_turns=0,
        contexts_placeholder=False,
        last_clarify_idx=None,
        refinement_history=[],
        best_confidence=0.0,
        query_keywords=[],
        missing_info_types=[]
    )
    
    result3 = answer_router(state3)
    print(f"Test 3 - Multiple turns ending with human: {result3} (expected: retrieve)")
    
    print("\n✅ Answer router tests completed")

def test_state_preservation():
    """Test how state is preserved across turns."""
    print("\n🧪 Testing state preservation...")
    
    # Simulate first turn result
    first_turn_result = {
        'messages': [
            HumanMessage(content="How do I configure email?"),
            AIMessage(content="To configure email, follow these steps...")
        ],
        'language': 'en',
        'contexts': [{'id': 'doc1', 'title': 'Email Configuration Guide', 'content': 'some content'}],
        'confidence': 0.8,
        'clarify_turns': 0,
        'contexts_placeholder': False,
        'last_clarify_idx': None,
        'refinement_history': [],
        'best_confidence': 0.8,
        'query_keywords': ['email', 'configure'],
        'missing_info_types': []
    }
    
    print("First turn result:")
    print(f"  Messages: {len(first_turn_result['messages'])}")
    print(f"  Contexts: {len(first_turn_result['contexts'])}")
    print(f"  Confidence: {first_turn_result['confidence']}")
    print(f"  Query keywords: {first_turn_result['query_keywords']}")
    
    # Simulate adding follow-up
    follow_up_state = AgentState(
        messages=first_turn_result['messages'] + [HumanMessage(content="What about SMTP settings?")],
        language=first_turn_result['language'],
        contexts=first_turn_result['contexts'],
        confidence=first_turn_result['confidence'],
        clarify_turns=first_turn_result['clarify_turns'],
        contexts_placeholder=first_turn_result['contexts_placeholder'],
        last_clarify_idx=first_turn_result['last_clarify_idx'],
        refinement_history=first_turn_result['refinement_history'],
        best_confidence=first_turn_result['best_confidence'],
        query_keywords=first_turn_result['query_keywords'],
        missing_info_types=first_turn_result['missing_info_types']
    )
    
    print("\nFollow-up state:")
    print(f"  Messages: {len(follow_up_state.messages)}")
    print(f"  Contexts preserved: {len(follow_up_state.contexts)}")
    print(f"  Confidence preserved: {follow_up_state.confidence}")
    print(f"  Query keywords preserved: {follow_up_state.query_keywords}")
    
    # Test the message content
    print("\nMessage history:")
    for i, msg in enumerate(follow_up_state.messages):
        msg_type = type(msg).__name__
        content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
        print(f"  {i}: {msg_type}: {content}")
    
    print("\n✅ State preservation test completed")

if __name__ == "__main__":
    test_answer_router_logic()
    test_state_preservation()
