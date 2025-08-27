#!/usr/bin/env python3
"""
Test script to reproduce context loss issue in continuous conversation.
"""

import os
import sys
sys.path.insert(0, 'src')

from langchain_core.messages import HumanMessage, AIMessage
from ragdoc.agent.graph import build_graph, AgentState

def test_conversation_context():
    """Test conversation context preservation across multiple turns."""
    print("🧪 Testing conversation context preservation...")
    
    # Build the graph
    graph = build_graph()
    
    # Initial state with first question
    initial_state = AgentState(
        messages=[HumanMessage(content="Hello, how do I configure NethServer email?")],
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
    
    print("First turn - Initial question:")
    print(f"Messages: {len(initial_state.messages)}")
    for i, msg in enumerate(initial_state.messages):
        print(f"  {i}: {type(msg).__name__}: {msg.content[:50]}...")
    
    # Run first turn
    try:
        result1 = graph.invoke(initial_state)
        print(f"\nAfter first turn:")
        print(f"Messages: {len(result1.get('messages', []))}")
        for i, msg in enumerate(result1.get('messages', [])):
            msg_type = type(msg).__name__ if hasattr(msg, '__name__') else msg.get('role', 'unknown')
            content = getattr(msg, 'content', msg.get('content', ''))[:50]
            print(f"  {i}: {msg_type}: {content}...")
        
        # Add a follow-up question
        follow_up_state = AgentState(
            messages=result1.get('messages', []) + [HumanMessage(content="Can you give me more details about SMTP configuration?")],
            language=result1.get('language', 'en'),
            contexts=result1.get('contexts', []),
            confidence=result1.get('confidence'),
            clarify_turns=result1.get('clarify_turns', 0),
            contexts_placeholder=result1.get('contexts_placeholder', False),
            last_clarify_idx=result1.get('last_clarify_idx'),
            refinement_history=result1.get('refinement_history', []),
            best_confidence=result1.get('best_confidence', 0.0),
            query_keywords=result1.get('query_keywords', []),
            missing_info_types=result1.get('missing_info_types', [])
        )
        
        print(f"\nSecond turn - Follow-up question:")
        print(f"Messages: {len(follow_up_state.messages)}")
        for i, msg in enumerate(follow_up_state.messages):
            msg_type = type(msg).__name__ if hasattr(msg, '__name__') else msg.get('role', 'unknown')
            content = getattr(msg, 'content', msg.get('content', ''))[:50]
            print(f"  {i}: {msg_type}: {content}...")
        
        # Run second turn
        result2 = graph.invoke(follow_up_state)
        print(f"\nAfter second turn:")
        print(f"Messages: {len(result2.get('messages', []))}")
        for i, msg in enumerate(result2.get('messages', [])):
            msg_type = type(msg).__name__ if hasattr(msg, '__name__') else msg.get('role', 'unknown')
            content = getattr(msg, 'content', msg.get('content', ''))[:50]
            print(f"  {i}: {msg_type}: {content}...")
        
        print("\n✅ Context test completed")
        
    except Exception as e:
        print(f"❌ Error during conversation test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_conversation_context()
