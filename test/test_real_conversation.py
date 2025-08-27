#!/usr/bin/env python3
"""
Test script to reproduce the context loss issue with real API calls.
Testing the specific conversation pattern described by the user.
"""

import os
import sys
sys.path.insert(0, 'src')

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from ragdoc.agent.graph import build_graph, AgentState

def test_nethsecurity_conversation():
    """Test the specific NethSecurity conversation to verify context preservation."""
    print("🧪 Testing NethSecurity Password Reset Conversation")
    print("="*80)
    
    # Verify environment
    print(f"✅ OpenAI API Key loaded: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
    print(f"✅ Database URL: {os.getenv('DATABASE_URL', 'Not set')}")
    
    # Build the graph
    graph = build_graph()
    
    # Step 1: First question
    print(f"\n🔹 Step 1: Initial question about NethSecurity password reset")
    initial_question = "abbiamo perso la password di un NethSecurity 8 c'è modo di resettarla?"
    print(f"Human: {initial_question}")
    
    initial_state = AgentState(
        messages=[HumanMessage(content=initial_question)],
        language="it",
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
    
    try:
        # Run first turn
        print("⏳ Processing first question...")
        result1 = graph.invoke(initial_state)
        
        print(f"\n🤖 Assistant Response (Turn 1):")
        last_message = result1['messages'][-1]
        assistant_response = last_message.content if hasattr(last_message, 'content') else str(last_message)
        print(assistant_response)
        
        print(f"\n📊 State after first turn:")
        print(f"  - Messages: {len(result1.get('messages', []))}")
        print(f"  - Contexts: {len(result1.get('contexts', []))}")
        print(f"  - Confidence: {result1.get('confidence', 'None')}")
        print(f"  - Language: {result1.get('language', 'None')}")
        
        # Pause between turns
        print(f"\n" + "="*50)
        input("Press Enter to continue with follow-up question...")
        
        # Step 2: Follow-up question
        print(f"\n🔹 Step 2: Follow-up question")
        followup_question = "c'è un altro modo per farlo? eventualmente qual'è quella di default?"
        print(f"Human: {followup_question}")
        
        # Create state for follow-up - this is what would happen in a real conversation
        followup_state = AgentState(
            messages=result1.get('messages', []) + [HumanMessage(content=followup_question)],
            language=result1.get('language', 'it'),
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
        
        print(f"\n📊 Conversation state before second turn:")
        print(f"  - Total messages: {len(followup_state.messages)}")
        print(f"  - Conversation context (last 3 messages):")
        for i, msg in enumerate(followup_state.messages[-3:]):
            msg_type = "Human" if isinstance(msg, HumanMessage) else "Assistant"
            content = getattr(msg, 'content', str(msg))[:150]
            print(f"    {msg_type}: {content}{'...' if len(str(content)) > 150 else ''}")
        
        # Run second turn
        print("⏳ Processing follow-up question...")
        result2 = graph.invoke(followup_state)
        
        print(f"\n🤖 Assistant Follow-up Response (Turn 2):")
        last_message = result2['messages'][-1]
        assistant_response2 = last_message.content if hasattr(last_message, 'content') else str(last_message)
        print(assistant_response2)
        
        print(f"\n📊 Final conversation state:")
        print(f"  - Total messages: {len(result2.get('messages', []))}")
        print(f"  - Contexts: {len(result2.get('contexts', []))}")
        print(f"  - Confidence: {result2.get('confidence', 'None')}")
        
        # Analyze the response for context awareness
        print(f"\n" + "="*60)
        print("🔍 CONTEXT ANALYSIS")
        print("="*60)
        
        response_lower = assistant_response2.lower()
        
        # Check if the response shows understanding of the context
        context_indicators = [
            "altro modo",
            "password", 
            "nethsecurity",
            "reset",
            "default",
            "predefinita",
            "alternativo",
            "recuper"
        ]
        
        found_indicators = [indicator for indicator in context_indicators if indicator in response_lower]
        print(f"✅ Context indicators found: {found_indicators}")
        
        # Check for problematic phrases that indicate lack of context
        problematic_phrases = [
            "maggiori dettagli",
            "specificare",
            "cosa ti riferisci",
            "potresti specificare",
            "avrei bisogno di",
            "non ho capito",
            "non è chiaro",
            "cosa intendi",
            "altro modo"  # This should be understood from context
        ]
        
        found_problems = [phrase for phrase in problematic_phrases if phrase in response_lower]
        if found_problems:
            print(f"❌ PROBLEM: Found phrases indicating context loss: {found_problems}")
            print(f"❌ The agent is asking for clarification instead of understanding the context!")
        else:
            print(f"✅ No problematic phrases found - good context awareness")
        
        # Specific analysis for this conversation
        if "altro modo" in response_lower and any(word in response_lower for word in ["password", "reset", "nethsecurity"]):
            print(f"✅ Response shows awareness of the password reset context")
        elif "maggiori dettagli" in response_lower or "specificare" in response_lower:
            print(f"❌ CONTEXT LOSS: Agent is asking for details instead of understanding 'altro modo' refers to password reset")
        else:
            print(f"⚠️  Context understanding unclear")
        
        # Print full conversation for review
        print(f"\n" + "="*60)
        print("📝 FULL CONVERSATION TRANSCRIPT")
        print("="*60)
        for i, msg in enumerate(result2.get('messages', [])):
            msg_type = "👤 Human" if isinstance(msg, HumanMessage) else "🤖 Assistant"
            content = getattr(msg, 'content', str(msg))
            print(f"\n{i+1}. {msg_type}:")
            print(f"   {content}")
            
        return result2
        
    except Exception as e:
        print(f"❌ Error during conversation test: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Check if we have the API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        sys.exit(1)
    
    print(f"✅ Using OpenAI API key: {os.getenv('OPENAI_API_KEY')[:20]}...")
    
    test_nethsecurity_conversation()
