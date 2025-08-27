from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from ragdoc.retrieval.retriever import Retriever, RetrieverConfig


import argparse
import logging
import os
import sys
from typing import Optional

from ragdoc.retrieval.retriever import Retriever, RetrieverConfig


def handle_query(args) -> int:
    # Set up debug logging for retrieval operations
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    
    dsn = args.database_url or os.getenv("DATABASE_URL", "")
    if not dsn:
        print("DATABASE_URL not provided.")
        return 2
    text = args.text.strip()
    if not text:
        print("Empty query text.")
        return 2

    # Create retriever config that ensures hybrid search is always executed
    # Handle BM25 enable/disable flags
    use_bm25 = args.use_bm25 and not args.no_bm25
    
    config = RetrieverConfig(
        dsn=dsn, 
        k=args.k,
        alpha=args.alpha,  # Allow configuring the hybrid balance
        use_fts=True,      # Always enable full-text search
        title_boost=args.title_boost,  # Allow configuring title boost
        use_bm25=use_bm25,  # Allow enabling/disabling BM25
        bm25_k1=args.bm25_k1,  # BM25 term frequency saturation
        bm25_b=args.bm25_b,    # BM25 length normalization
        sparse_weight=args.sparse_weight  # Weight for sparse vector component
    )
    
    retriever = Retriever(config)
    
    print(f"Executing hybrid search with query: {text!r}")
    print(f"Configuration: alpha={config.alpha:.2f}, use_fts={config.use_fts}, title_boost={config.title_boost:.1f}")
    print(f"BM25 Configuration: use_bm25={config.use_bm25}, k1={config.bm25_k1:.1f}, b={config.bm25_b:.2f}, sparse_weight={config.sparse_weight:.2f}")
    
    results = retriever.search(text, k=args.k)

    print(f"\nHybrid search results ({len(results)} found):")
    print("=" * 80)
    
    for i, r in enumerate(results, start=1):
        if config.use_bm25:
            print(f"#{i} hybrid_score={r.score:.4f} vector_score={r.vscore:.4f} lexical_score={r.lscore:.4f} sparse_score={r.sscore:.4f}")
        else:
            print(f"#{i} hybrid_score={r.score:.4f} vector_score={r.vscore:.4f} lexical_score={r.lscore:.4f}")
        print(f"   type={r.source_type} path={r.path}")
        if r.source_url:
            print(f"   url: {r.source_url}")
        if r.title:
            print(f"   title: {r.title}")
        print(f"   preview: {r.content_preview!r}")
        print()
        
    if not results:
        print("No results found.")
    else:
        print(f"Hybrid search completed successfully with {len(results)} results.")
    
    return 0


def handle_chat(args) -> int:
    """Handle the chat subcommand."""
    # Load environment variables from .env file if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not available, environment variables must be set manually
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment")
        print("Make sure you have a .env file with your OpenAI API key or set the environment variable")
        return 1
        
    if not os.getenv("DATABASE_URL"):
        print("❌ Error: DATABASE_URL not found in environment")
        print("Make sure you have a database configured")
        return 1
    
    # Import chat functionality
    try:
        from ragdoc.agent.graph import build_graph, AgentState
        from langchain_core.messages import HumanMessage, AIMessage
    except ImportError as e:
        print(f"❌ Error importing required modules for chat: {e}")
        print("Make sure all dependencies are installed")
        return 1
    
    if args.test:
        return run_automated_chat_test(args.language)
    else:
        return run_interactive_chat(args.language)


def run_interactive_chat(language: str = "it") -> int:
    """Run interactive chat session."""
    from ragdoc.agent.graph import build_graph, AgentState
    from langchain_core.messages import HumanMessage, AIMessage
    
    print("🤖 RAGDoc Chat Interface")
    print("=" * 50)
    print(f"Language: {language}")
    print("Type 'quit', 'exit', or press Ctrl+C to stop")
    print("Type 'debug' to see current conversation state")
    print("Type 'clear' to clear conversation history")
    print("=" * 50)
    
    # Initialize graph and conversation state
    graph = build_graph()
    conversation_state = AgentState(
        messages=[],
        language=language,
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
        while True:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
                
            # Handle special commands
            if user_input.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
                
            if user_input.lower() == 'debug':
                show_debug_info(conversation_state)
                continue
                
            if user_input.lower() == 'clear':
                conversation_state = AgentState(
                    messages=[],
                    language=language,
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
                print("🧹 Conversation history cleared.")
                continue
            
            # Process user message
            print("🔄 Processing...")
            try:
                conversation_state = process_message(graph, conversation_state, user_input, language)
                
                # Get the last AI message
                last_message = None
                for msg in reversed(conversation_state.messages):
                    if isinstance(msg, AIMessage):
                        last_message = msg
                        break
                        
                if last_message:
                    print(f"\n🤖 RAGDoc: {last_message.content}")
                else:
                    print("\n❌ No response generated.")
                    
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("You can continue chatting or type 'debug' for more info.")
                
    except KeyboardInterrupt:
        print("\n\n👋 Chat interrupted. Goodbye!")
        return 0
    
    return 0


def run_automated_chat_test(language: str = "it") -> int:
    """Run automated test of the conversation issue."""
    from ragdoc.agent.graph import build_graph, AgentState
    from langchain_core.messages import HumanMessage, AIMessage
    
    print("🧪 Running automated conversation test...")
    print("=" * 50)
    
    # Initialize
    graph = build_graph()
    conversation_state = AgentState(
        messages=[],
        language=language,
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
    
    # Test conversation sequence
    test_messages = [
        "abbiamo perso la password di un NethSecurity 8 c'è modo di resettarla?",
        "c'è un altro modo per farlo? eventualmente qual'è quella di default?"
    ]
    
    try:
        for i, message in enumerate(test_messages, 1):
            print(f"\n🔹 Test Message {i}:")
            print(f"👤 Human: {message}")
            
            conversation_state = process_message(graph, conversation_state, message, language)
            
            # Get the last AI message
            last_message = None
            for msg in reversed(conversation_state.messages):
                if isinstance(msg, AIMessage):
                    last_message = msg
                    break
            
            if last_message:
                response = last_message.content
                print(f"🤖 RAGDoc: {response[:200]}{'...' if len(response) > 200 else ''}")
                
                # Analyze response for the second message
                if i == 2:
                    print(f"\n🔍 Analyzing follow-up response for context awareness...")
                    response_lower = response.lower()
                    
                    # Check for context understanding
                    context_indicators = ["altro modo", "alternativo", "default", "password", "admin"]
                    found_indicators = [ind for ind in context_indicators if ind in response_lower]
                    
                    # Check for problematic phrases
                    problematic_phrases = [
                        "maggiori dettagli", "specificare", "cosa ti riferisci", 
                        "potresti specificare", "avrei bisogno di", "non ho capito"
                    ]
                    found_problems = [phrase for phrase in problematic_phrases if phrase in response_lower]
                    
                    print(f"✅ Context indicators found: {found_indicators}")
                    if found_problems:
                        print(f"❌ Problematic phrases found: {found_problems}")
                        print("❌ TEST FAILED: Agent is asking for clarification instead of understanding context")
                        return 1
                    else:
                        print(f"✅ No problematic clarification requests found")
                        
                    if found_indicators:
                        print(f"✅ TEST PASSED: Agent shows context awareness")
                        return 0
                    else:
                        print(f"⚠️  TEST UNCLEAR: Limited context indicators found")
                        return 0
            else:
                print("❌ No response generated")
                return 1
                        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return 1
    
    return 0


def process_message(graph, conversation_state, user_input: str, language: str):
    """Process a user message and return updated conversation state."""
    from ragdoc.agent.graph import AgentState
    from langchain_core.messages import HumanMessage
    
    # Add user message to conversation
    new_messages = conversation_state.messages + [HumanMessage(content=user_input)]
    
    # Create new state with user message
    new_state = AgentState(
        messages=new_messages,
        language=conversation_state.language,
        contexts=conversation_state.contexts,
        confidence=conversation_state.confidence,
        clarify_turns=conversation_state.clarify_turns,
        contexts_placeholder=conversation_state.contexts_placeholder,
        last_clarify_idx=conversation_state.last_clarify_idx,
        refinement_history=conversation_state.refinement_history,
        best_confidence=conversation_state.best_confidence,
        query_keywords=conversation_state.query_keywords,
        missing_info_types=conversation_state.missing_info_types
    )
    
    # Process through graph
    result = graph.invoke(new_state)
    
    # Return updated conversation state
    return AgentState(
        messages=result.get('messages', []),
        language=result.get('language', language),
        contexts=result.get('contexts', []),
        confidence=result.get('confidence'),
        clarify_turns=result.get('clarify_turns', 0),
        contexts_placeholder=result.get('contexts_placeholder', False),
        last_clarify_idx=result.get('last_clarify_idx'),
        refinement_history=result.get('refinement_history', []),
        best_confidence=result.get('best_confidence', 0.0),
        query_keywords=result.get('query_keywords', []),
        missing_info_types=result.get('missing_info_types', [])
    )


def show_debug_info(conversation_state):
    """Show debug information about current conversation state."""
    print("\n" + "=" * 30 + " DEBUG INFO " + "=" * 30)
    print(f"Messages: {len(conversation_state.messages)}")
    print(f"Language: {conversation_state.language}")
    print(f"Contexts: {len(conversation_state.contexts)}")
    print(f"Confidence: {conversation_state.confidence}")
    print(f"Clarify turns: {conversation_state.clarify_turns}")
    print(f"Best confidence: {conversation_state.best_confidence}")
    print(f"Query keywords: {conversation_state.query_keywords}")
    print(f"Refinement history: {len(conversation_state.refinement_history)} items")
    
    print("\nConversation history:")
    for i, msg in enumerate(conversation_state.messages):
        from langchain_core.messages import HumanMessage
        msg_type = "👤 Human" if isinstance(msg, HumanMessage) else "🤖 AI"
        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        print(f"  {i+1}. {msg_type}: {content}")
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragdoc-test", description="Test utilities for ragdoc")
    sub = parser.add_subparsers(dest="action", required=True)

    # Query subcommand (existing)
    q = sub.add_parser("query", help="Query the vector DB with a text prompt using hybrid search")
    q.add_argument("text", help="Query text")
    q.add_argument("--k", type=int, default=5, help="Top K results (default: 5)")
    q.add_argument("--database-url", default=os.getenv("DATABASE_URL", ""), help="Postgres DSN")
    q.add_argument("--alpha", type=float, default=0.7, help="Hybrid search balance: vector weight (default: 0.7)")
    q.add_argument("--title-boost", type=float, default=1.5, help="Title match boost factor (default: 1.5)")
    q.add_argument("--use-bm25", action="store_true", default=True, help="Enable BM25 sparse vector search (default: True)")
    q.add_argument("--no-bm25", action="store_true", help="Disable BM25 sparse vector search")
    q.add_argument("--bm25-k1", type=float, default=1.2, help="BM25 term frequency saturation parameter (default: 1.2)")
    q.add_argument("--bm25-b", type=float, default=0.75, help="BM25 length normalization parameter (default: 0.75)")
    q.add_argument("--sparse-weight", type=float, default=0.3, help="Weight for sparse vector component (default: 0.3)")
    q.add_argument("--debug", action="store_true", help="Enable debug logging to see executed queries")
    q.set_defaults(func=handle_query)

    # Chat subcommand (new)
    c = sub.add_parser("chat", help="Interactive chat interface with the RAGDoc agent")
    c.add_argument(
        "--language", "-l", 
        choices=["it", "en"], 
        default="it",
        help="Chat language (default: it)"
    )
    c.add_argument(
        "--test", "-t",
        action="store_true",
        help="Run automated test of conversation context preservation"
    )
    c.set_defaults(func=handle_chat)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
