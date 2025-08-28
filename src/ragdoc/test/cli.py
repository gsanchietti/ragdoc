from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

from ragdoc.retrieval.retriever import Retriever, RetrieverConfig


def load_env_config():
    """Load configuration from .env file if available."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not available, environment variables must be set manually


def get_config_value(key: str, default=None, value_type=str):
    """Get configuration value from environment with type conversion."""
    value = os.getenv(key, default)
    if value is None:
        return None
    
    if value_type == bool:
        return value.lower() in ('true', '1', 'yes', 'on')
    elif value_type == int:
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    elif value_type == float:
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    else:
        return value


def handle_query(args) -> int:
    # Load environment configuration
    load_env_config()
    
    # Set up debug logging for retrieval operations
    log_level = get_config_value("RAGDOC_LOG_LEVEL", "INFO").upper()
    if args.debug or log_level == "DEBUG":
        level = logging.DEBUG
    else:
        level = getattr(logging, log_level, logging.INFO)
        
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    
    # Get database URL from args or environment
    dsn = args.database_url or get_config_value("DATABASE_URL", "")
    if not dsn:
        print("❌ DATABASE_URL not provided in arguments or .env file.")
        return 2
        
    text = args.text.strip()
    if not text:
        print("❌ Empty query text.")
        return 2

    # Get configuration values from .env with command line overrides
    k = args.k if hasattr(args, 'k') and args.k != 5 else get_config_value("RAGDOC_RETRIEVAL_TOP_K", 5, int)
    alpha = args.alpha if hasattr(args, 'alpha') and args.alpha != 0.7 else get_config_value("RAGDOC_RETRIEVAL_ALPHA", 0.7, float)
    title_boost = args.title_boost if hasattr(args, 'title_boost') and args.title_boost != 1.5 else get_config_value("RAGDOC_RETRIEVAL_TITLE_BOOST", 1.5, float)
    use_fts = get_config_value("RAGDOC_RETRIEVAL_USE_FTS", True, bool)
    
    # Handle BM25 configuration
    use_bm25_env = get_config_value("RAGDOC_RETRIEVAL_USE_BM25", True, bool)
    use_bm25 = args.use_bm25 and not args.no_bm25 if hasattr(args, 'use_bm25') else use_bm25_env
    bm25_k1 = args.bm25_k1 if hasattr(args, 'bm25_k1') and args.bm25_k1 != 1.2 else get_config_value("RAGDOC_RETRIEVAL_BM25_K1", 1.2, float)
    bm25_b = args.bm25_b if hasattr(args, 'bm25_b') and args.bm25_b != 0.75 else get_config_value("RAGDOC_RETRIEVAL_BM25_B", 0.75, float)
    sparse_weight = args.sparse_weight if hasattr(args, 'sparse_weight') and args.sparse_weight != 0.3 else get_config_value("RAGDOC_RETRIEVAL_SPARSE_WEIGHT", 0.3, float)
    
    # Create retriever config with .env defaults and CLI overrides
    config = RetrieverConfig(
        dsn=dsn, 
        k=k,
        alpha=alpha,
        use_fts=use_fts,
        title_boost=title_boost,
        use_bm25=use_bm25,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        sparse_weight=sparse_weight
    )
    
    retriever = Retriever(config)
    
    print(f"🔍 Executing hybrid search with query: {text!r}")
    print(f"📊 Configuration: alpha={config.alpha:.2f}, use_fts={config.use_fts}, title_boost={config.title_boost:.1f}")
    print(f"🎯 BM25 Configuration: use_bm25={config.use_bm25}, k1={config.bm25_k1:.1f}, b={config.bm25_b:.2f}, sparse_weight={config.sparse_weight:.2f}")
    
    results = retriever.search(text, k=k)

    print(f"\n📋 Hybrid search results ({len(results)} found):")
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
    # Load environment configuration first
    load_env_config()
    
    # Check required environment variables
    openai_key = get_config_value("OPENAI_API_KEY")
    if not openai_key:
        print("❌ Error: OPENAI_API_KEY not found in environment")
        print("💡 Make sure you have a .env file with your OpenAI API key or set the environment variable")
        return 1
        
    database_url = get_config_value("DATABASE_URL")
    if not database_url:
        print("❌ Error: DATABASE_URL not found in environment")
        print("💡 Make sure you have a database configured in your .env file")
        return 1
    
    # Set up logging from .env configuration
    log_level = get_config_value("RAGDOC_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    
    # Import chat functionality
    try:
        from ragdoc.agent.graph import build_graph, AgentState
        from langchain_core.messages import HumanMessage, AIMessage
    except ImportError as e:
        print(f"❌ Error importing required modules for chat: {e}")
        print("💡 Make sure all dependencies are installed")
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
    print(f"🌐 Language: {language}")
    print(f"🗄️  Database: {get_config_value('DATABASE_URL', 'Not configured')[:50]}...")
    print(f"🧠 Answer Model: {get_config_value('RAGDOC_ANSWER_MODEL', 'gpt-4o-mini')}")
    print(f"🎯 Confidence Threshold: {get_config_value('RAGDOC_CONFIDENCE_THRESHOLD', 0.65, float)}")
    print(f"📊 Retrieval Top-K: {get_config_value('RAGDOC_RETRIEVAL_TOP_K', 8, int)}")
    print(f"🔍 BM25 Enabled: {get_config_value('RAGDOC_RETRIEVAL_USE_BM25', True, bool)}")
    print("=" * 50)
    print("💬 Type 'quit', 'exit', or press Ctrl+C to stop")
    print("🐛 Type 'debug' to see current conversation state")
    print("🧹 Type 'clear' to clear conversation history")
    print("⚙️  Type 'config' to show current configuration")
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
                
            if user_input.lower() == 'config':
                show_config_info()
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


def show_config_info():
    """Show current configuration from .env file."""
    print("\n" + "=" * 30 + " CONFIGURATION " + "=" * 30)
    
    # Core settings
    print("🔧 Core Settings:")
    print(f"   Database URL: {get_config_value('DATABASE_URL', 'Not set')[:50]}...")
    print(f"   OpenAI API Key: {'✅ Set' if get_config_value('OPENAI_API_KEY') else '❌ Not set'}")
    print(f"   Log Level: {get_config_value('RAGDOC_LOG_LEVEL', 'INFO')}")
    
    # Model configuration
    print("\n🧠 Model Configuration:")
    print(f"   Answer Model: {get_config_value('RAGDOC_ANSWER_MODEL', 'gpt-4o-mini')}")
    print(f"   Summarization Model: {get_config_value('RAGDOC_SUMMARIZATION_MODEL', 'gpt-4o-mini')}")
    print(f"   Translation Model: {get_config_value('RAGDOC_TRANSLATION_MODEL', 'gpt-4o-mini')}")
    
    # Retrieval settings
    print("\n🔍 Retrieval Configuration:")
    print(f"   Top-K Results: {get_config_value('RAGDOC_RETRIEVAL_TOP_K', 8, int)}")
    print(f"   Confidence Threshold: {get_config_value('RAGDOC_CONFIDENCE_THRESHOLD', 0.65, float)}")
    print(f"   Alpha (Vector Weight): {get_config_value('RAGDOC_RETRIEVAL_ALPHA', 0.7, float)}")
    print(f"   Title Boost: {get_config_value('RAGDOC_RETRIEVAL_TITLE_BOOST', 1.5, float)}")
    print(f"   Use FTS: {get_config_value('RAGDOC_RETRIEVAL_USE_FTS', True, bool)}")
    
    # BM25 settings
    print("\n🎯 BM25 Configuration:")
    print(f"   Use BM25: {get_config_value('RAGDOC_RETRIEVAL_USE_BM25', True, bool)}")
    print(f"   BM25 K1: {get_config_value('RAGDOC_RETRIEVAL_BM25_K1', 1.2, float)}")
    print(f"   BM25 B: {get_config_value('RAGDOC_RETRIEVAL_BM25_B', 0.75, float)}")
    print(f"   Sparse Weight: {get_config_value('RAGDOC_RETRIEVAL_SPARSE_WEIGHT', 0.3, float)}")
    
    print("=" * 76)


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
    # Load .env configuration first to get defaults
    load_env_config()
    
    # Get default values from .env
    default_k = get_config_value("RAGDOC_RETRIEVAL_TOP_K", 5, int)
    default_database_url = get_config_value("DATABASE_URL", "")
    default_alpha = get_config_value("RAGDOC_RETRIEVAL_ALPHA", 0.7, float)
    default_title_boost = get_config_value("RAGDOC_RETRIEVAL_TITLE_BOOST", 1.5, float)
    default_use_bm25 = get_config_value("RAGDOC_RETRIEVAL_USE_BM25", True, bool)
    default_bm25_k1 = get_config_value("RAGDOC_RETRIEVAL_BM25_K1", 1.2, float)
    default_bm25_b = get_config_value("RAGDOC_RETRIEVAL_BM25_B", 0.75, float)
    default_sparse_weight = get_config_value("RAGDOC_RETRIEVAL_SPARSE_WEIGHT", 0.3, float)
    
    parser = argparse.ArgumentParser(
        prog="ragdoc-test", 
        description="Test utilities for ragdoc - loads configuration from .env file"
    )
    sub = parser.add_subparsers(dest="action", required=True)

    # Query subcommand with .env defaults
    q = sub.add_parser("query", help="Query the vector DB with a text prompt using hybrid search")
    q.add_argument("text", help="Query text")
    q.add_argument("--k", type=int, default=default_k, 
                   help=f"Top K results (default: {default_k} from .env or fallback)")
    q.add_argument("--database-url", default=default_database_url, 
                   help=f"Postgres DSN (default: from .env DATABASE_URL)")
    q.add_argument("--alpha", type=float, default=default_alpha, 
                   help=f"Hybrid search balance: vector weight (default: {default_alpha} from .env)")
    q.add_argument("--title-boost", type=float, default=default_title_boost, 
                   help=f"Title match boost factor (default: {default_title_boost} from .env)")
    q.add_argument("--use-bm25", action="store_true", default=default_use_bm25, 
                   help=f"Enable BM25 sparse vector search (default: {default_use_bm25} from .env)")
    q.add_argument("--no-bm25", action="store_true", 
                   help="Disable BM25 sparse vector search")
    q.add_argument("--bm25-k1", type=float, default=default_bm25_k1, 
                   help=f"BM25 term frequency saturation parameter (default: {default_bm25_k1} from .env)")
    q.add_argument("--bm25-b", type=float, default=default_bm25_b, 
                   help=f"BM25 length normalization parameter (default: {default_bm25_b} from .env)")
    q.add_argument("--sparse-weight", type=float, default=default_sparse_weight, 
                   help=f"Weight for sparse vector component (default: {default_sparse_weight} from .env)")
    q.add_argument("--debug", action="store_true", 
                   help="Enable debug logging (overrides RAGDOC_LOG_LEVEL in .env)")
    q.set_defaults(func=handle_query)

    # Chat subcommand
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
