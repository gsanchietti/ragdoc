from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional
import os
import logging

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from ragdoc.retrieval.retriever import Retriever, RetrieverConfig
from .prompts import get_agent_prompts
# Logger setup
logger = logging.getLogger("ragdoc.agent")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(os.getenv("RAGDOC_LOG_LEVEL", "INFO").upper())


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(str(p.get("text") or p.get("content") or p))
            else:
                parts.append(str(p))
        return " ".join(s for s in parts if s).strip()
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or content).strip()
    return str(content).strip()


def _translate_to_english(text: str) -> str:
    """Translate text to English for database search if it's not already in English."""
    if not text or not text.strip():
        return text
    
    # Enhanced heuristic to detect if text is likely English
    text_clean = text.lower().strip()
    
    # Common English function words and patterns
    english_indicators = ['the', 'and', 'or', 'to', 'of', 'in', 'for', 'with', 'on', 'at', 'by', 'from', 'how', 'what', 'where', 'when', 'why', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did', 'can', 'could', 'will', 'would', 'should']
    
    # Common non-English patterns that suggest translation is needed
    non_english_patterns = [
        # Italian
        'come', 'configurare', 'dove', 'quando', 'perché', 'cosa', 'quale', 'sono', 'è', 'del', 'della', 'degli', 'delle',
        # Spanish  
        'cómo', 'configurar', 'dónde', 'cuándo', 'por qué', 'qué', 'cuál', 'es', 'son', 'del', 'de la', 'los', 'las',
        # French
        'comment', 'configurer', 'où', 'quand', 'pourquoi', 'quoi', 'quel', 'quelle', 'est', 'sont', 'du', 'de la', 'des',
        # German
        'wie', 'konfigurieren', 'wo', 'wann', 'warum', 'was', 'welche', 'ist', 'sind', 'der', 'die', 'das', 'den',
        # Portuguese
        'como', 'configurar', 'onde', 'quando', 'por que', 'o que', 'qual', 'é', 'são', 'do', 'da', 'dos', 'das'
    ]
    
    has_english_words = any(word in text_clean for word in english_indicators)
    has_non_english_words = any(pattern in text_clean for pattern in non_english_patterns)
    
    # If text is very short, don't translate
    if len(text_clean) < 5:
        logger.debug("Translation: skipping (too short): %r", text[:50])
        return text
    
    # If text has clear non-English patterns, translate
    if has_non_english_words:
        logger.debug("Translation: non-English patterns detected: %r", text[:50])
    # If text has English words and no non-English patterns, likely English
    elif has_english_words and not has_non_english_words:
        logger.debug("Translation: skipping (appears to be English): %r", text[:50])
        return text
    # For ambiguous cases, check if it's mostly ASCII and short
    elif len(text_clean) < 15 and text.isascii():
        logger.debug("Translation: skipping (short ASCII, likely English/technical): %r", text[:50])
        return text
    
    try:
        # Use ChatOpenAI for translation
        llm = ChatOpenAI(model=os.getenv("RAGDOC_TRANSLATION_MODEL", "gpt-4o-mini"), temperature=0.0)
        
        translation_prompt = f"""Translate the following text to English. 
If the text is already in English, return it unchanged.
Only return the translated text, no explanations or additional text.

Text to translate: {text}"""
        
        response = llm.invoke([HumanMessage(content=translation_prompt)])
        translated = _content_to_text(response.content)
        
        logger.debug("Translation: %r -> %r", text[:50], translated[:50])
        return translated
        
    except Exception as e:
        logger.warning("Translation failed, using original text: %s", e)
        return text


# Agent State
@dataclass
class AgentState:
    messages: list[AnyMessage] = field(default_factory=list)
    language: Literal["it", "en", "auto"] = "it"
    contexts: list[dict[str, Any]] = field(default_factory=list)
    confidence: Optional[float] = None
    clarify_turns: int = 0
    contexts_placeholder: bool = False
    last_clarify_idx: Optional[int] = None
    # Enhanced refinement state
    refinement_history: list[dict[str, Any]] = field(default_factory=list)  # Track refinement attempts
    best_confidence: float = 0.0  # Track best confidence achieved
    query_keywords: list[str] = field(default_factory=list)  # Extracted keywords from user query
    missing_info_types: list[str] = field(default_factory=list)  # Types of info to clarify


def system_prompt(language: str = "it") -> str:
    """Get system prompt for the specified language from configuration."""
    prompts = get_agent_prompts()
    return prompts.get_system_prompt(language)


def analyze_retrieval_quality(contexts: list[dict[str, Any]], query: str) -> dict[str, Any]:
    """Analyze the quality of retrieved contexts and suggest improvements."""
    if not contexts or all(c.get("_placeholder") for c in contexts):
        return {
            "quality": "no_results",
            "max_score": 0.0,
            "avg_score": 0.0,
            "suggestions": ["add_specifics", "use_different_terms", "provide_context"]
        }
    
    non_placeholder = [c for c in contexts if not c.get("_placeholder")]
    if not non_placeholder:
        return {
            "quality": "no_results", 
            "max_score": 0.0,
            "avg_score": 0.0,
            "suggestions": ["add_specifics", "use_different_terms", "provide_context"]
        }
    
    scores = [c.get("score", 0.0) for c in non_placeholder]
    max_score = max(scores)
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    # Analyze query keywords vs titles for relevance
    query_words = set(query.lower().split())
    title_words = set()
    for c in non_placeholder[:3]:  # Top 3 results
        title = (c.get("title") or "").lower()
        title_words.update(title.split())
    
    keyword_overlap = len(query_words.intersection(title_words)) / max(len(query_words), 1)
    
    # Determine quality and suggestions
    if max_score >= 0.8 and avg_score >= 0.6:
        quality = "excellent"
        suggestions = []
    elif max_score >= 0.65 and avg_score >= 0.4:
        quality = "good"
        suggestions = ["be_more_specific"] if keyword_overlap < 0.3 else []
    elif max_score >= 0.4:
        quality = "fair"
        suggestions = ["add_specifics", "use_technical_terms"] if keyword_overlap < 0.5 else ["provide_context"]
    else:
        quality = "poor"
        suggestions = ["add_specifics", "use_different_terms", "provide_context", "check_spelling"]
    
    return {
        "quality": quality,
        "max_score": max_score,
        "avg_score": avg_score,
        "keyword_overlap": keyword_overlap,
        "suggestions": suggestions
    }


def extract_query_info(query: str) -> dict[str, Any]:
    """Extract keywords and identify missing information types from user query."""
    if not query or not query.strip():
        return {"keywords": [], "missing_types": ["topic", "context"]}
    
    # Simple keyword extraction (could be enhanced with NLP)
    words = query.lower().split()
    # Filter out common words
    stop_words = {
        "il", "la", "lo", "le", "gli", "i", "un", "una", "uno", "di", "da", "del", "della", "delle", "degli", "dei",
        "che", "per", "con", "su", "in", "a", "come", "quando", "dove", "perché", "cosa", "quale", "quanto",
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "how", "what", "where", "when", "why", "who", "which", "can", "could", "would", "should", "is", "are"
    }
    keywords = [w for w in words if len(w) > 2 and w not in stop_words]
    
    # Identify missing information types based on query analysis
    missing_types = []
    
    # Check for product/service specificity
    if not any(word in query.lower() for word in ["nethserver", "nethsecurity", "nextcloud", "webtop", "freepbx"]):
        missing_types.append("product")
    
    # Check for version specificity
    if not any(char.isdigit() for char in query):
        missing_types.append("version")
    
    # Check for error context
    if "error" in query.lower() or "errore" in query.lower():
        if "log" not in query.lower() and "messaggio" not in query.lower() and "message" not in query.lower():
            missing_types.append("error_details")
    
    # Check for configuration context
    if any(word in query.lower() for word in ["config", "configur", "impost", "setup"]):
        if not any(word in query.lower() for word in ["file", "dove", "where", "come", "how"]):
            missing_types.append("config_location")
    
    # Generic context if query is very short
    if len(keywords) < 2:
        missing_types.append("context")
    
    return {"keywords": keywords, "missing_types": missing_types or ["specifics"]}


def generate_targeted_question(state: AgentState, retrieval_analysis: dict[str, Any], query_info: dict[str, Any]) -> str:
    """Generate a targeted clarifying question based on retrieval analysis and query info."""
    suggestions = retrieval_analysis.get("suggestions", [])
    missing_types = query_info.get("missing_types", [])
    quality = retrieval_analysis.get("quality", "poor")
    
    prompts = get_agent_prompts()
    
    # Priority order for questions
    if "product" in missing_types:
        return prompts.get_question("product", state.language)
    elif "error_details" in missing_types:
        return prompts.get_question("error_details", state.language)
    elif "version" in missing_types and quality != "no_results":
        return prompts.get_question("version", state.language)
    elif "config_location" in missing_types:
        return prompts.get_question("config_location", state.language)
    elif "add_specifics" in suggestions and quality in ["fair", "good"]:
        return prompts.get_question("add_specifics", state.language)
    elif "use_different_terms" in suggestions:
        return prompts.get_question("use_different_terms", state.language)
    elif "use_technical_terms" in suggestions:
        return prompts.get_question("use_technical_terms", state.language)
    elif "provide_context" in suggestions:
        return prompts.get_question("provide_context", state.language)
    elif "check_spelling" in suggestions:
        return prompts.get_question("check_spelling", state.language)
    elif "context" in missing_types:
        return prompts.get_question("context", state.language)
    elif "specifics" in missing_types:
        return prompts.get_question("specifics", state.language)
    else:
        return prompts.get_fallback_question(state.language)


def clarify_node(state: AgentState) -> AgentState:
    """Enhanced clarifying question node with targeted analysis."""
    logger.debug("Node: clarify/start | turns=%s confidence=%.4f", state.clarify_turns, state.confidence or 0.0)
    
    # Get the current user query
    current_query = ""
    for m in reversed(state.messages):
        if isinstance(m, HumanMessage):
            current_query = _content_to_text(m.content)
            break
        elif isinstance(m, dict) and "content" in m:
            current_query = _content_to_text(m.get("content"))
            break
    
    # Analyze retrieval quality
    retrieval_analysis = analyze_retrieval_quality(state.contexts, current_query)
    
    # Extract query information
    query_info = extract_query_info(current_query)
    
    # Generate targeted question
    question = generate_targeted_question(state, retrieval_analysis, query_info)
    
    # Record this refinement attempt
    refinement_record = {
        "turn": state.clarify_turns + 1,
        "query": current_query[:100],  # Truncate for logging
        "confidence": state.confidence or 0.0,
        "quality": retrieval_analysis.get("quality"),
        "suggestions": retrieval_analysis.get("suggestions", []),
        "missing_types": query_info.get("missing_types", []),
        "question": question[:100]  # Truncate for logging
    }
    
    logger.debug("Node: clarify/analysis | quality=%s confidence=%.4f suggestions=%s", 
                retrieval_analysis.get("quality"), state.confidence or 0.0, retrieval_analysis.get("suggestions", []))
    
    msg = AIMessage(content=question)
    logger.debug("Node: clarify/emit_question | turn=%d len(messages)=%s", state.clarify_turns + 1, len(state.messages))
    
    return AgentState(
        messages=state.messages + [msg],
        language=state.language,
        contexts=state.contexts,
        confidence=state.confidence,
        clarify_turns=state.clarify_turns + 1,
        contexts_placeholder=state.contexts_placeholder,
        last_clarify_idx=len(state.messages),
        refinement_history=state.refinement_history + [refinement_record],
        best_confidence=max(state.best_confidence, state.confidence or 0.0),
        query_keywords=query_info.get("keywords", []),
        missing_info_types=query_info.get("missing_types", [])
    )


def _summarize_conversation(messages: list[AnyMessage], language: str = "it") -> str:
    """Summarize the conversation to improve search query context."""
    if len(messages) <= 1:
        # No conversation to summarize
        return ""
    
    # Build conversation text from messages
    conversation_parts = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = _content_to_text(msg.content)
            conversation_parts.append(f"User: {content}")
        elif isinstance(msg, AIMessage):
            content = _content_to_text(msg.content)
            conversation_parts.append(f"Assistant: {content}")
    
    if len(conversation_parts) <= 1:
        return ""
    
    conversation_text = "\n".join(conversation_parts)
    
    # Get prompts for summarization
    prompts = get_agent_prompts()
    system_prompt = prompts.get_summarization_system_prompt(language)
    user_prompt = prompts.get_summarization_user_prompt(language).format(conversation=conversation_text)
    
    # Use ChatOpenAI for summarization
    llm = ChatOpenAI(model=os.getenv("RAGDOC_SUMMARIZATION_MODEL", "gpt-4o-mini"), temperature=0.1)
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        summary = _content_to_text(response.content)
        logger.debug("Node: retrieve/summarization | summary_length=%d", len(summary))
        return summary
    except Exception as e:
        logger.warning("Node: retrieve/summarization_failed | error=%s", str(e))
        return ""


def retrieve_node(state: AgentState) -> AgentState:
    # Use last user message as query
    logger.debug("Node: retrieve/start | prior_contexts=%s placeholder=%s", bool(state.contexts), state.contexts_placeholder)
    query = None
    for m in reversed(state.messages):
        if isinstance(m, HumanMessage):
            query = _content_to_text(m.content)
            break
        # some frontends may pass plain dicts
        if isinstance(m, dict) and "content" in m:
            query = _content_to_text(m.get("content"))
            break
    if not query:
        logger.debug("Node: retrieve/empty_query | attempting history-aware retrieval from previous messages")

    # Summarize conversation for better context if there are multiple messages
    conversation_summary = ""
    if len(state.messages) > 1:
        conversation_summary = _summarize_conversation(state.messages, state.language)
        if conversation_summary:
            logger.debug("Node: retrieve/summary | length=%d preview=%s...", 
                        len(conversation_summary), conversation_summary[:100])
    
    # Enhance query with conversation summary for better search
    enhanced_query = query
    if conversation_summary and query:
        enhanced_query = f"{conversation_summary}\n\nCurrent question: {query}"
        logger.debug("Node: retrieve/enhanced_query | original_length=%d enhanced_length=%d", 
                    len(query), len(enhanced_query))
    elif conversation_summary and not query:
        enhanced_query = conversation_summary
        logger.debug("Node: retrieve/using_summary_as_query | length=%d", len(enhanced_query))

    # Translate query to English for better database search
    original_query = enhanced_query
    if enhanced_query:
        enhanced_query = _translate_to_english(enhanced_query)
        if enhanced_query != original_query:
            logger.debug("Node: retrieve/translated | original=%r translated=%r", original_query[:50], enhanced_query[:50])

    cfg = RetrieverConfig(dsn=os.getenv("DATABASE_URL", ""))
    retriever = Retriever(cfg)
    k = int(os.getenv("RAGDOC_RETRIEVAL_TOP_K", "8"))
    logger.debug("Node: retrieve/search | query=%r k=%s with_history=%s", enhanced_query, k, True)
    results = retriever.search(enhanced_query or "", k=k, messages=state.messages)
    logger.debug(
        "Node: retrieve/results | found=%s top=%s",
        len(results),
        [
            (
                getattr(r, "id", None),
                round(getattr(r, "score", 0.0), 4),
                (getattr(r, "title", None) or getattr(r, "path", ""))[:80],
            )
            for r in results[:5]
        ],
    )
    contexts = [
        {
            "id": r.id,
            "path": r.path,
            "url": r.source_url,
            "title": r.title,
            "score": r.score,
            "preview": r.content_preview,
        }
        for r in results
    ]
    placeholder = False
    if not contexts:
        if state.contexts:
            # keep prior non-empty contexts
            contexts = state.contexts
            placeholder = state.contexts_placeholder
            logger.debug("Node: retrieve/reuse_prior_contexts | count=%s", len(contexts))
        else:
            # synth placeholder to keep contexts list non-empty but signal placeholder
            prompts = get_agent_prompts()
            contexts = [
                {
                    "id": "placeholder",
                    "_placeholder": True,
                    "title": prompts.get_placeholder_title(state.language),
                    "path": "",
                    "url": "",
                    "score": 0.0,
                    "preview": prompts.get_placeholder_preview(state.language),
                }
            ]
            placeholder = True
            logger.debug("Node: retrieve/placeholder_inserted")
    confidence = max((c.get("score", 0.0) for c in contexts if not c.get("_placeholder")), default=0.0)
    logger.debug("Node: retrieve/done | confidence=%.4f placeholder=%s", confidence, placeholder)
    return AgentState(
        messages=state.messages,
        language=state.language,
        contexts=contexts,
        confidence=confidence,
        clarify_turns=state.clarify_turns,
        contexts_placeholder=placeholder,
        last_clarify_idx=state.last_clarify_idx,
        refinement_history=state.refinement_history,
        best_confidence=max(state.best_confidence, confidence),
        query_keywords=state.query_keywords,
        missing_info_types=state.missing_info_types,
    )


def answer_node(state: AgentState) -> AgentState:
    import json

    language = state.language if state.language in ("it", "en") else "it"
    sys_msg = SystemMessage(content=system_prompt(language))
    logger.debug(
        "Node: answer/start | contexts=%s non_placeholder=%s",
        len(state.contexts),
        len([c for c in state.contexts if not c.get("_placeholder")]),
    )

    # Build a brief context string with citations (skip placeholders)
    refs = []
    for c in [c for c in state.contexts if not c.get("_placeholder")][:5]:
        if c.get("url"):
            refs.append(f"- {c.get('title') or c.get('path')}: {c.get('url')}")
        else:
            refs.append(f"- {c.get('title') or c.get('path')}")
    
    prompts = get_agent_prompts()
    refs_text = "\n".join(refs) if refs else prompts.get_no_sources_found(language)

    prompt = prompts.get_answer_instruction(language, refs_text)

    llm = ChatOpenAI(model=os.getenv("RAGDOC_ANSWER_MODEL", "gpt-4o-mini"), temperature=0.1)
    response = llm.invoke([sys_msg] + state.messages + [HumanMessage(content=prompt)])
    logger.debug("Node: answer/done | appended AIMessage")
    return AgentState(
        messages=state.messages + [response],
        language=state.language,
        contexts=state.contexts,
        confidence=state.confidence,
        clarify_turns=state.clarify_turns,
        contexts_placeholder=state.contexts_placeholder,
        last_clarify_idx=state.last_clarify_idx,
        refinement_history=state.refinement_history,
        best_confidence=state.best_confidence,
        query_keywords=state.query_keywords,
        missing_info_types=state.missing_info_types,
    )


def escalate_node(state: AgentState) -> AgentState:
    """Enhanced escalation with refinement history summary."""
    logger.debug("Node: escalate/start | turns=%s confidence=%s best_confidence=%s", 
                state.clarify_turns, state.confidence, state.best_confidence)
    
    # Build escalation message with refinement history
    prompts = get_agent_prompts()
    
    attempts = len(state.refinement_history) if state.refinement_history else None
    confidence = state.best_confidence if state.best_confidence > 0 else None
    keywords = state.query_keywords if state.query_keywords else None
    
    base_msg = prompts.get_escalate_message(
        language=state.language,
        attempts=attempts,
        confidence=confidence, 
        keywords=keywords
    )
    
    msg = AIMessage(content=base_msg)
    logger.debug("Node: escalate/done | appended AIMessage with refinement summary")
    
    return AgentState(
        messages=state.messages + [msg],
        language=state.language,
        contexts=state.contexts,
        confidence=state.confidence,
        clarify_turns=state.clarify_turns,
        contexts_placeholder=state.contexts_placeholder,
        last_clarify_idx=state.last_clarify_idx,
        refinement_history=state.refinement_history,
        best_confidence=state.best_confidence,
        query_keywords=state.query_keywords,
        missing_info_types=state.missing_info_types,
    )


def clarify_router(state: AgentState) -> str:
    # If a new HumanMessage exists after the last clarify AI message, continue to retrieve; else wait (end run).
    if state.last_clarify_idx is None:
        logger.debug("Router: clarify -> wait | no last_clarify_idx")
        return "wait"
    for idx, m in enumerate(state.messages):
        if idx > state.last_clarify_idx and isinstance(m, HumanMessage):
            logger.debug("Router: clarify -> retrieve | human feedback detected at idx=%s", idx)
            return "retrieve"
    logger.debug("Router: clarify -> wait | no human feedback after idx=%s", state.last_clarify_idx)
    return "wait"


def supervisor(state: AgentState) -> str:
    """Enhanced supervisor with improved decision logic."""
    threshold = float(os.getenv("RAGDOC_CONFIDENCE_THRESHOLD", "0.65"))
    
    # Check if we've reached the maximum number of iterations (5)
    if state.clarify_turns >= 5:
        decision = "escalate"
        logger.debug(
            "Supervisor: decision=%s | max_iterations_reached turns=%s",
            decision,
            state.clarify_turns,
        )
        return decision
    
    # If no contexts or only placeholders, need clarification
    if (not state.contexts) or state.contexts_placeholder:
        decision = "clarify"
        logger.debug("Supervisor: decision=%s | need more info", decision)
        return decision
    
    # Check confidence against threshold
    current_confidence = state.confidence or 0.0
    
    # Enhanced decision logic: consider improvement trajectory
    if current_confidence >= threshold:
        decision = "answer"
        logger.debug(
            "Supervisor: decision=%s | confidence=%.4f threshold=%.4f",
            decision,
            current_confidence,
            threshold,
        )
        return decision
    
    # If we have made some clarify attempts, check if we're improving
    if state.clarify_turns > 0:
        # If confidence is improving and we haven't tried many times, continue
        if current_confidence > state.best_confidence * 0.9 and state.clarify_turns < 3:
            decision = "clarify"
            logger.debug(
                "Supervisor: decision=%s | improving_trajectory confidence=%.4f best=%.4f turn=%d",
                decision,
                current_confidence,
                state.best_confidence,
                state.clarify_turns,
            )
            return decision
        
        # If we have reasonable confidence but it's not perfect, try once more
        elif current_confidence >= threshold * 0.8 and state.clarify_turns < 3:
            decision = "clarify"
            logger.debug(
                "Supervisor: decision=%s | decent_confidence confidence=%.4f (80%% of threshold=%.4f) turn=%d",
                decision,
                current_confidence,
                threshold * 0.8,
                state.clarify_turns,
            )
            return decision
    
    # If confidence is too low and we've tried a few times, escalate
    if current_confidence < threshold * 0.6:
        decision = "escalate"
        logger.debug(
            "Supervisor: decision=%s | low_confidence=%.4f threshold=%.4f turns=%d",
            decision,
            current_confidence,
            threshold,
            state.clarify_turns,
        )
        return decision
    
    # Default: try to clarify if we haven't reached the limit
    decision = "clarify"
    logger.debug(
        "Supervisor: decision=%s | default_clarify confidence=%.4f threshold=%.4f turns=%d",
        decision,
        current_confidence,
        threshold,
        state.clarify_turns,
    )
    return decision


def build_graph() -> Any:
    from typing import Any as _Any
    import os

    graph = StateGraph(AgentState)
    logger.debug("Graph: build/start")
    graph.add_node("clarify", clarify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("escalate", escalate_node)

    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges("retrieve", supervisor, {"clarify": "clarify", "answer": "answer", "escalate": "escalate"})
    # Human-in-the-loop: if feedback already present, loop to retrieve; otherwise end and wait for user
    graph.add_conditional_edges("clarify", clarify_router, {"retrieve": "retrieve", "wait": END})
    graph.add_edge("answer", END)
    graph.add_edge("escalate", END)

    compiled = graph.compile()
    logger.debug("Graph: build/done")
    return compiled
