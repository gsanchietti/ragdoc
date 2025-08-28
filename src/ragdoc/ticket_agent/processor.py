from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, Optional
from pathlib import Path
from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from .models import TicketAgentConfig
from .freshdesk_client import FreshdeskAPI

# Logger setup
logger = logging.getLogger("ragdoc.ticket_agent.processor")


class TicketProcessor:
    """Process Freshdesk tickets into tutorial markdown files."""
    
    def __init__(self, config: TicketAgentConfig):
        self.config = config
        self.freshdesk = FreshdeskAPI(config.freshdesk_domain, config.freshdesk_api_key)
        
        # Initialize AI models
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
      
        self.analyzer = ChatOpenAI(
            model=config.analysis_model, 
            api_key=openai_api_key,
            temperature=0.3
        )
        
        # Ensure output directory exists
        Path(config.output_directory).mkdir(parents=True, exist_ok=True)
        
    def _detect_language(self, text: str) -> str:
        """Detect if text is in English or another language."""
        # Simple heuristic-based detection
        english_indicators = ['the', 'and', 'to', 'is', 'in', 'it', 'you', 'that', 'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they', 'i', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by', 'word', 'but', 'not', 'what', 'all', 'were', 'we', 'when', 'your', 'can', 'said', 'there', 'each', 'which', 'she', 'do', 'how', 'their', 'if', 'will', 'up', 'other', 'about', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more', 'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call', 'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part']
        
        # Non-English patterns
        italian_patterns = ['che', 'con', 'del', 'della', 'per', 'una', 'alla', 'nel', 'sul', 'sono', 'come', 'dove', 'quando', 'perché', 'così', 'più', 'molto', 'tutto', 'anche', 'solo', 'già', 'ancora', 'sempre', 'ogni', 'alcuni', 'molti', 'tutti', 'nessuno', 'qualche', 'altri', 'stesso', 'primo', 'ultimo', 'nuovo', 'grande', 'piccolo', 'buono', 'cattivo', 'bello', 'brutto']
        spanish_patterns = ['que', 'con', 'del', 'para', 'una', 'por', 'como', 'donde', 'cuando', 'porque', 'así', 'más', 'muy', 'todo', 'también', 'solo', 'ya', 'siempre', 'cada', 'algunos', 'muchos', 'todos', 'nadie', 'algún', 'otros', 'mismo', 'primero', 'último', 'nuevo', 'grande', 'pequeño', 'bueno', 'malo', 'hermoso', 'feo']
        
        text_lower = text.lower()
        words = text_lower.split()
        
        english_count = sum(1 for word in words if word in english_indicators)
        italian_count = sum(1 for word in words if word in italian_patterns)
        spanish_count = sum(1 for word in words if word in spanish_patterns)
        
        total_words = len(words)
        if total_words == 0:
            return "unknown"
            
        english_ratio = english_count / total_words
        non_english_ratio = (italian_count + spanish_count) / total_words
        
        if english_ratio > 0.15:
            return "english"
        elif non_english_ratio > 0.1:
            return "non-english"
        else:
            return "unknown"
    
    def _analyze_ticket_content(self, ticket_data: Dict[str, Any], conversations: list) -> Dict[str, str]:
        """Analyze ticket content and extract tutorial information."""
        
        # Load the template
        template = self._load_template()

        # Prepare ticket content
        title = ticket_data.get('subject', 'Untitled')
        description = ticket_data.get('description_text', '')
        id = ticket_data.get('id', 0)
        source = f'{self.config.freshdesk_base_url}/a/tickets/{id}'

        # Collect conversation content
        conversation_text = ""
        for conv in conversations:
            if conv.get('body_text'):
                conversation_text += f"\n{conv['body_text']}"
        
       
        # Analyze content with AI
        analysis_prompt = f"""
Before starting the analysis, translate everything to English:
- Title: {title}
- Description: {description}
- Conversation: {conversation_text}

Metadata:
- Source: {source}

Analyze this support ticket and extract information for creating a tutorial:

Problem description: Clear description of the problem or issue
Question: Formulate the problem also as a question with rich details
Solution steps: Step-by-step solution or guidance
Additional information: Any additional helpful information, tips, or context
Tags: Comma-separated relevant tags or keywords
Complexity: Simple|Medium|Advanced
Quality: 1-10 rating of how useful this would be as a tutorial

Focus on creating practical, actionable content. If the ticket doesn't contain enough information for a good tutorial, set tutorial_quality to a low score.
Never include names or personal information.
Never include steps to enable remote access control.
Replace IP addresses and hostnames with example placeholders.

Return the response populating this template with extraced information:

{template}
"""
        
        try:
            response = self.analyzer.invoke([HumanMessage(content=analysis_prompt)])    
            return response.content.strip()
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return None
    
    def _load_template(self) -> str:
        """Load the tutorial template."""
        template_path = Path(self.config.template_file)
        
        if template_path.exists():
            return template_path.read_text(encoding='utf-8')
        else:
            raise FileNotFoundError(f"Template file not found: {self.config.template_file}")
    
    def process_ticket(self, ticket_id: int) -> Optional[str]:
        """Process a single ticket and generate a tutorial."""
        logger.info(f"Processing ticket {ticket_id}")
        
        # Fetch ticket data
        ticket_data = self.freshdesk.get_ticket(ticket_id)
        if not ticket_data:
            logger.error(f"Could not fetch ticket {ticket_id}")
            return None
        
        # Fetch conversations
        conversations = self.freshdesk.get_conversations(ticket_id)
        
        # Check if ticket is solved/closed
        status = ticket_data.get('status', 0)
        if status not in [4, 5]:  # 4=Resolved, 5=Closed
            logger.warning(f"Ticket {ticket_id} is not solved/closed (status: {status})")
        
        # Analyze content
        response = self._analyze_ticket_content(ticket_data, conversations)
        if response is None:
            logger.error(f"Analysis failed for ticket {ticket_id}")
            return None
        
        # Save to file
        output_path = Path(self.config.output_directory) / f"ticket_{ticket_id}_tutorial.md"
        output_path.write_text(response, encoding='utf-8')
        
        logger.info(f"Generated tutorial: {output_path}")
        return str(output_path)


def create_ticket_agent(config: TicketAgentConfig) -> TicketProcessor:
    """Create a ticket processing agent with the given configuration."""
    return TicketProcessor(config)
