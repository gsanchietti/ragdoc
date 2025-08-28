#!/usr/bin/env python3
"""
RAGDoc Ticket Agent CLI

Command-line interface for processing a single Freshdesk ticket into tutorial markdown file.
"""

import argparse
import logging

from .models import TicketAgentConfig
from .processor import create_ticket_agent


def setup_logging(debug: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("ragdoc_ticket_agent.log")
        ]
    )


def handle_process_ticket(args) -> int:
    """Handle processing a single ticket."""
    try:
        config = TicketAgentConfig(
            freshdesk_domain=args.freshdesk_domain,
            freshdesk_api_key=args.freshdesk_api_key,
            template_file=args.template,
            output_directory=args.output_dir,
            translation_model=args.translation_model,
            analysis_model=args.analysis_model
        )
        
        agent = create_ticket_agent(config)
        
        print(f"🎫 Processing ticket {args.ticket_id}...")
        filepath = agent.process_ticket(args.ticket_id)
        
        if filepath:
            print(f"✅ Tutorial generated: {filepath}")
            print(f"💡 Next step: Use 'ragdoc fetch' to parse this file into your knowledge base")
            return 0
        else:
            print(f"❌ Failed to generate tutorial for ticket {args.ticket_id}")
            print("💡 This might happen if the ticket has low quality content or isn't solved/closed")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def main(argv: list[str] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="ragdoc-ticket-agent",
        description="Process a Freshdesk ticket into tutorial markdown file"
    )
    
    # Global options
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--freshdesk-domain", help="Freshdesk domain (or set FRESHDESK_DOMAIN env var)")
    parser.add_argument("--freshdesk-api-key", help="Freshdesk API key (or set FRESHDESK_API_KEY env var)")
    parser.add_argument("--template", default="configs/tutorial_template.md", 
                       help="Template file path (default: configs/tutorial_template.md)")
    parser.add_argument("--output-dir", default="generated_tutorials",
                       help="Output directory for tutorials (default: generated_tutorials)")
    parser.add_argument("--translation-model", default="gpt-4o-mini",
                       help="Model for translation (default: gpt-4o-mini)")
    parser.add_argument("--analysis-model", default="gpt-4o-mini", 
                       help="Model for analysis (default: gpt-4o-mini)")
    
    # Process single ticket (no subcommand needed)
    parser.add_argument("ticket_id", type=int, help="Freshdesk ticket ID")
    
    args = parser.parse_args(argv)
    
    # Setup logging
    setup_logging(args.debug)
    
    # Execute ticket processing
    return handle_process_ticket(args)


if __name__ == "__main__":
    raise SystemExit(main())
