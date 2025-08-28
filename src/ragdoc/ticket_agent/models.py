from __future__ import annotations

import os
from dataclasses import dataclass


def _load_env_config():
    """Load environment variables from .env file if available."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not available, environment variables must be set manually


# Load .env configuration at module import
_load_env_config()


@dataclass
class TicketAgentConfig:
    """Configuration for the Freshdesk ticket processing agent."""
    
    # Freshdesk API settings
    freshdesk_domain: str = None
    freshdesk_api_key: str = None
    freshdesk_base_url: str = None

    # AI model settings
    translation_model: str = None
    analysis_model: str = None
    
    # Template and output settings
    template_file: str = None
    output_directory: str = None
    
    def __post_init__(self):
        """Load configuration from environment variables if not provided."""
        if self.freshdesk_domain is None:
            self.freshdesk_domain = os.getenv("FRESHDESK_DOMAIN")
        if self.freshdesk_api_key is None:
            self.freshdesk_api_key = os.getenv("FRESHDESK_API_KEY")
        if self.freshdesk_base_url is None:
            self.freshdesk_base_url = os.getenv("FRESHDESK_BASE_URL")
        if self.translation_model is None:
            self.translation_model = os.getenv("TICKET_AGENT_TRANSLATION_MODEL", "gpt-4o-mini")
        if self.analysis_model is None:
            self.analysis_model = os.getenv("TICKET_AGENT_ANALYSIS_MODEL", "gpt-4o-mini")
        if self.template_file is None:
            self.template_file = os.getenv("TICKET_AGENT_TEMPLATE", "configs/tutorial_template.md")
        if self.output_directory is None:
            self.output_directory = os.getenv("TICKET_AGENT_OUTPUT_DIR", "generated_tutorials")
        
        # Validate required fields
        if not self.freshdesk_domain:
            raise ValueError("FRESHDESK_DOMAIN must be provided via parameter or environment variable")
        if not self.freshdesk_api_key:
            raise ValueError("FRESHDESK_API_KEY must be provided via parameter or environment variable")
