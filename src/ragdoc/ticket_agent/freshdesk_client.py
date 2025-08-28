from __future__ import annotations

import logging
from typing import Any, Dict, Optional
import requests

# Logger setup
logger = logging.getLogger("ragdoc.ticket_agent.freshdesk")


class FreshdeskAPI:
    """Client for Freshdesk API interactions."""
    
    def __init__(self, domain: str, api_key: str):
        self.domain = domain.replace('.freshdesk.com', '')
        self.api_key = api_key
        self.base_url = f"https://{self.domain}.freshdesk.com/api/v2"
        self.session = requests.Session()
        self.session.auth = (api_key, 'X')
        
    def get_ticket(self, ticket_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a single ticket by ID."""
        try:
            response = self.session.get(f"{self.base_url}/tickets/{ticket_id}")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch ticket {ticket_id}: {e}")
            return None
    
    def get_conversations(self, ticket_id: int) -> list[Dict[str, Any]]:
        """Fetch conversations for a ticket."""
        try:
            response = self.session.get(f"{self.base_url}/tickets/{ticket_id}/conversations")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch conversations for ticket {ticket_id}: {e}")
            return []
