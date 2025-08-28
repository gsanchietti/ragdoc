"""Ticket Agent subpackage public API."""

from .cli import main as ticket_agent_main
from .models import TicketAgentConfig
from .processor import TicketProcessor, create_ticket_agent
from .freshdesk_client import FreshdeskAPI

__all__ = [
    "ticket_agent_main",
    "TicketAgentConfig",
    "TicketProcessor",
    "create_ticket_agent", 
    "FreshdeskAPI"
]
