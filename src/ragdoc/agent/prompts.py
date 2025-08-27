from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path


def load_yaml_config() -> Dict[str, Any]:
    """Load prompts configuration from YAML file."""
    config_path = Path(__file__).parent.parent.parent.parent / "configs" / "prompts.yaml"
    if not config_path.exists():
        # Fallback to package directory
        config_path = Path(__file__).parent / "prompts.yaml"
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    else:
        # Return empty config if file doesn't exist
        return {}


@dataclass
class AgentPrompts:
    """Configuration class for all agent prompts and responses loaded from YAML."""
    
    # Load configuration from YAML
    _config: Dict[str, Any] = field(default_factory=load_yaml_config)
    
    def _get_config_value(self, section: str, key: str, language: str = "", fallback: str = "") -> str:
        """Get configuration value with environment variable override support."""
        # Check environment variable first
        if language:
            env_key = f"RAGDOC_{section.upper()}_{key.upper()}_{language.upper()}"
        else:
            env_key = f"RAGDOC_{section.upper()}_{key.upper()}"
        
        env_value = os.getenv(env_key)
        if env_value:
            return env_value
        
        # Get from YAML config
        try:
            if language:
                return self._config.get(section, {}).get(f"{key}_{language}", fallback)
            else:
                return self._config.get(section, {}).get(key, fallback)
        except (KeyError, AttributeError):
            return fallback
    
    @classmethod
    def from_env(cls) -> AgentPrompts:
        """Create AgentPrompts instance with environment variable overrides."""
        return cls()
    
    def get_system_prompt(self, language: str = "it") -> str:
        """Get system prompt for the specified language."""
        return self._get_config_value("system", "prompt", language, 
            "You are ragdoc, a support assistant." if language == "en" 
            else "Sei ragdoc, un assistente di supporto.")
    
    def get_summarization_system_prompt(self, language: str = "it") -> str:
        """Get summarization system prompt for the specified language."""
        return self._get_config_value("summarization", "system_prompt", language,
            "Analyze the conversation and summarize key points." if language == "en"
            else "Analizza la conversazione e riassumi i punti chiave.")
    
    def get_summarization_user_prompt(self, language: str = "it") -> str:
        """Get summarization user prompt for the specified language."""
        return self._get_config_value("summarization", "user_prompt", language,
            "Conversation to summarize:\n{conversation}\n\nProvide a summary." if language == "en"
            else "Conversazione da riassumere:\n{conversation}\n\nFornisci un riassunto.")
    
    def get_answer_instruction(self, language: str = "it", refs_text: str = "") -> str:
        """Get answer instruction for the specified language with references formatted."""
        template = self._get_config_value("answer", "instruction", language,
            "Use the following excerpts to answer. Include a 'References' section with the URLs used.\n\nReferences:\n{refs_text}\n\nAnswer now." if language == "en"
            else "Usa i seguenti estratti per rispondere. Includi una sezione 'Riferimenti' con le URL usate.\n\nRiferimenti:\n{refs_text}\n\nRispondi ora.")
        
        # Format the template with refs_text
        return template.format(refs_text=refs_text)
    
    def get_no_sources_found(self, language: str = "it") -> str:
        """Get no sources found message for the specified language."""
        return self._get_config_value("answer", "no_sources_found", language,
            "- (no sources found)" if language == "en"
            else "- (nessuna fonte trovata)")
    
    def get_placeholder_title(self, language: str = "it") -> str:
        """Get placeholder title for the specified language."""
        return self._get_config_value("placeholder", "title", language,
            "Details required" if language == "en"
            else "Dettagli richiesti")
    
    def get_placeholder_preview(self, language: str = "it") -> str:
        """Get placeholder preview for the specified language."""
        return self._get_config_value("placeholder", "preview", language,
            "Provide more details to improve the search." if language == "en"
            else "Fornisci maggiori dettagli per migliorare la ricerca.")
    
    def get_escalate_message(self, language: str = "it", attempts: Optional[int] = None, 
                           confidence: Optional[float] = None, keywords: Optional[list] = None) -> str:
        """Get escalation message for the specified language."""
        base_msg = self._get_config_value("escalation", "base", language,
            "I'm escalating this case to a human colleague." if language == "en"
            else "Sto passando il caso a un collega umano.")
        
        if attempts is not None:
            attempts_msg = self._get_config_value("escalation", "attempts", language,
                f"\n\nRefinement attempts made: {attempts}" if language == "en"
                else f"\n\nTentativi di raffinamento effettuati: {attempts}")
            base_msg += attempts_msg.format(attempts=attempts)
        
        if confidence is not None:
            confidence_msg = self._get_config_value("escalation", "confidence", language,
                f"\n\nBest confidence achieved: {confidence:.2f}" if language == "en"
                else f"\n\nMigliore confidenza raggiunta: {confidence:.2f}")
            base_msg += confidence_msg.format(confidence=confidence)
        
        if keywords:
            keywords_str = ", ".join(keywords)
            keywords_msg = self._get_config_value("escalation", "keywords", language,
                f"\n\nKey terms identified: {keywords_str}" if language == "en"
                else f"\n\nTermini chiave identificati: {keywords_str}")
            base_msg += keywords_msg.format(keywords=keywords_str)
        
        return base_msg
    
    def get_question(self, question_type: str, language: str = "it") -> str:
        """Get clarifying question for the specified type and language."""
        fallback = f"Can you be more specific?" if language == "en" else "Puoi essere più specifico?"
        return self._get_config_value("questions", f"{question_type}", language, fallback)


# Global instance
_agent_prompts = None

def get_agent_prompts() -> AgentPrompts:
    """Get the global agent prompts configuration instance."""
    global _agent_prompts
    if _agent_prompts is None:
        _agent_prompts = AgentPrompts.from_env()
    return _agent_prompts


def reload_agent_prompts() -> AgentPrompts:
    """Reload agent prompts configuration from environment variables."""
    global _agent_prompts
    _agent_prompts = AgentPrompts.from_env()
    return _agent_prompts
