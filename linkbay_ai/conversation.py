"""
Conversation Context - Gestione conversazioni multi-turn
"""
from typing import List, Optional
from .schemas import Message, ConversationConfig
import logging

logger = logging.getLogger(__name__)

class ConversationContext:
    """
    Gestisce il contesto di una conversazione multi-turn
    Mantiene la history e gestisce il token budget
    """
    
    def __init__(self, config: Optional[ConversationConfig] = None):
        self.config = config or ConversationConfig()
        self.history: List[Message] = []
        self.total_tokens = 0
    
    def add_message(self, role: str, content: str, tokens: Optional[int] = None):
        """
        Aggiungi un messaggio alla conversazione
        
        Args:
            role: Ruolo (user/assistant/system)
            content: Contenuto del messaggio
            tokens: Numero di token (opzionale)
        """
        message = Message(role=role, content=content, tokens=tokens)
        self.history.append(message)
        
        if tokens:
            self.total_tokens += tokens
        
        # Gestisci overflow della finestra di contesto
        self._manage_context_window()
        
        logger.debug(f"💬 Messaggio aggiunto | History: {len(self.history)} messaggi")
    
    def _manage_context_window(self):
        """Gestisci il limite di messaggi e token"""
        # Rimuovi messaggi vecchi se troppi
        while len(self.history) > self.config.max_messages:
            removed = self.history.pop(0)
            if removed.tokens:
                self.total_tokens -= removed.tokens
            logger.debug(f"🗑️ Rimosso messaggio vecchio dalla history")
        
        # Se abilitato, riassumi messaggi vecchi quando vicini al limite
        if self.config.summarize_old_messages and len(self.history) > self.config.max_messages * 0.8:
            logger.info("📝 Potrebbe essere necessario riassumere messaggi vecchi")
    
    def get_messages(self, last_n: Optional[int] = None) -> List[Message]:
        """
        Ottieni i messaggi della conversazione
        
        Args:
            last_n: Se specificato, ritorna solo gli ultimi N messaggi
            
        Returns:
            Lista di messaggi
        """
        if last_n:
            return self.history[-last_n:]
        return self.history
    
    def get_context_for_api(self) -> List[dict]:
        """
        Ottieni i messaggi in formato API
        
        Returns:
            Lista di dict nel formato {role, content}
        """
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.history
        ]
    
    def clear(self):
        """Svuota la conversazione"""
        self.history.clear()
        self.total_tokens = 0
        logger.info("🗑️ Conversazione svuotata")
    
    def get_stats(self) -> dict:
        """Ottieni statistiche conversazione"""
        user_messages = sum(1 for m in self.history if m.role == "user")
        assistant_messages = sum(1 for m in self.history if m.role == "assistant")
        system_messages = sum(1 for m in self.history if m.role == "system")
        
        return {
            "total_messages": len(self.history),
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "system_messages": system_messages,
            "total_tokens": self.total_tokens,
            "max_messages": self.config.max_messages,
            "context_window": self.config.context_window
        }
    
    def add_system_prompt(self, prompt: str):
        """Aggiungi un system prompt all'inizio della conversazione"""
        system_msg = Message(role="system", content=prompt)
        self.history.insert(0, system_msg)
        logger.debug("🎯 System prompt aggiunto")
