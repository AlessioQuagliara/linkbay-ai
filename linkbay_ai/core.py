from .providers import BaseProvider, DeepSeekProvider, OpenAIProvider, LocalProvider
from .schemas import (
    ProviderConfig, GenerationParams, AIRequest, AIResponse, 
    Message, BudgetConfig, ConversationConfig, ToolCall
)
from .cost_controller import CostController, BudgetExceededException
from .semantic_cache import SemanticCache
from .conversation import ConversationContext
from .tools import ToolsManager, create_default_tools_manager
from typing import Dict, Any, List, Optional, AsyncIterator
import logging

logger = logging.getLogger(__name__)

class AllProvidersFailedException(Exception):
    """Eccezione quando tutti i provider falliscono"""
    pass

class AIOrchestrator:
    """
    Orchestratore AI enterprise-ready con:
    - Multi-provider con fallback automatico
    - Budget tracking e rate limiting
    - Semantic caching
    - Streaming support
    - Conversation context
    - Tool/function calling
    - Smart routing
    """
    
    def __init__(
        self,
        budget_config: Optional[BudgetConfig] = None,
        conversation_config: Optional[ConversationConfig] = None,
        enable_cache: bool = True,
        enable_tools: bool = True
    ):
        # Provider management
        self.providers: List[BaseProvider] = []
        self.provider_priority: Dict[str, int] = {}
        
        # Feature modules
        self.cost_controller = CostController(budget_config or BudgetConfig())
        self.semantic_cache = SemanticCache() if enable_cache else None
        self.conversation = ConversationContext(conversation_config)
        self.tools_manager = create_default_tools_manager() if enable_tools else None
        
        # Analytics
        self.request_history: List[Dict] = []
        self.provider_stats: Dict[str, Dict[str, int]] = {}
        
        logger.info("🚀 AIOrchestrator inizializzato")
    
    def register_provider(self, provider: BaseProvider, priority: int = 99):
        """
        Registra un provider con priorità
        
        Args:
            provider: Provider da registrare
            priority: Priorità (più basso = più alto)
        """
        self.providers.append(provider)
        self.provider_priority[provider.name] = priority
        
        # Ordina per priorità
        self.providers.sort(key=lambda p: self.provider_priority[p.name])
        
        # Inizializza stats
        if provider.name not in self.provider_stats:
            self.provider_stats[provider.name] = {
                "requests": 0,
                "successes": 0,
                "failures": 0
            }
        
        logger.info(f"✅ Provider registrato: {provider.name} (priorità: {priority})")
    
    async def chat(
        self,
        prompt: str,
        model: Optional[str] = None,
        use_conversation: bool = True,
        use_cache: bool = True,
        use_tools: bool = False,
        max_retries: int = 3
    ) -> AIResponse:
        """
        Esegui una chat completion con tutte le features enterprise
        
        Args:
            prompt: Prompt dell'utente
            model: Modello specifico (opzionale, usa smart routing)
            use_conversation: Usa conversation context
            use_cache: Usa semantic cache
            use_tools: Abilita function calling
            max_retries: Numero massimo di tentativi
            
        Returns:
            AIResponse con la risposta
        """
        # 1. Controlla cache
        if use_cache and self.semantic_cache:
            cached = await self.semantic_cache.get_cached_response(prompt)
            if cached:
                return AIResponse(
                    content=cached,
                    model="cached",
                    provider="cache",
                    cached=True
                )
        
        # 2. Aggiungi a conversation context
        if use_conversation:
            self.conversation.add_message("user", prompt)
            messages = self.conversation.get_messages()
        else:
            messages = [Message(role="user", content=prompt)]
        
        # 3. Smart routing del modello
        if not model:
            model = self._smart_route_model(prompt)
        
        # 4. Stima token e controlla budget
        estimated_tokens = len(prompt.split()) * 2  # Stima approssimativa
        try:
            await self.cost_controller.check_budget(estimated_tokens, model)
        except BudgetExceededException as e:
            logger.error(f"💰 {str(e)}")
            raise
        
        # 5. Prepara parametri
        params = GenerationParams(model=model)
        if use_tools and self.tools_manager:
            params.tools = self.tools_manager.get_tool_definitions()
        
        # 6. Esegui con fallback multi-provider
        response = await self._execute_with_fallback(messages, params, max_retries)
        
        # 7. Gestisci tool calls se presenti
        if response.tool_calls and self.tools_manager:
            tool_result = await self._handle_tool_calls(response.tool_calls)
            # Aggiungi risultato tool e richiama API
            self.conversation.add_message("assistant", response.content or "")
            self.conversation.add_message("system", f"Tool result: {tool_result}")
            messages = self.conversation.get_messages()
            response = await self._execute_with_fallback(messages, params, max_retries)
        
        # 8. Aggiorna conversazione
        if use_conversation:
            self.conversation.add_message(
                "assistant",
                response.content,
                tokens=response.tokens_used
            )
        
        # 9. Salva in cache
        if use_cache and self.semantic_cache and not response.cached:
            await self.semantic_cache.cache_response(prompt, response.content)
        
        # 10. Registra utilizzo
        self.cost_controller.record_usage(response.tokens_used, model)
        
        # 11. Analytics
        self.request_history.append({
            "prompt": prompt,
            "model": model,
            "provider": response.provider,
            "tokens": response.tokens_used,
            "cached": response.cached
        })
        
        return response
    
    async def chat_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        use_conversation: bool = True
    ) -> AsyncIterator[str]:
        """
        Chat con streaming per UX migliore
        
        Args:
            prompt: Prompt dell'utente
            model: Modello specifico
            use_conversation: Usa conversation context
            
        Yields:
            Chunk di risposta
        """
        if use_conversation:
            self.conversation.add_message("user", prompt)
            messages = self.conversation.get_messages()
        else:
            messages = [Message(role="user", content=prompt)]
        
        if not model:
            model = self._smart_route_model(prompt)
        
        params = GenerationParams(model=model)
        
        # Trova provider disponibile
        for provider in self.providers:
            if not provider.is_available():
                continue
            
            try:
                full_response = []
                async for chunk in provider.stream(messages, params):
                    full_response.append(chunk)
                    yield chunk
                
                # Aggiorna conversazione con risposta completa
                if use_conversation:
                    self.conversation.add_message("assistant", "".join(full_response))
                
                return
                
            except Exception as e:
                logger.warning(f"⚠️ {provider.name} streaming failed: {e}")
                continue
        
        raise AllProvidersFailedException("Tutti i provider hanno fallito")
    
    async def _execute_with_fallback(
        self,
        messages: List[Message],
        params: GenerationParams,
        max_retries: int
    ) -> AIResponse:
        """Esegui con fallback automatico tra provider"""
        
        if not self.providers:
            raise ValueError("Nessun provider configurato")
        
        for provider in self.providers:
            # Salta provider non disponibili
            if not provider.is_available():
                logger.warning(f"⚠️ {provider.name} non disponibile, skip")
                continue
            
            stats = self.provider_stats[provider.name]
            
            for attempt in range(max_retries):
                try:
                    stats["requests"] += 1
                    logger.info(f"🔄 Tentativo {attempt + 1}/{max_retries} con {provider.name}")
                    
                    response = await provider.chat(messages, params)
                    
                    stats["successes"] += 1
                    logger.info(f"✅ Successo con {provider.name}")
                    
                    return response
                    
                except Exception as e:
                    stats["failures"] += 1
                    logger.warning(f"❌ {provider.name} fallito (tentativo {attempt + 1}): {e}")
                    
                    if attempt < max_retries - 1:
                        continue
                    else:
                        logger.error(f"💥 {provider.name} fallito dopo {max_retries} tentativi")
                        break
        
        # Tutti i provider hanno fallito
        raise AllProvidersFailedException(
            "Tutti i provider AI hanno fallito. Verifica la connessione e le API keys."
        )
    
    async def _handle_tool_calls(self, tool_calls: List[Dict]) -> Any:
        """Gestisci tool calls"""
        if not self.tools_manager:
            return None
        
        results = []
        for tc in tool_calls:
            tool_call = ToolCall(name=tc["name"], arguments=tc["arguments"])
            result = await self.tools_manager.execute_tool(tool_call)
            results.append(result)
        
        return results
    
    def _smart_route_model(self, prompt: str) -> str:
        """
        Smart routing: seleziona il modello migliore in base al prompt
        
        Args:
            prompt: Prompt da analizzare
            
        Returns:
            Nome del modello da usare
        """
        prompt_lower = prompt.lower()
        
        # Task che richiedono reasoning
        reasoning_keywords = [
            "analizza", "ragiona", "spiega perché", "confronta",
            "valuta", "quale è meglio", "pro e contro", "calcola",
            "risolvi", "dimostra", "deduce"
        ]
        
        if any(keyword in prompt_lower for keyword in reasoning_keywords):
            logger.info("🧠 Smart routing: deepseek-reasoner (reasoning richiesto)")
            return "deepseek-reasoner"
        
        # Task semplici
        simple_keywords = [
            "traduci", "riassumi", "lista", "elenca",
            "genera html", "formato json"
        ]
        
        if any(keyword in prompt_lower for keyword in simple_keywords):
            logger.info("⚡ Smart routing: deepseek-chat (task semplice)")
            return "deepseek-chat"
        
        # Default: chat model
        logger.info("💬 Smart routing: deepseek-chat (default)")
        return "deepseek-chat"
    
    def get_analytics(self) -> Dict[str, Any]:
        """Ottieni analytics complete"""
        return {
            "total_requests": len(self.request_history),
            "providers": self.provider_stats,
            "budget": self.cost_controller.get_current_usage(),
            "cache": self.semantic_cache.get_stats() if self.semantic_cache else None,
            "conversation": self.conversation.get_stats(),
            "tools": self.tools_manager.list_tools() if self.tools_manager else []
        }
    
    def reset_conversation(self):
        """Reset conversazione"""
        self.conversation.clear()
    
    def add_system_prompt(self, prompt: str):
        """Aggiungi system prompt"""
        self.conversation.add_system_prompt(prompt)