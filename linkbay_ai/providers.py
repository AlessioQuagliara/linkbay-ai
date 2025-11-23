from openai import OpenAI
from .schemas import ProviderConfig, GenerationParams, AIResponse, Message, ProviderType
from typing import Dict, Any, List, AsyncIterator, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseProvider(ABC):
    """Base class per tutti i provider AI"""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.name = config.provider_type.value
    
    @abstractmethod
    async def chat(self, messages: List[Message], params: GenerationParams = None) -> AIResponse:
        """Esegui una chat completion"""
        pass
    
    @abstractmethod
    async def stream(self, messages: List[Message], params: GenerationParams = None) -> AsyncIterator[str]:
        """Esegui una chat completion con streaming"""
        pass
    
    def is_available(self) -> bool:
        """Verifica se il provider è disponibile"""
        try:
            return self._health_check()
        except:
            return False
    
    def _health_check(self) -> bool:
        """Health check specifico del provider"""
        return True

class DeepSeekProvider(BaseProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout
        )
    
    async def chat(self, messages: List[Message], params: GenerationParams = None) -> AIResponse:
        if params is None:
            params = GenerationParams(model=self.config.default_model)
        
        try:
            # Converti i messaggi nel formato OpenAI
            api_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            
            response = self.client.chat.completions.create(
                model=params.model,
                messages=api_messages,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                stream=False,
                tools=params.tools
            )
            
            # Gestisci tool calls se presenti
            tool_calls = None
            if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                tool_calls = [
                    {"name": tc.function.name, "arguments": tc.function.arguments}
                    for tc in response.choices[0].message.tool_calls
                ]
            
            return AIResponse(
                content=response.choices[0].message.content or "",
                model=params.model,
                provider=self.name,
                tokens_used=response.usage.total_tokens if response.usage else 0,
                tool_calls=tool_calls
            )
        except Exception as e:
            logger.error(f"DeepSeek API error: {str(e)}")
            raise Exception(f"DeepSeek API error: {str(e)}")
    
    async def stream(self, messages: List[Message], params: GenerationParams = None) -> AsyncIterator[str]:
        if params is None:
            params = GenerationParams(model=self.config.default_model)
        
        try:
            api_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            
            response = self.client.chat.completions.create(
                model=params.model,
                messages=api_messages,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"DeepSeek streaming error: {str(e)}")
            raise Exception(f"DeepSeek streaming error: {str(e)}")

class OpenAIProvider(BaseProvider):
    """OpenAI provider come fallback"""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = OpenAI(
            api_key=config.api_key,
            timeout=config.timeout
        )
    
    async def chat(self, messages: List[Message], params: GenerationParams = None) -> AIResponse:
        if params is None:
            params = GenerationParams(model="gpt-3.5-turbo")
        
        try:
            api_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            
            response = self.client.chat.completions.create(
                model=params.model,
                messages=api_messages,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                stream=False,
                tools=params.tools
            )
            
            return AIResponse(
                content=response.choices[0].message.content or "",
                model=params.model,
                provider=self.name,
                tokens_used=response.usage.total_tokens if response.usage else 0
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise Exception(f"OpenAI API error: {str(e)}")
    
    async def stream(self, messages: List[Message], params: GenerationParams = None) -> AsyncIterator[str]:
        if params is None:
            params = GenerationParams(model="gpt-3.5-turbo")
        
        try:
            api_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            
            response = self.client.chat.completions.create(
                model=params.model,
                messages=api_messages,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI streaming error: {str(e)}")
            raise Exception(f"OpenAI streaming error: {str(e)}")

class LocalProvider(BaseProvider):
    """Local provider fallback (mock per ora)"""
    
    async def chat(self, messages: List[Message], params: GenerationParams = None) -> AIResponse:
        # Implementazione mock - potrebbe usare Ollama o modello locale
        return AIResponse(
            content="Local provider non ancora implementato. Tutti i provider remoti sono down.",
            model="local",
            provider=self.name,
            tokens_used=0
        )
    
    async def stream(self, messages: List[Message], params: GenerationParams = None) -> AsyncIterator[str]:
        yield "Local provider non ancora implementato."