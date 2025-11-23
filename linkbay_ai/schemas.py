from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
from enum import Enum

class ProviderType(str, Enum):
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    LOCAL = "local"

class ProviderConfig(BaseModel):
    api_key: str
    base_url: str
    default_model: str = "deepseek-chat"
    provider_type: ProviderType = ProviderType.DEEPSEEK
    priority: int = 1  # Lower = higher priority
    timeout: int = 30

class GenerationParams(BaseModel):
    model: str
    max_tokens: int = 1000
    stream: bool = False
    temperature: float = 0.7
    tools: Optional[List[Dict[str, Any]]] = None

class AIRequest(BaseModel):
    prompt: str
    params: Optional[GenerationParams] = None
    context: Optional[Dict[str, Any]] = None

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    tokens: Optional[int] = None

class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

class AIResponse(BaseModel):
    content: str
    model: str
    provider: str
    tokens_used: int = 0
    cached: bool = False
    tool_calls: Optional[List[ToolCall]] = None

class BudgetConfig(BaseModel):
    max_tokens_per_hour: int = 100000
    max_tokens_per_day: int = 1000000
    max_cost_per_hour: float = 10.0  # USD
    alert_threshold: float = 0.8  # 80%

class CacheEntry(BaseModel):
    query: str
    embedding: List[float]
    response: str
    timestamp: datetime = Field(default_factory=datetime.now)
    hits: int = 0

class ConversationConfig(BaseModel):
    max_messages: int = 10
    context_window: int = 4096
    summarize_old_messages: bool = True