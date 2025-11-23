"""
LinkBay-AI - Enterprise-ready AI orchestration library
"""

from .core import AIOrchestrator, AllProvidersFailedException
from .providers import (
    BaseProvider,
    DeepSeekProvider, 
    OpenAIProvider, 
    LocalProvider
)
from .schemas import (
    ProviderConfig,
    ProviderType,
    GenerationParams,
    AIRequest,
    AIResponse,
    Message,
    BudgetConfig,
    ConversationConfig,
    ToolCall
)
from .cost_controller import CostController, BudgetExceededException
from .semantic_cache import SemanticCache
from .conversation import ConversationContext
from .tools import ToolsManager, CommonTools, create_default_tools_manager
from .prompt_library import PromptLibrary
from .utils import (
    generate_html_tailwind,
    analyze_sales_data,
    analyze_traffic_data,
    fill_form_fields
)

__version__ = "0.2.0"  # Bump version per nuove features enterprise

__all__ = [
    # Core
    "AIOrchestrator",
    "AllProvidersFailedException",
    
    # Providers
    "BaseProvider",
    "DeepSeekProvider",
    "OpenAIProvider",
    "LocalProvider",
    
    # Schemas
    "ProviderConfig",
    "ProviderType",
    "GenerationParams",
    "AIRequest",
    "AIResponse",
    "Message",
    "BudgetConfig",
    "ConversationConfig",
    "ToolCall",
    
    # Features
    "CostController",
    "BudgetExceededException",
    "SemanticCache",
    "ConversationContext",
    "ToolsManager",
    "CommonTools",
    "create_default_tools_manager",
    "PromptLibrary",
    
    # Utils (legacy)
    "generate_html_tailwind",
    "analyze_sales_data", 
    "analyze_traffic_data",
    "fill_form_fields"
]