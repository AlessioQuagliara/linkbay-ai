"""
Tools Manager - Function calling e tool execution
"""
from typing import Dict, Any, Callable, List, Optional
from .schemas import ToolCall
import logging
import json

logger = logging.getLogger(__name__)

class ToolExecutionError(Exception):
    """Errore durante l'esecuzione di un tool"""
    pass

class ToolsManager:
    """
    Gestisce la registrazione ed esecuzione di tools/functions
    per il function calling
    """
    
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.tool_definitions: List[Dict[str, Any]] = []
    
    def register_tool(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: Dict[str, Any]
    ):
        """
        Registra un nuovo tool
        
        Args:
            name: Nome del tool
            function: Funzione da eseguire
            description: Descrizione del tool
            parameters: JSON Schema dei parametri
        """
        self.tools[name] = function
        
        # Crea definizione in formato OpenAI
        tool_def = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }
        
        self.tool_definitions.append(tool_def)
        logger.info(f"🔧 Tool registrato: {name}")
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Ottieni le definizioni dei tools per l'API"""
        return self.tool_definitions
    
    async def execute_tool(self, tool_call: ToolCall) -> Any:
        """
        Esegui un tool call
        
        Args:
            tool_call: Tool call da eseguire
            
        Returns:
            Risultato dell'esecuzione
            
        Raises:
            ToolExecutionError: Se l'esecuzione fallisce
        """
        if tool_call.name not in self.tools:
            raise ToolExecutionError(f"Tool non trovato: {tool_call.name}")
        
        try:
            function = self.tools[tool_call.name]
            
            # Esegui la funzione
            logger.info(f"🔧 Esecuzione tool: {tool_call.name}")
            result = await function(**tool_call.arguments)
            
            logger.info(f"✅ Tool eseguito con successo: {tool_call.name}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Errore esecuzione tool {tool_call.name}: {e}")
            raise ToolExecutionError(f"Errore esecuzione {tool_call.name}: {str(e)}")
    
    def list_tools(self) -> List[str]:
        """Lista tutti i tool disponibili"""
        return list(self.tools.keys())


# ============= TOOLS PREDEFINITI =============

class CommonTools:
    """Collezione di tools comuni predefiniti"""
    
    @staticmethod
    async def search_products(query: str, category: Optional[str] = None, max_results: int = 10) -> List[Dict]:
        """
        Cerca prodotti nel catalogo
        
        Args:
            query: Query di ricerca
            category: Categoria (opzionale)
            max_results: Numero massimo risultati
            
        Returns:
            Lista di prodotti
        """
        # Implementazione mock - connetti al tuo database
        logger.info(f"🔍 Ricerca prodotti: {query} (categoria: {category})")
        return [
            {
                "id": 1,
                "name": f"Prodotto per {query}",
                "price": 99.99,
                "category": category or "generale"
            }
        ]
    
    @staticmethod
    async def get_weather(location: str) -> Dict[str, Any]:
        """
        Ottieni meteo per una località
        
        Args:
            location: Nome località
            
        Returns:
            Dati meteo
        """
        logger.info(f"🌤️ Richiesta meteo per: {location}")
        # Implementazione mock - usa API meteo reale
        return {
            "location": location,
            "temperature": 22,
            "condition": "Soleggiato",
            "humidity": 65
        }
    
    @staticmethod
    async def calculate(expression: str) -> float:
        """
        Calcola un'espressione matematica
        
        Args:
            expression: Espressione matematica
            
        Returns:
            Risultato del calcolo
        """
        logger.info(f"🔢 Calcolo: {expression}")
        try:
            # Usa eval in modo sicuro (per produzione usa una libreria dedicata)
            result = eval(expression, {"__builtins__": {}}, {})
            return float(result)
        except Exception as e:
            raise ValueError(f"Espressione non valida: {e}")
    
    @staticmethod
    async def get_user_info(user_id: str) -> Dict[str, Any]:
        """
        Ottieni informazioni utente
        
        Args:
            user_id: ID utente
            
        Returns:
            Dati utente
        """
        logger.info(f"👤 Richiesta info utente: {user_id}")
        # Implementazione mock - connetti al tuo database
        return {
            "user_id": user_id,
            "name": "Mario Rossi",
            "email": "mario.rossi@example.com",
            "subscription": "premium"
        }
    
    @staticmethod
    async def send_notification(user_id: str, message: str, channel: str = "email") -> bool:
        """
        Invia notifica a un utente
        
        Args:
            user_id: ID utente
            message: Messaggio da inviare
            channel: Canale (email/sms/push)
            
        Returns:
            True se inviato con successo
        """
        logger.info(f"📧 Invio notifica a {user_id} via {channel}")
        # Implementazione mock - usa servizio notifiche reale
        return True


def create_default_tools_manager() -> ToolsManager:
    """
    Crea un ToolsManager con i tools comuni preregistrati
    
    Returns:
        ToolsManager configurato
    """
    manager = ToolsManager()
    
    # Registra search_products
    manager.register_tool(
        name="search_products",
        function=CommonTools.search_products,
        description="Cerca prodotti nel catalogo e-commerce",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query di ricerca per i prodotti"
                },
                "category": {
                    "type": "string",
                    "description": "Categoria prodotti (opzionale)"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Numero massimo risultati",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    )
    
    # Registra get_weather
    manager.register_tool(
        name="get_weather",
        function=CommonTools.get_weather,
        description="Ottieni le previsioni meteo per una località",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Nome della località"
                }
            },
            "required": ["location"]
        }
    )
    
    # Registra calculate
    manager.register_tool(
        name="calculate",
        function=CommonTools.calculate,
        description="Calcola un'espressione matematica",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Espressione matematica da calcolare (es: '2 + 2 * 3')"
                }
            },
            "required": ["expression"]
        }
    )
    
    # Registra get_user_info
    manager.register_tool(
        name="get_user_info",
        function=CommonTools.get_user_info,
        description="Ottieni informazioni su un utente",
        parameters={
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "ID dell'utente"
                }
            },
            "required": ["user_id"]
        }
    )
    
    logger.info(f"✅ ToolsManager creato con {len(manager.list_tools())} tools")
    return manager
