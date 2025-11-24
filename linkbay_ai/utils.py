from .core import AIOrchestrator
from typing import Dict, Any, List

async def generate_html_tailwind(ai: AIOrchestrator, description: str) -> str:
    """Genera HTML Tailwind usando il modello chat rapido."""
    prompt = f"""
    Genera codice HTML con Tailwind CSS per: {description}
    Restituisci SOLO il codice HTML, senza commenti o spiegazioni.
    Usa classi Tailwind per lo styling.
    """
    response = await ai.chat(prompt, model="deepseek-chat", use_conversation=False)
    return response.content

async def fill_form_fields(ai: AIOrchestrator, user_input: str, fields: List[str]) -> Dict[str, Any]:
    """Estrae valori chiave/valore dal testo utente come JSON."""
    prompt = f"""
    Dall'input utente: "{user_input}"
    Estrai i valori per questi campi: {', '.join(fields)}
    Restituisci SOLO un JSON con chiavi: {', '.join(fields)}
    Per campi non trovati, usa null.
    """
    response = await ai.chat(prompt, model="deepseek-chat", use_conversation=False)
    return _parse_json_response(response.content)

async def analyze_sales_data(ai: AIOrchestrator, csv_data: str) -> str:
    """Chiede insight sui dati di vendita forniti."""
    prompt = f"""
    Analizza questi dati di vendita e fornisci insights:
    {csv_data}
    
    Cerca:
    - Andamento vendite
    - Prodotti top
    - Suggerimenti miglioramento
    """
    response = await ai.chat(prompt, model="deepseek-reasoning", use_conversation=False)
    return response.content

async def analyze_traffic_data(ai: AIOrchestrator, log_data: str) -> str:
    """Analizza i log di traffico e restituisce considerazioni."""
    prompt = f"""
    Analizza questi dati di traffico:
    {log_data}
    
    Cerca:
    - Picchi di traffico
    - Pagine più visitate
    - Problemi prestazioni
    """
    response = await ai.chat(prompt, model="deepseek-reasoning", use_conversation=False)
    return response.content

def _parse_json_response(response: str) -> Dict[str, Any]:
    import json
    try:
        return json.loads(response.strip())
    except Exception:
        return {"error": "Failed to parse AI response as JSON", "raw": response.strip()}