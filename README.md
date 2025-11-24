# LinkBay-AI 0.2.1

Un orchestratore **async-first** per integrare più LLM in modo semplice, senza architetture pesanti. L’obiettivo è essere pragmatici: feature utili, API coerenti, esempi pronti da copiare.

> **Principali capacità**
> - Fallback automatico tra provider (DeepSeek → OpenAI → Local mock)
> - Budget control (token & costi) con alert
> - Cache semantica opzionale basata su embeddings
> - Conversazioni multi-turn e tool/function calling
> - Streaming token-by-token

## Indice

1. [Installazione](#installazione)
2. [Quick start](#quick-start)
3. [Panoramica componenti](#panoramica-componenti)
4. [Ricette d'uso](#ricette-duso)
5. [Configurazione](#configurazione)
6. [Troubleshooting](#troubleshooting)
7. [FAQ performance e sicurezza](#faq-performance-e-sicurezza)
8. [Licenza](#licenza)

## Installazione

```bash
# Dipendenze core
pip install linkbay-ai

# Con cache semantica (scarica sentence-transformers, ~500 MB)
pip install "linkbay-ai[cache]"
```

| Requisito | Versione | Note |
|-----------|----------|------|
| Python    | 3.8 – 3.12 | testato in CI |
| openai    | >= 1.0.0  | SDK usato anche per DeepSeek |
| sentence-transformers *(opz.)* | >= 2.0.0 | richiesto solo se `enable_cache=True` |

## Quick start

```python
import asyncio
from linkbay_ai import (
    AIOrchestrator,
    DeepSeekProvider,
    OpenAIProvider,
    ProviderConfig,
    BudgetConfig,
)

async def main() -> None:
    ai = AIOrchestrator(
        budget_config=BudgetConfig(max_tokens_per_hour=100_000),
        enable_cache=True,
        enable_tools=False,
    )

    deepseek_cfg = ProviderConfig(
        api_key="DEEPSEEK_KEY",
        base_url="https://api.deepseek.com",
        priority=1,
    )
    openai_cfg = ProviderConfig(
        api_key="OPENAI_KEY",
        base_url="https://api.openai.com/v1",
        priority=2,
    )

    ai.register_provider(DeepSeekProvider(deepseek_cfg), priority=1)
    ai.register_provider(OpenAIProvider(openai_cfg), priority=2)

    response = await ai.chat("Traduci 'Hello world' in italiano")
    print(response.content)  # → "Ciao mondo"

    print(ai.get_analytics())

asyncio.run(main())
```

> **Importante:** l’API è completamente async. In ambienti sync usa `asyncio.run` oppure integra direttamente le coroutine in FastAPI, Quart, ecc.

## Panoramica componenti

| Componente | Ruolo | Default |
|------------|-------|---------|
| `AIOrchestrator` | Coordina provider, budget, cache, conversazioni, tools | — |
| `BaseProvider` + `{DeepSeek, OpenAI, Local}` | Adapter verso LLM esterni | DeepSeek + OpenAI |
| `CostController` | Tracking token/costo e alert soglia | 100 k token/h, $10/h, alert 80% |
| `SemanticCache` | Cache in-memory con embedding `all-MiniLM-L6-v2` | TTL 24h, max 1000 entry, soglia 0.95 |
| `ConversationContext` | History multi-turn con trimming | 10 messaggi, context window 4096 |
| `ToolsManager` | Function calling (4 tools mock + custom) | tools pre-registrati |
| `PromptLibrary` | 20+ template `string.Template` | HTML, analytics, business, reasoning |

## Ricette d'uso

### 1. Multi-provider fallback

```python
async def resilient_chat(prompt: str):
    orchestrator = AIOrchestrator()
    orchestrator.register_provider(DeepSeekProvider(primary_cfg), priority=1)
    orchestrator.register_provider(OpenAIProvider(backup_cfg), priority=2)

    return await orchestrator.chat(prompt)
```

- Provider provati in ordine di priorità (numero più basso = più importante).
- `max_retries` (default 3) gestisce i retry per singolo provider.
- Se tutti falliscono viene sollevata `AllProvidersFailedException`.

### 2. Budget control

```python
from linkbay_ai import BudgetConfig, BudgetExceededException

ai = AIOrchestrator(
    budget_config=BudgetConfig(
        max_tokens_per_hour=50_000,
        max_tokens_per_day=500_000,
        max_cost_per_hour=5.0,
    )
)

try:
    response = await ai.chat("Genera un business plan di 5 pagine")
except BudgetExceededException as exc:
    logger.warning("Budget superato: %s", exc)

print(ai.get_analytics()["budget"])  # Statistiche correnti
```

### 3. Cache semantica

```python
ai = AIOrchestrator(enable_cache=True)

first = await ai.chat("Cos'è l'e-commerce?")
second = await ai.chat("Che cosa si intende per commercio elettronico?")

print(first.cached, second.cached)  # False, True
```

Dettagli rapidi:
- Embedding model: `SentenceTransformer('all-MiniLM-L6-v2')`
- Soglia hit: `0.95` (configurabile)
- TTL: 24h, max 1000 entry, eviction per hit count
- Se `sentence-transformers` manca, la cache si disattiva automaticamente

### 4. Conversazioni multi-turn

```python
from linkbay_ai import ConversationConfig

ai = AIOrchestrator(conversation_config=ConversationConfig(max_messages=6))
ai.add_system_prompt("Sei un assistente e-commerce positivo")

await ai.chat("Che scarpe avete?", use_conversation=True)
reply = await ai.chat("Quali taglie sono disponibili?", use_conversation=True)

print(reply.content)
```

`get_analytics()["conversation"]` mostra conteggio messaggi e token utilizzati.

### 5. Streaming token-by-token

```python
async for chunk in ai.chat_stream("Scrivi un pitch di 100 parole", use_conversation=False):
    print(chunk, end="", flush=True)
```

### 6. Prompt library

```python
from linkbay_ai import PromptLibrary

prompt = PromptLibrary.generate_html("Card prodotto responsive con CTA")
html_response = await ai.chat(prompt, use_conversation=False)

prompt = PromptLibrary.analyze_sales("Prodotto A,100\nProdotto B,250")
analysis_response = await ai.chat(prompt, model="deepseek-reasoner")
```

Template principali:
- **Generici:** `SUMMARIZE`, `TRANSLATE`, `EXTRACT_KEYWORDS`
- **UI:** `GENERATE_HTML`, `GENERATE_COMPONENT`
- **Dati:** `ANALYZE_DATA`, `ANALYZE_SALES`, `ANALYZE_TRAFFIC`
- **Business:** `WRITE_EMAIL`, `GENERATE_DESCRIPTION`
- **Reasoning:** `DEEP_REASONING`, `DEBUG_CODE`

### 7. Tool / function calling

```python
from linkbay_ai import ToolCall

response = await ai.chat("Che meteo fa a Milano?", use_tools=True)
if response.tool_calls:
    meteo = await ai.tools_manager.execute_tool(ToolCall(**response.tool_calls[0]))
    print(meteo)
```

Registrare un tool custom:

```python
async def get_exchange_rate(base: str, quote: str) -> dict:
    return {"pair": f"{base}/{quote}", "rate": 1.08}

ai.tools_manager.register_tool(
    name="fx_rate",
    function=get_exchange_rate,
    description="Restituisce il tasso di cambio",
    parameters={
        "type": "object",
        "properties": {
            "base": {"type": "string"},
            "quote": {"type": "string"},
        },
        "required": ["base", "quote"],
    },
)
```

### 8. Helper leggeri

```python
from linkbay_ai.utils import generate_html_tailwind, fill_form_fields

html = await generate_html_tailwind(ai, "Navbar minimal con CTA")
form = await fill_form_fields(
    ai,
    "Mi chiamo Alessio, email alessio@example.com",
    ["nome", "email"],
)

print(form)  # {"nome": "Luca", "email": "luca@example.com"}
```

## Configurazione

### Provider

```python
ProviderConfig(
    api_key: str,
    base_url: str,
    default_model: str = "deepseek-chat",
    provider_type: Literal["deepseek", "openai", "local"],
    priority: int = 1,
    timeout: int = 30,
)
```

### Budget

```python
BudgetConfig(
    max_tokens_per_hour=100_000,
    max_tokens_per_day=1_000_000,
    max_cost_per_hour=10.0,
    alert_threshold=0.8,
)
```

### Conversazione

```python
ConversationConfig(
    max_messages=10,
    context_window=4096,
    summarize_old_messages=True,
)
```

### Generation params

Override puntuali passando `GenerationParams(model=..., max_tokens=..., temperature=...)` a `ai.chat(...)`.

## Troubleshooting

| Problema | Causa | Soluzione |
|----------|-------|-----------|
| `Client.__init__() got an unexpected keyword argument 'proxies'` | L'SDK OpenAI >=1.0 non accetta `proxies` | Configura proxy via variabili d'ambiente (`HTTP_PROXY`, `HTTPS_PROXY`) e rimuovi il parametro |
| `sentence-transformers` non trovato | Cache attivata senza dipendenza opzionale | Installa `linkbay-ai[cache]` oppure imposta `enable_cache=False` |
| `RuntimeError: Event loop is closed` | `asyncio.run` dentro un loop già attivo | Usa `await ai.chat` diretto (es. in FastAPI) |
| `BudgetExceededException` frequente | Limiti troppo bassi | Alza le soglie o abilita la cache per ridurre i token |

## FAQ performance e sicurezza

- **Cache & memoria:** 1000 entry con `MiniLM` ≈ 70 MB. Riduci `max_entries` se servono footprint minori.
- **Overhead embedding:** ~3–5 ms su CPU moderna. Per workload non ripetitivi disattiva la cache.
- **Persistenza cache:** al momento in-memory. Puoi derivare `SemanticCache` per usare Redis o DB.
- **Logging / metriche:** `AIOrchestrator` usa `logging`. Collega handler strutturati o esponi `get_analytics()` come endpoint.
- **Sicurezza input:** nessuna sanitizzazione automatica. Filtra prompt e gestisci PII lato applicazione.
- **Gestione chiavi:** conserva le API key in un secret manager (Vault, AWS SM). LinkBay-AI non salva né ruota credenziali.

## Licenza

MIT © Alessio Quagliara

## Supporto

- Issues: https://github.com/AlessioQuagliara/linkbay_ai/issues
- Email: quagliara.alessio@gmail.com

Contribuisci, apri una issue o raccontaci come stai usando la libreria 🧡