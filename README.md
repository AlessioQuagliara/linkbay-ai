# 🤖 LinkBay-AI Enterprise v0.2.0

[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)]()
[![Status](https://img.shields.io/badge/status-beta-orange)]()

**AI orchestration library enterprise-ready per CMS LinkBay**

Libreria Python avanzata per orchestrazione AI con multi-provider fallback, budget control, semantic caching, streaming, conversation management, function calling e smart routing.

## ✨ Features Enterprise

### 🔴 Funzionalità Critiche
- ✅ **Multi-Provider Fallback**: DeepSeek → OpenAI → Local (zero downtime)
- ✅ **Cost Controller**: Budget tracking, rate limiting, alert automatici
- ✅ **Semantic Cache**: Cache intelligente basata su similarità semantica
- ✅ **Streaming Support**: Response streaming per UX ottimale

### 🟡 Funzionalità Importanti
- ✅ **Prompt Library**: 20+ templates riutilizzabili per task comuni
- ✅ **Conversation Context**: Gestione conversazioni multi-turn
- ✅ **Function Calling**: Tool use con 4+ tools predefiniti
- ✅ **Smart Routing**: Selezione automatica del modello ottimale

## 📋 Indice

- [Installazione](#installazione)
- [Quick Start](#quick-start)
- [Features Enterprise](#features-enterprise)
  - [Multi-Provider Fallback](#1-multi-provider-fallback)
  - [Cost Controller](#2-cost-controller)
  - [Semantic Cache](#3-semantic-cache)
  - [Streaming](#4-streaming)
  - [Prompt Library](#5-prompt-library)
  - [Conversation Context](#6-conversation-context)
  - [Function Calling](#7-function-calling)
  - [Smart Routing](#8-smart-routing)
- [Esempi Completi](#esempi)
- [Licenza](#licenza)

## 📦 Installazione

```bash
# Installazione base
pip install git+https://github.com/AlessioQuagliara/linkbay-ai.git

# Con semantic cache (opzionale ma consigliato)
pip install git+https://github.com/AlessioQuagliara/linkbay-ai.git[cache]
# oppure
pip install sentence-transformers
```

## ⚡ Quick Start

```python
import asyncio
from linkbay_ai import (
    AIOrchestrator,
    DeepSeekProvider,
    OpenAIProvider,
    ProviderConfig,
    BudgetConfig
)

async def main():
    # Configura orchestrator
    ai = AIOrchestrator(
        budget_config=BudgetConfig(max_tokens_per_hour=100000),
        enable_cache=True,
        enable_tools=True
    )
    
    # Registra provider con fallback
    deepseek_config = ProviderConfig(
        api_key="your-deepseek-key",
        base_url="https://api.deepseek.com",
        priority=1
    )
    openai_config = ProviderConfig(
        api_key="your-openai-key", 
        base_url="https://api.openai.com/v1",
        priority=2  # Fallback
    )
    
    ai.register_provider(DeepSeekProvider(deepseek_config), priority=1)
    ai.register_provider(OpenAIProvider(openai_config), priority=2)
    
    # Chat con tutte le features
    response = await ai.chat("Traduci 'Hello' in italiano")
    print(response.content)  # "Ciao"
    
    # Vedi analytics
    print(ai.get_analytics())

asyncio.run(main())
```

## Funzionalità
```bash
risposta = ai.chat("Spiega il machine learning")
```

# Compila il form 
```bash
campi = ["nome", "email", "telefono"]
dati = fill_form_fields(ai, "Mi chiamo Mario, cell 123456", campi)
# {'nome': 'Mario', 'email': None, 'telefono': '123456'}
```

# Genera HTML Tailwind
```bash
html = generate_html_tailwind(ai, "bottone rosso con testo bianco")
```
# Analisi Vendite
```bash
report = analyze_sales_data(ai, "vendite.csv content...")
```

# Analisi Traffico
```bash
insights = analyze_traffic_data(ai, "log traffico...")
```

## esempi

# Modello Reasoning (pensante)
```bash
analisi = ai.chat("Analizza questi dati...", model="deepseek-reasoning")
```

# Modello Chat (veloce)
```bash
risposta = ai.chat("Traduci questo testo", model="deepseek-chat")
```

# Esempio d'uso dentro al programma
```bash
from linkbay_ai import AIOrchestrator, DeepSeekProvider, ProviderConfig

config = ProviderConfig(
    api_key="sk-tua-chiave",
    base_url="https://api.deepseek.com"
)

ai = AIOrchestrator()
ai.register_provider("deepseek", DeepSeekProvider(config))

# Usa tutte le funzionalità!
html = generate_html_tailwind(ai, "navbar responsive")
```