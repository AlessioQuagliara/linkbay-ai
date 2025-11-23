"""
Prompt Library - Templates riutilizzabili per task comuni
"""
from typing import Dict, Any, Optional
from string import Template

class PromptLibrary:
    """
    Libreria di prompt templates per task comuni
    Usa Python string templates per parametrizzazione
    """
    
    # ============= TASK GENERICI =============
    
    SUMMARIZE = Template("""
Riassumi il seguente testo in modo conciso e chiaro.
Mantieni i punti chiave e le informazioni più importanti.

Testo:
$text

Riassunto:
""")
    
    TRANSLATE = Template("""
Traduci il seguente testo in $lang.
Mantieni il tono e lo stile del testo originale.

Testo:
$text

Traduzione:
""")
    
    EXTRACT_KEYWORDS = Template("""
Estrai le parole chiave più importanti dal seguente testo.
Fornisci 5-10 keywords separate da virgole.

Testo:
$text

Keywords:
""")
    
    # ============= GENERAZIONE HTML/UI =============
    
    GENERATE_HTML = Template("""
Genera codice HTML5 moderno con Tailwind CSS per il seguente requisito:

$description

Requisiti:
- Usa Tailwind CSS per lo styling
- Codice responsive e accessibile
- Includi commenti per sezioni complesse
- Struttura semantica HTML5

Restituisci SOLO il codice HTML, senza spiegazioni.
""")
    
    GENERATE_COMPONENT = Template("""
Genera un componente $framework per:

$description

Requisiti:
- Componente riutilizzabile e modulare
- Props/parametri tipizzati
- Stile: $style
- Responsive design

Restituisci il codice del componente.
""")
    
    # ============= ANALISI DATI =============
    
    ANALYZE_DATA = Template("""
Analizza i seguenti dati e fornisci insights dettagliati:

$data

Fornisci:
1. Trend e pattern principali
2. Anomalie o outlier
3. Raccomandazioni basate sui dati
4. Metriche chiave

Analisi:
""")
    
    ANALYZE_SALES = Template("""
Analizza questi dati di vendita:

$data

Fornisci:
1. Andamento vendite nel periodo
2. Prodotti/categorie top performer
3. Trend stagionali
4. Suggerimenti per migliorare le vendite

Analisi:
""")
    
    ANALYZE_TRAFFIC = Template("""
Analizza questi dati di traffico web:

$data

Fornisci:
1. Picchi e pattern di traffico
2. Pagine più visitate
3. Problemi di performance rilevati
4. Opportunità di ottimizzazione

Analisi:
""")
    
    # ============= FORM PROCESSING =============
    
    EXTRACT_FORM_DATA = Template("""
Dall'input dell'utente, estrai i valori per questi campi del form:

Campi richiesti: $fields

Input utente:
"$user_input"

Restituisci SOLO un JSON valido con le chiavi: $fields
Per campi non trovati nell'input, usa null.
Non includere spiegazioni, solo il JSON.

JSON:
""")
    
    VALIDATE_FORM_DATA = Template("""
Valida i seguenti dati del form secondo le regole specificate:

Dati: $data
Regole: $rules

Restituisci un JSON con:
{
  "valid": true/false,
  "errors": ["errore1", "errore2", ...],
  "suggestions": ["suggerimento1", ...]
}
""")
    
    # ============= CODE GENERATION =============
    
    GENERATE_API_ENDPOINT = Template("""
Genera un endpoint FastAPI per:

$description

Requisiti:
- Framework: FastAPI
- Validazione input con Pydantic
- Gestione errori appropriata
- Documentazione docstring
- Type hints completi

Linguaggio: $language

Codice:
""")
    
    GENERATE_SQL_QUERY = Template("""
Genera una query SQL per:

$description

Database: $database
Tabelle disponibili: $tables

Requisiti:
- Query ottimizzata
- Include commenti
- Gestione NULL values
- Use prepared statement parameters where needed

Query SQL:
""")
    
    # ============= BUSINESS/CONTENT =============
    
    WRITE_EMAIL = Template("""
Scrivi una email professionale per:

Contesto: $context
Destinatario: $recipient
Tono: $tone

Email:
""")
    
    GENERATE_DESCRIPTION = Template("""
Genera una descrizione $type per:

$product_name

Caratteristiche:
$features

Requisiti:
- Lunghezza: $length parole
- Tono: $tone
- Include call-to-action
- Ottimizzata SEO

Descrizione:
""")
    
    # ============= REASONING =============
    
    DEEP_REASONING = Template("""
Analizza approfonditamente il seguente problema usando ragionamento step-by-step:

Problema:
$problem

Procedi così:
1. Analizza il problema e identifica i componenti chiave
2. Esplora possibili approcci e soluzioni
3. Valuta pro/contro di ogni soluzione
4. Fornisci una raccomandazione finale con giustificazione

Ragionamento:
""")
    
    DEBUG_CODE = Template("""
Analizza questo codice e identifica bug o problemi:

Linguaggio: $language

Codice:
$code

Errore riportato: $error

Fornisci:
1. Causa del problema
2. Fix suggerito con codice
3. Best practices per evitare problemi simili

Analisi:
""")
    
    # ============= HELPER METHODS =============
    
    @classmethod
    def render(cls, template: Template, **kwargs) -> str:
        """
        Rendi un template con i parametri forniti
        
        Args:
            template: Template da rendere
            **kwargs: Parametri per il template
            
        Returns:
            Prompt renderizzato
        """
        try:
            return template.substitute(**kwargs)
        except KeyError as e:
            raise ValueError(f"Parametro mancante nel template: {e}")
    
    @classmethod
    def summarize(cls, text: str) -> str:
        """Helper per riassumere testo"""
        return cls.render(cls.SUMMARIZE, text=text)
    
    @classmethod
    def translate(cls, text: str, lang: str) -> str:
        """Helper per tradurre testo"""
        return cls.render(cls.TRANSLATE, text=text, lang=lang)
    
    @classmethod
    def generate_html(cls, description: str) -> str:
        """Helper per generare HTML"""
        return cls.render(cls.GENERATE_HTML, description=description)
    
    @classmethod
    def extract_form_data(cls, user_input: str, fields: list) -> str:
        """Helper per estrarre dati form"""
        fields_str = ", ".join(fields)
        return cls.render(
            cls.EXTRACT_FORM_DATA,
            user_input=user_input,
            fields=fields_str
        )
    
    @classmethod
    def analyze_sales(cls, data: str) -> str:
        """Helper per analizzare vendite"""
        return cls.render(cls.ANALYZE_SALES, data=data)
    
    @classmethod
    def analyze_traffic(cls, data: str) -> str:
        """Helper per analizzare traffico"""
        return cls.render(cls.ANALYZE_TRAFFIC, data=data)
    
    @classmethod
    def deep_reasoning(cls, problem: str) -> str:
        """Helper per deep reasoning"""
        return cls.render(cls.DEEP_REASONING, problem=problem)
