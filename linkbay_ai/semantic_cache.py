"""
Semantic Cache - Cache risposte basata su similarità semantica
"""
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from .schemas import CacheEntry
import logging
import numpy as np

logger = logging.getLogger(__name__)

class SemanticCache:
    """
    Cache semantica per evitare chiamate duplicate
    Usa embeddings per trovare query simili
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.95,
                 max_entries: int = 1000,
                 ttl_hours: int = 24):
        """
        Args:
            similarity_threshold: Soglia di similarità (0-1)
            max_entries: Numero massimo di entry in cache
            ttl_hours: Time-to-live in ore
        """
        self.threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl = timedelta(hours=ttl_hours)
        self.cache: List[CacheEntry] = []
        
        # Lazy import per non rendere obbligatorio
        self.model = None
        self._embeddings_available = False
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self._embeddings_available = True
            logger.info("✅ Semantic cache attivata con sentence-transformers")
        except ImportError:
            logger.warning(
                "⚠️ sentence-transformers non disponibile. "
                "Cache semantica disabilitata. "
                "Installa con: pip install sentence-transformers"
            )
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Genera embedding per il testo"""
        if not self._embeddings_available:
            return None
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Errore generazione embedding: {e}")
            return None
    
    def _cosine_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calcola similarità coseno tra due embeddings"""
        a = np.array(emb1)
        b = np.array(emb2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def _cleanup_old_entries(self):
        """Rimuovi entry scadute"""
        now = datetime.now()
        self.cache = [
            entry for entry in self.cache
            if now - entry.timestamp < self.ttl
        ]
        
        # Se ancora troppo pieno, rimuovi i meno usati
        if len(self.cache) > self.max_entries:
            self.cache.sort(key=lambda x: x.hits, reverse=True)
            self.cache = self.cache[:self.max_entries]
    
    async def get_cached_response(self, query: str) -> Optional[str]:
        """
        Cerca una risposta cached per query simili
        
        Args:
            query: Query da cercare
            
        Returns:
            Risposta cached se trovata, altrimenti None
        """
        if not self._embeddings_available:
            return None
        
        self._cleanup_old_entries()
        
        # Genera embedding per la query
        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            return None
        
        # Cerca entry simili
        best_match = None
        best_similarity = 0.0
        
        for entry in self.cache:
            similarity = self._cosine_similarity(query_embedding, entry.embedding)
            
            if similarity > best_similarity and similarity >= self.threshold:
                best_similarity = similarity
                best_match = entry
        
        if best_match:
            # Aggiorna contatore hit
            best_match.hits += 1
            logger.info(
                f"🎯 Cache HIT! Similarità: {best_similarity:.3f} | "
                f"Hits totali: {best_match.hits}"
            )
            return best_match.response
        
        logger.debug(f"❌ Cache MISS (best similarity: {best_similarity:.3f})")
        return None
    
    async def cache_response(self, query: str, response: str):
        """
        Salva una risposta in cache
        
        Args:
            query: Query originale
            response: Risposta da cachare
        """
        if not self._embeddings_available:
            return
        
        embedding = self._get_embedding(query)
        if embedding is None:
            return
        
        entry = CacheEntry(
            query=query,
            embedding=embedding,
            response=response,
            timestamp=datetime.now(),
            hits=0
        )
        
        self.cache.append(entry)
        logger.debug(f"💾 Risposta cachata | Cache size: {len(self.cache)}")
    
    def get_stats(self) -> Dict:
        """Ottieni statistiche cache"""
        if not self.cache:
            return {
                "size": 0,
                "total_hits": 0,
                "avg_hits": 0,
                "enabled": self._embeddings_available
            }
        
        total_hits = sum(entry.hits for entry in self.cache)
        
        return {
            "size": len(self.cache),
            "max_size": self.max_entries,
            "total_hits": total_hits,
            "avg_hits": total_hits / len(self.cache),
            "threshold": self.threshold,
            "ttl_hours": self.ttl.total_seconds() / 3600,
            "enabled": self._embeddings_available
        }
    
    def clear_cache(self):
        """Svuota completamente la cache"""
        self.cache.clear()
        logger.info("🗑️ Cache svuotata")
