"""
Cost Controller - Budget tracking e rate limiting
"""
from typing import Dict, Optional
from datetime import datetime, timedelta
from .schemas import BudgetConfig
import logging

logger = logging.getLogger(__name__)

class BudgetExceededException(Exception):
    """Eccezione quando il budget viene superato"""
    pass

class CostController:
    """Controlla i costi e limita l'utilizzo dei token"""
    
    def __init__(self, config: BudgetConfig):
        self.config = config
        self.hourly_usage: Dict[str, int] = {}  # timestamp_hour -> tokens
        self.daily_usage: Dict[str, int] = {}   # timestamp_day -> tokens
        self.hourly_cost: Dict[str, float] = {}
        
        # Prezzi per token (esempio - aggiusta in base al provider)
        self.token_costs = {
            "deepseek-chat": 0.14 / 1_000_000,      # $0.14 per 1M tokens
            "deepseek-reasoner": 0.55 / 1_000_000,  # $0.55 per 1M tokens
            "gpt-3.5-turbo": 0.5 / 1_000_000,
            "gpt-4": 30.0 / 1_000_000
        }
    
    def _get_hour_key(self) -> str:
        """Ottieni chiave per l'ora corrente"""
        return datetime.now().strftime("%Y-%m-%d-%H")
    
    def _get_day_key(self) -> str:
        """Ottieni chiave per il giorno corrente"""
        return datetime.now().strftime("%Y-%m-%d")
    
    def _cleanup_old_entries(self):
        """Rimuovi entry vecchie per risparmiare memoria"""
        now = datetime.now()
        hour_threshold = (now - timedelta(hours=2)).strftime("%Y-%m-%d-%H")
        day_threshold = (now - timedelta(days=2)).strftime("%Y-%m-%d")
        
        # Cleanup hourly
        self.hourly_usage = {k: v for k, v in self.hourly_usage.items() if k > hour_threshold}
        self.hourly_cost = {k: v for k, v in self.hourly_cost.items() if k > hour_threshold}
        
        # Cleanup daily
        self.daily_usage = {k: v for k, v in self.daily_usage.items() if k > day_threshold}
    
    async def check_budget(self, estimated_tokens: int, model: str = "deepseek-chat") -> bool:
        """
        Verifica se c'è budget disponibile
        
        Args:
            estimated_tokens: Token stimati per la richiesta
            model: Modello da utilizzare
            
        Returns:
            True se c'è budget disponibile
            
        Raises:
            BudgetExceededException: Se il budget è stato superato
        """
        self._cleanup_old_entries()
        
        hour_key = self._get_hour_key()
        day_key = self._get_day_key()
        
        # Controlla token orari
        current_hourly = self.hourly_usage.get(hour_key, 0)
        if current_hourly + estimated_tokens > self.config.max_tokens_per_hour:
            raise BudgetExceededException(
                f"Budget orario superato: {current_hourly + estimated_tokens} / {self.config.max_tokens_per_hour}"
            )
        
        # Controlla token giornalieri
        current_daily = self.daily_usage.get(day_key, 0)
        if current_daily + estimated_tokens > self.config.max_tokens_per_day:
            raise BudgetExceededException(
                f"Budget giornaliero superato: {current_daily + estimated_tokens} / {self.config.max_tokens_per_day}"
            )
        
        # Controlla costi orari
        estimated_cost = estimated_tokens * self.token_costs.get(model, 0.001)
        current_cost = self.hourly_cost.get(hour_key, 0.0)
        if current_cost + estimated_cost > self.config.max_cost_per_hour:
            raise BudgetExceededException(
                f"Budget costi orario superato: ${current_cost + estimated_cost:.4f} / ${self.config.max_cost_per_hour}"
            )
        
        # Alert se vicino al limite
        hourly_percent = (current_hourly + estimated_tokens) / self.config.max_tokens_per_hour
        if hourly_percent > self.config.alert_threshold:
            logger.warning(
                f"⚠️ Budget alert: utilizzo al {hourly_percent * 100:.1f}% del limite orario"
            )
        
        return True
    
    def record_usage(self, tokens_used: int, model: str = "deepseek-chat"):
        """
        Registra l'utilizzo effettivo
        
        Args:
            tokens_used: Token utilizzati
            model: Modello utilizzato
        """
        hour_key = self._get_hour_key()
        day_key = self._get_day_key()
        
        # Aggiorna contatori
        self.hourly_usage[hour_key] = self.hourly_usage.get(hour_key, 0) + tokens_used
        self.daily_usage[day_key] = self.daily_usage.get(day_key, 0) + tokens_used
        
        # Aggiorna costi
        cost = tokens_used * self.token_costs.get(model, 0.001)
        self.hourly_cost[hour_key] = self.hourly_cost.get(hour_key, 0.0) + cost
        
        logger.info(f"📊 Token utilizzati: {tokens_used} | Costo: ${cost:.4f} | Modello: {model}")
    
    def get_current_usage(self) -> Dict[str, any]:
        """Ottieni statistiche di utilizzo correnti"""
        hour_key = self._get_hour_key()
        day_key = self._get_day_key()
        
        hourly_tokens = self.hourly_usage.get(hour_key, 0)
        daily_tokens = self.daily_usage.get(day_key, 0)
        hourly_cost = self.hourly_cost.get(hour_key, 0.0)
        
        return {
            "hourly": {
                "tokens": hourly_tokens,
                "limit": self.config.max_tokens_per_hour,
                "percent": (hourly_tokens / self.config.max_tokens_per_hour) * 100,
                "cost": hourly_cost,
                "cost_limit": self.config.max_cost_per_hour
            },
            "daily": {
                "tokens": daily_tokens,
                "limit": self.config.max_tokens_per_day,
                "percent": (daily_tokens / self.config.max_tokens_per_day) * 100
            }
        }
    
    def reset_budgets(self):
        """Reset manuale dei budget (admin only)"""
        self.hourly_usage.clear()
        self.daily_usage.clear()
        self.hourly_cost.clear()
        logger.warning("🔄 Budget reset effettuato")
