from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime

@dataclass
class FinancialMetrics:
    revenue: float = 0.0
    profit: float = 0.0
    ebitda: float = 0.0
    burn_rate: Optional[float] = None
    cash_flow: Optional[float] = None
    revenue_growth: Optional[float] = None
    gross_margin: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

@dataclass
class MarketMetrics:
    total_market_size: Optional[float] = None
    target_market_size: Optional[float] = None
    market_growth_rate: Optional[float] = None
    market_share: Optional[float] = None
    competitors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

@dataclass
class BusinessProfile:
    business_name: str
    industry: str
    business_stage: str
    description: str
    founding_date: Optional[datetime] = None
    location: Optional[str] = None
    team_size: Optional[int] = None
    business_model: Optional[str] = None
    financial_metrics: FinancialMetrics = field(default_factory=FinancialMetrics)
    market_metrics: MarketMetrics = field(default_factory=MarketMetrics)
    key_products: List[str] = field(default_factory=list)
    target_customers: List[str] = field(default_factory=list)
    competitive_advantages: List[str] = field(default_factory=list)
    challenges: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "business_name": self.business_name,
            "industry": self.industry,
            "business_stage": self.business_stage,
            "description": self.description,
            "founding_date": self.founding_date.isoformat() if self.founding_date else None,
            "location": self.location,
            "team_size": self.team_size,
            "business_model": self.business_model,
            "financial_metrics": self.financial_metrics.to_dict(),
            "market_metrics": self.market_metrics.to_dict(),
            "key_products": self.key_products,
            "target_customers": self.target_customers,
            "competitive_advantages": self.competitive_advantages,
            "challenges": self.challenges
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BusinessProfile':
        financial_data = data.pop('financial_metrics', {})
        market_data = data.pop('market_metrics', {})
        
        if 'founding_date' in data and isinstance(data['founding_date'], str):
            data['founding_date'] = datetime.fromisoformat(data['founding_date'])
            
        return cls(
            **{k: v for k, v in data.items() if k in cls.__annotations__},
            financial_metrics=FinancialMetrics(**financial_data),
            market_metrics=MarketMetrics(**market_data)
        )
