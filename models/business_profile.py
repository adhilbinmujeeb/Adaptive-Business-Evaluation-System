from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime

@dataclass
class FinancialMetrics:
    revenue: float = 0.0
    profit: float = 0.0
    ebitda: float = 0.0
    growth_rate: float = 0.0
    burn_rate: Optional[float] = None
    cash_balance: Optional[float] = None
    revenue_growth: Optional[float] = None
    ebitda_margin: Optional[float] = None
    profit_margin: Optional[float] = None
    free_cash_flow: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "revenue": self.revenue,
            "profit": self.profit,
            "ebitda": self.ebitda,
            "growth_rate": self.growth_rate,
            "burn_rate": self.burn_rate,
            "cash_balance": self.cash_balance,
            "revenue_growth": self.revenue_growth,
            "ebitda_margin": self.ebitda_margin,
            "profit_margin": self.profit_margin,
            "free_cash_flow": self.free_cash_flow
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FinancialMetrics':
        return cls(
            revenue=float(data.get("revenue", 0.0)),
            profit=float(data.get("profit", 0.0)),
            ebitda=float(data.get("ebitda", 0.0)),
            growth_rate=float(data.get("growth_rate", 0.0)),
            burn_rate=float(data.get("burn_rate")) if data.get("burn_rate") is not None else None,
            cash_balance=float(data.get("cash_balance")) if data.get("cash_balance") is not None else None,
            revenue_growth=float(data.get("revenue_growth")) if data.get("revenue_growth") is not None else None,
            ebitda_margin=float(data.get("ebitda_margin")) if data.get("ebitda_margin") is not None else None,
            profit_margin=float(data.get("profit_margin")) if data.get("profit_margin") is not None else None,
            free_cash_flow=float(data.get("free_cash_flow")) if data.get("free_cash_flow") is not None else None
        )

@dataclass
class MarketMetrics:
    total_market_size: float = 0.0
    market_share: float = 0.0
    competitor_count: int = 0
    market_growth_rate: float = 0.0
    target_market_size: Optional[float] = None
    market_penetration: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_market_size": self.total_market_size,
            "market_share": self.market_share,
            "competitor_count": self.competitor_count,
            "market_growth_rate": self.market_growth_rate,
            "target_market_size": self.target_market_size,
            "market_penetration": self.market_penetration
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketMetrics':
        return cls(
            total_market_size=float(data.get("total_market_size", 0.0)),
            market_share=float(data.get("market_share", 0.0)),
            competitor_count=int(data.get("competitor_count", 0)),
            market_growth_rate=float(data.get("market_growth_rate", 0.0)),
            target_market_size=float(data.get("target_market_size")) if data.get("target_market_size") is not None else None,
            market_penetration=float(data.get("market_penetration")) if data.get("market_penetration") is not None else None
        )

@dataclass
class OperationalMetrics:
    employee_count: int = 0
    customer_count: int = 0
    churn_rate: float = 0.0
    retention_rate: float = 0.0
    acquisition_cost: Optional[float] = None
    lifetime_value: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "employee_count": self.employee_count,
            "customer_count": self.customer_count,
            "churn_rate": self.churn_rate,
            "retention_rate": self.retention_rate,
            "acquisition_cost": self.acquisition_cost,
            "lifetime_value": self.lifetime_value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OperationalMetrics':
        return cls(
            employee_count=int(data.get("employee_count", 0)),
            customer_count=int(data.get("customer_count", 0)),
            churn_rate=float(data.get("churn_rate", 0.0)),
            retention_rate=float(data.get("retention_rate", 0.0)),
            acquisition_cost=float(data.get("acquisition_cost")) if data.get("acquisition_cost") is not None else None,
            lifetime_value=float(data.get("lifetime_value")) if data.get("lifetime_value") is not None else None
        )

@dataclass
class BusinessProfile:
    id: str
    name: str
    industry: str
    business_stage: str
    business_type: str
    description: str
    founded_date: datetime
    financial_metrics: FinancialMetrics = field(default_factory=FinancialMetrics)
    market_metrics: MarketMetrics = field(default_factory=MarketMetrics)
    operational_metrics: OperationalMetrics = field(default_factory=OperationalMetrics)
    competitive_advantages: List[str] = field(default_factory=list)
    key_risks: List[str] = field(default_factory=list)
    growth_opportunities: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "industry": self.industry,
            "business_stage": self.business_stage,
            "business_type": self.business_type,
            "description": self.description,
            "founded_date": self.founded_date.isoformat(),
            "financial_metrics": self.financial_metrics.to_dict(),
            "market_metrics": self.market_metrics.to_dict(),
            "operational_metrics": self.operational_metrics.to_dict(),
            "competitive_advantages": self.competitive_advantages,
            "key_risks": self.key_risks,
            "growth_opportunities": self.growth_opportunities,
            "last_updated": self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BusinessProfile':
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            industry=data.get("industry", ""),
            business_stage=data.get("business_stage", ""),
            business_type=data.get("business_type", ""),
            description=data.get("description", ""),
            founded_date=datetime.fromisoformat(data.get("founded_date", datetime.now().isoformat())),
            financial_metrics=FinancialMetrics.from_dict(data.get("financial_metrics", {})),
            market_metrics=MarketMetrics.from_dict(data.get("market_metrics", {})),
            operational_metrics=OperationalMetrics.from_dict(data.get("operational_metrics", {})),
            competitive_advantages=data.get("competitive_advantages", []),
            key_risks=data.get("key_risks", []),
            growth_opportunities=data.get("growth_opportunities", []),
            last_updated=datetime.fromisoformat(data.get("last_updated", datetime.now().isoformat()))
        )
    
    def calculate_derived_metrics(self) -> Dict[str, float]:
        """Calculate derived metrics from existing data."""
        metrics = {}
        
        try:
            # Financial ratios
            if self.financial_metrics.revenue > 0:
                metrics["profit_margin"] = self.financial_metrics.profit / self.financial_metrics.revenue
                if self.financial_metrics.ebitda is not None:
                    metrics["ebitda_margin"] = self.financial_metrics.ebitda / self.financial_metrics.revenue
            
            # Market metrics
            if self.market_metrics.total_market_size > 0:
                metrics["market_share"] = self.financial_metrics.revenue / self.market_metrics.total_market_size
            
            # Operational metrics
            if self.operational_metrics.customer_count > 0 and self.financial_metrics.revenue > 0:
                metrics["revenue_per_customer"] = self.financial_metrics.revenue / self.operational_metrics.customer_count
                
        except Exception as e:
            print(f"Error calculating derived metrics: {str(e)}")
            
        return metrics
    
    def get_key_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of key metrics."""
        return {
            "revenue": self.financial_metrics.revenue,
            "profit": self.financial_metrics.profit,
            "growth_rate": self.financial_metrics.growth_rate,
            "market_share": self.market_metrics.market_share,
            "customer_count": self.operational_metrics.customer_count,
            "employee_count": self.operational_metrics.employee_count
        }
