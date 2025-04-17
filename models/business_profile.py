from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class FinancialMetrics:
    """Financial metrics for a business."""
    revenue: float
    profit: Optional[float] = None
    ebitda: Optional[float] = None
    operating_costs: Optional[float] = None
    revenue_growth: Optional[float] = 0.0
    growth_rate: Optional[float] = 0.0
    ebitda_margin: Optional[float] = None
    profit_margin: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "revenue": self.revenue,
            "profit": self.profit,
            "ebitda": self.ebitda,
            "operating_costs": self.operating_costs,
            "revenue_growth": self.revenue_growth,
            "growth_rate": self.growth_rate,
            "ebitda_margin": self.ebitda_margin,
            "profit_margin": self.profit_margin
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FinancialMetrics':
        return cls(
            revenue=data.get("revenue", 0),
            profit=data.get("profit"),
            ebitda=data.get("ebitda"),
            operating_costs=data.get("operating_costs"),
            revenue_growth=data.get("revenue_growth", 0.0),
            growth_rate=data.get("growth_rate", 0.0),
            ebitda_margin=data.get("ebitda_margin"),
            profit_margin=data.get("profit_margin")
        )

    def calculate_margins(self):
        """Calculate financial margins if not already set."""
        if self.revenue > 0:
            if self.profit is not None and self.profit_margin is None:
                self.profit_margin = self.profit / self.revenue
            if self.ebitda is not None and self.ebitda_margin is None:
                self.ebitda_margin = self.ebitda / self.revenue

@dataclass
class MarketMetrics:
    """Market-related metrics for a business."""
    total_market_size: float
    market_share: Optional[float] = None
    competitor_count: Optional[int] = None
    competitors: Optional[List[str]] = None
    market_growth_rate: Optional[float] = None
    target_market_segments: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_market_size": self.total_market_size,
            "market_share": self.market_share,
            "competitor_count": self.competitor_count,
            "competitors": self.competitors,
            "market_growth_rate": self.market_growth_rate,
            "target_market_segments": self.target_market_segments
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketMetrics':
        return cls(
            total_market_size=data.get("total_market_size", 0),
            market_share=data.get("market_share"),
            competitor_count=data.get("competitor_count"),
            competitors=data.get("competitors"),
            market_growth_rate=data.get("market_growth_rate"),
            target_market_segments=data.get("target_market_segments")
        )

    def calculate_market_share(self, revenue: float):
        """Calculate market share if not already set."""
        if self.market_share is None and self.total_market_size > 0:
            self.market_share = revenue / self.total_market_size

@dataclass
class OperationalMetrics:
    """Operational metrics for a business."""
    customer_count: Optional[int] = None
    employee_count: Optional[int] = None
    customer_acquisition_cost: Optional[float] = None
    lifetime_value: Optional[float] = None
    churn_rate: Optional[float] = None
    retention_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "customer_count": self.customer_count,
            "employee_count": self.employee_count,
            "customer_acquisition_cost": self.customer_acquisition_cost,
            "lifetime_value": self.lifetime_value,
            "churn_rate": self.churn_rate,
            "retention_rate": self.retention_rate
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OperationalMetrics':
        return cls(
            customer_count=data.get("customer_count"),
            employee_count=data.get("employee_count"),
            customer_acquisition_cost=data.get("customer_acquisition_cost"),
            lifetime_value=data.get("lifetime_value"),
            churn_rate=data.get("churn_rate"),
            retention_rate=data.get("retention_rate")
        )

@dataclass
class BusinessProfile:
    """Complete business profile with all metrics."""
    business_name: str
    industry: str
    business_stage: str
    description: Optional[str]
    business_model: str
    financial_metrics: FinancialMetrics
    market_metrics: MarketMetrics
    operational_metrics: Optional[OperationalMetrics] = None
    founding_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "business_name": self.business_name,
            "industry": self.industry,
            "business_stage": self.business_stage,
            "description": self.description,
            "business_model": self.business_model,
            "financial_metrics": self.financial_metrics.to_dict(),
            "market_metrics": self.market_metrics.to_dict(),
            "operational_metrics": self.operational_metrics.to_dict() if self.operational_metrics else None,
            "founding_date": self.founding_date.isoformat() if self.founding_date else None,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BusinessProfile':
        return cls(
            business_name=data["business_name"],
            industry=data["industry"],
            business_stage=data["business_stage"],
            description=data.get("description"),
            business_model=data["business_model"],
            financial_metrics=FinancialMetrics.from_dict(data["financial_metrics"]),
            market_metrics=MarketMetrics.from_dict(data["market_metrics"]),
            operational_metrics=OperationalMetrics.from_dict(data["operational_metrics"]) if data.get("operational_metrics") else None,
            founding_date=datetime.fromisoformat(data["founding_date"]) if data.get("founding_date") else None,
            last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else None
        )

    def calculate_derived_metrics(self):
        """Calculate any derived metrics that haven't been set."""
        # Calculate financial margins
        self.financial_metrics.calculate_margins()
        
        # Calculate market share
        self.market_metrics.calculate_market_share(self.financial_metrics.revenue)
        
        # Update last_updated timestamp
        self.last_updated = datetime.now()

    def get_business_age(self) -> Optional[float]:
        """Calculate the age of the business in years."""
        if self.founding_date:
            delta = datetime.now() - self.founding_date
            return delta.days / 365.25
        return None

    def get_key_metrics(self) -> Dict[str, Any]:
        """Get a summary of key metrics."""
        return {
            "revenue": self.financial_metrics.revenue,
            "growth_rate": self.financial_metrics.growth_rate,
            "profit_margin": self.financial_metrics.profit_margin,
            "market_share": self.market_metrics.market_share,
            "business_age": self.get_business_age()
        }

    def validate(self) -> List[str]:
        """Validate the business profile and return any issues."""
        issues = []
        
        if self.financial_metrics.revenue <= 0:
            issues.append("Revenue must be greater than zero")
            
        if self.market_metrics.total_market_size <= 0:
            issues.append("Market size must be greater than zero")
            
        if self.financial_metrics.growth_rate is not None and (self.financial_metrics.growth_rate < -1 or self.financial_metrics.growth_rate > 10):
            issues.append("Growth rate seems unrealistic")
            
        return issues
