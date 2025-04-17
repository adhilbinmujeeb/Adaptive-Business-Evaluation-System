from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum

class ValuationMethod(Enum):
    REVENUE_MULTIPLE = "Revenue Multiple"
    EBITDA_MULTIPLE = "EBITDA Multiple"
    DCF = "Discounted Cash Flow"
    ASSET_BASED = "Asset Based"
    COMPARABLE = "Comparable Company Analysis"

@dataclass
class ValuationResult:
    method: ValuationMethod
    value: float
    confidence_score: float
    multiplier_used: Optional[float] = None
    assumptions: Dict[str, Any] = None
    comparable_companies: List[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "value": self.value,
            "confidence_score": self.confidence_score,
            "multiplier_used": self.multiplier_used,
            "assumptions": self.assumptions,
            "comparable_companies": self.comparable_companies
        }

@dataclass
class ValuationSummary:
    business_name: str
    industry: str
    business_stage: str
    valuation_date: str
    results: List[ValuationResult]
    recommended_range: Dict[str, float]
    confidence_level: str
    key_factors: List[str]
    risk_factors: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "business_name": self.business_name,
            "industry": self.industry,
            "business_stage": self.business_stage,
            "valuation_date": self.valuation_date,
            "results": [result.to_dict() for result in self.results],
            "recommended_range": self.recommended_range,
            "confidence_level": self.confidence_level,
            "key_factors": self.key_factors,
            "risk_factors": self.risk_factors
        }

    def get_average_valuation(self) -> float:
        """Calculate weighted average valuation based on confidence scores."""
        if not self.results:
            return 0.0
            
        total_weighted_value = sum(r.value * r.confidence_score for r in self.results)
        total_weights = sum(r.confidence_score for r in self.results)
        
        return total_weighted_value / total_weights if total_weights > 0 else 0.0
