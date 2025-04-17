from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
from ..models.valuation import ValuationMethod, ValuationResult, ValuationSummary
from ..models.business_profile import BusinessProfile
from ..core.database import DatabaseConnection
from ..core.config import VALUATION_METRICS

class ValuationService:
    def __init__(self):
        self.db = DatabaseConnection()

    def calculate_revenue_multiple_valuation(
        self,
        profile: BusinessProfile,
        industry_metrics: Dict[str, Any]
    ) -> ValuationResult:
        """Calculate valuation using revenue multiple method."""
        revenue = profile.financial_metrics.revenue
        if not revenue or revenue <= 0:
            return None

        # Get industry-specific multiples
        industry_config = VALUATION_METRICS.get(
            profile.industry,
            VALUATION_METRICS['default']
        )
        
        stage_multiples = industry_config['revenue_multiple_ranges'][profile.business_stage.lower()]
        
        # Calculate base multiple
        base_multiple = (stage_multiples['min'] + stage_multiples['max']) / 2
        
        # Adjust multiple based on growth and margins
        growth_adjustment = 0.0
        if profile.financial_metrics.revenue_growth:
            growth_adjustment = min(max(profile.financial_metrics.revenue_growth - 0.15, -0.5), 0.5)
            
        margin_adjustment = 0.0
        if profile.financial_metrics.gross_margin:
            margin_adjustment = min(max(profile.financial_metrics.gross_margin - 0.5, -0.3), 0.3)
            
        final_multiple = base_multiple * (1 + growth_adjustment + margin_adjustment)
        
        # Calculate valuation
        value = revenue * final_multiple
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(profile, 'revenue_multiple')
        
        return ValuationResult(
            method=ValuationMethod.REVENUE_MULTIPLE,
            value=value,
            confidence_score=confidence_score,
            multiplier_used=final_multiple,
            assumptions={
                "base_multiple": base_multiple,
                "growth_adjustment": growth_adjustment,
                "margin_adjustment": margin_adjustment
            }
        )

    def calculate_ebitda_multiple_valuation(
        self,
        profile: BusinessProfile,
        industry_metrics: Dict[str, Any]
    ) -> ValuationResult:
        """Calculate valuation using EBITDA multiple method."""
        ebitda = profile.financial_metrics.ebitda
        if not ebitda or ebitda <= 0:
            return None

        # Similar structure to revenue multiple calculation
        industry_config = VALUATION_METRICS.get(
            profile.industry,
            VALUATION_METRICS['default']
        )
        
        stage_multiples = industry_config['ebitda_multiple_ranges'][profile.business_stage.lower()]
        base_multiple = (stage_multiples['min'] + stage_multiples['max']) / 2
        
        # Adjustments based on margins and growth
        margin_adjustment = 0.0
        if profile.financial_metrics.gross_margin:
            margin_adjustment = min(max(profile.financial_metrics.gross_margin - 0.5, -0.3), 0.3)
            
        value = ebitda * base_multiple * (1 + margin_adjustment)
        
        confidence_score = self._calculate_confidence_score(profile, 'ebitda_multiple')
        
        return ValuationResult(
            method=ValuationMethod.EBITDA_MULTIPLE,
            value=value,
            confidence_score=confidence_score,
            multiplier_used=base_multiple,
            assumptions={
                "base_multiple": base_multiple,
                "margin_adjustment": margin_adjustment
            }
        )

    def calculate_dcf_valuation(
        self,
        profile: BusinessProfile,
        projected_cash_flows: List[float],
        discount_rate: float = 0.15,
        terminal_growth_rate: float = 0.03
    ) -> ValuationResult:
        """Calculate valuation using DCF method."""
        if not projected_cash_flows:
            return None

        # Calculate present value of projected cash flows
        present_values = []
        for i, cf in enumerate(projected_cash_flows):
            present_values.append(cf / (1 + discount_rate) ** (i + 1))

        # Calculate terminal value
        terminal_value = (projected_cash_flows[-1] * (1 + terminal_growth_rate)) / (discount_rate - terminal_growth_rate)
        terminal_value_pv = terminal_value / (1 + discount_rate) ** len(projected_cash_flows)

        # Total value is sum of PV of cash flows plus PV of terminal value
        value = sum(present_values) + terminal_value_pv
        
        confidence_score = self._calculate_confidence_score(profile, 'dcf')
        
        return ValuationResult(
            method=ValuationMethod.DCF,
            value=value,
            confidence_score=confidence_score,
            assumptions={
                "discount_rate": discount_rate,
                "terminal_growth_rate": terminal_growth_rate,
                "projection_years": len(projected_cash_flows)
            }
        )

    def get_comparable_companies(
        self,
        profile: BusinessProfile,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get comparable companies from the database."""
        return self.db.find_similar_businesses(profile.industry, limit)

    def _calculate_confidence_score(
        self,
        profile: BusinessProfile,
        method: str
    ) -> float:
        """Calculate confidence score for valuation method."""
        base_score = 0.7  # Start with a base confidence score
        
        if method == 'revenue_multiple':
            if not profile.financial_metrics.revenue:
                return 0.0
            if profile.financial_metrics.revenue_growth:
                base_score += 0.1
            if profile.financial_metrics.gross_margin:
                base_score += 0.1
                
        elif method == 'ebitda_multiple':
            if not profile.financial_metrics.ebitda:
                return 0.0
            if profile.financial_metrics.gross_margin:
                base_score += 0.15
                
        elif method == 'dcf':
            base_score = 0.6  # DCF is inherently more speculative
            if profile.financial_metrics.revenue_growth:
                base_score += 0.1
            if profile.financial_metrics.cash_flow:
                base_score += 0.1
                
        # Adjust based on data completeness
        completeness = self._calculate_data_completeness(profile)
        base_score *= completeness
        
        return min(base_score, 1.0)

    def _calculate_data_completeness(self, profile: BusinessProfile) -> float:
        """Calculate completeness of business profile data."""
        required_fields = [
            profile.financial_metrics.revenue,
            profile.financial_metrics.profit,
            profile.market_metrics.market_size,
            profile.target_customers,
            profile.competitive_advantages
        ]
        
        return sum(1 for field in required_fields if field) / len(required_fields)

    def generate_valuation_summary(
        self,
        profile: BusinessProfile,
        results: List[ValuationResult]
    ) -> ValuationSummary:
        """Generate a comprehensive valuation summary."""
        if not results:
            return None

        # Calculate recommended range
        values = [r.value for r in results]
        confidence_scores = [r.confidence_score for r in results]
        
        weighted_avg = np.average(values, weights=confidence_scores)
        std_dev = np.std(values)
        
        recommended_range = {
            "low": max(0, weighted_avg - std_dev),
            "mid": weighted_avg,
            "high": weighted_avg + std_dev
        }
        
        # Determine overall confidence level
        avg_confidence = np.mean(confidence_scores)
        if avg_confidence >= 0.8:
            confidence_level = "High"
        elif avg_confidence >= 0.6:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        # Get comparable companies
        comparables = self.get_comparable_companies(profile)
        
        return ValuationSummary(
            business_name=profile.business_name,
            industry=profile.industry,
            business_stage=profile.business_stage,
            valuation_date=datetime.now().strftime("%Y-%m-%d"),
            results=results,
            recommended_range=recommended_range,
            confidence_level=confidence_level,
            key_factors=self._extract_key_factors(profile),
            risk_factors=self._identify_risk_factors(profile)
        )

    def _extract_key_factors(self, profile: BusinessProfile) -> List[str]:
        """Extract key factors affecting valuation."""
        factors = []
        
        if profile.financial_metrics.revenue_growth and profile.financial_metrics.revenue_growth > 0.3:
            factors.append("High revenue growth")
            
        if profile.financial_metrics.gross_margin and profile.financial_metrics.gross_margin > 0.6:
            factors.append("Strong gross margins")
            
        if profile.competitive_advantages:
            factors.append("Strong competitive advantages")
            
        if profile.market_metrics.market_growth_rate and profile.market_metrics.market_growth_rate > 0.1:
            factors.append("Growing market")
            
        return factors

    def _identify_risk_factors(self, profile: BusinessProfile) -> List[str]:
        """Identify risk factors affecting valuation."""
        risks = []
        
        if profile.financial_metrics.burn_rate and profile.financial_metrics.burn_rate > 0:
            risks.append("High burn rate")
            
        if not profile.market_metrics.market_share or profile.market_metrics.market_share < 0.05:
            risks.append("Low market share")
            
        if len(profile.market_metrics.competitors) > 5:
            risks.append("Highly competitive market")
            
        if profile.challenges:
            risks.extend(profile.challenges)
            
        return risks
