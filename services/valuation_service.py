from typing import List, Dict, Optional, Any
import numpy as np
from datetime import datetime

# Change from relative to absolute imports
from models.business_profile import BusinessProfile
from models.valuation import ValuationMethod, ValuationResult, ValuationSummary
from core.database import DatabaseConnection
from core.llm_service import LLMService

class ValuationService:
    def __init__(self):
        self.db = DatabaseConnection()
        self.llm = LLMService()
        
    def calculate_revenue_multiple_valuation(
        self,
        business_profile: BusinessProfile,
        industry_metrics: Dict[str, Any]
    ) -> Optional[ValuationResult]:
        """Calculate valuation using revenue multiple method."""
        if not business_profile.financial_metrics.revenue:
            return None
            
        # Get industry average multiple
        industry_multiple = industry_metrics.get('revenue_multiple', 3.0)
        
        # Adjust multiple based on growth and profitability
        adjusted_multiple = self._adjust_multiple(
            base_multiple=industry_multiple,
            business_profile=business_profile,
            industry_metrics=industry_metrics
        )
        
        value = business_profile.financial_metrics.revenue * adjusted_multiple
        
        confidence_score = self._calculate_confidence_score(
            business_profile=business_profile,
            method=ValuationMethod.REVENUE_MULTIPLE
        )
        
        return ValuationResult(
            method=ValuationMethod.REVENUE_MULTIPLE,
            value=value,
            confidence_score=confidence_score,
            multiplier_used=adjusted_multiple,
            assumptions={
                "industry_multiple": industry_multiple,
                "adjusted_multiple": adjusted_multiple,
                "annual_revenue": business_profile.financial_metrics.revenue
            }
        )
        
    def calculate_ebitda_multiple_valuation(
        self,
        business_profile: BusinessProfile,
        industry_metrics: Dict[str, Any]
    ) -> Optional[ValuationResult]:
        """Calculate valuation using EBITDA multiple method."""
        if not business_profile.financial_metrics.ebitda:
            return None
            
        # Get industry average multiple
        industry_multiple = industry_metrics.get('ebitda_multiple', 8.0)
        
        # Adjust multiple based on margins and growth
        adjusted_multiple = self._adjust_multiple(
            base_multiple=industry_multiple,
            business_profile=business_profile,
            industry_metrics=industry_metrics,
            is_ebitda=True
        )
        
        value = business_profile.financial_metrics.ebitda * adjusted_multiple
        
        confidence_score = self._calculate_confidence_score(
            business_profile=business_profile,
            method=ValuationMethod.EBITDA_MULTIPLE
        )
        
        return ValuationResult(
            method=ValuationMethod.EBITDA_MULTIPLE,
            value=value,
            confidence_score=confidence_score,
            multiplier_used=adjusted_multiple,
            assumptions={
                "industry_multiple": industry_multiple,
                "adjusted_multiple": adjusted_multiple,
                "annual_ebitda": business_profile.financial_metrics.ebitda
            }
        )
        
    def calculate_dcf_valuation(
        self,
        business_profile: BusinessProfile,
        projected_cash_flows: List[float]
    ) -> Optional[ValuationResult]:
        """Calculate valuation using Discounted Cash Flow method."""
        if not projected_cash_flows or len(projected_cash_flows) < 5:
            return None
            
        # Calculate discount rate (WACC)
        wacc = self._calculate_wacc(business_profile)
        
        # Calculate terminal value
        terminal_growth_rate = 0.02  # 2% perpetual growth
        terminal_value = (projected_cash_flows[-1] * (1 + terminal_growth_rate)) / (wacc - terminal_growth_rate)
        
        # Calculate present value of cash flows
        present_values = []
        for i, cf in enumerate(projected_cash_flows):
            present_values.append(cf / ((1 + wacc) ** (i + 1)))
        
        # Add terminal value
        present_values.append(terminal_value / ((1 + wacc) ** len(projected_cash_flows)))
        
        value = sum(present_values)
        
        confidence_score = self._calculate_confidence_score(
            business_profile=business_profile,
            method=ValuationMethod.DCF
        )
        
        return ValuationResult(
            method=ValuationMethod.DCF,
            value=value,
            confidence_score=confidence_score,
            assumptions={
                "wacc": wacc,
                "terminal_growth_rate": terminal_growth_rate,
                "projected_cash_flows": projected_cash_flows,
                "terminal_value": terminal_value
            }
        )
        
    def get_comparable_companies(
        self,
        business_profile: BusinessProfile,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve comparable companies from the database."""
        return self.db.find_similar_businesses(
            industry=business_profile.industry,
            revenue_range=self._get_revenue_range(business_profile.financial_metrics.revenue),
            business_stage=business_profile.business_stage,
            limit=limit
        )
        
    def generate_valuation_summary(
        self,
        business_profile: BusinessProfile,
        valuation_results: List[ValuationResult]
    ) -> ValuationSummary:
        """Generate a comprehensive valuation summary."""
        if not valuation_results:
            raise ValueError("No valuation results provided")
            
        # Calculate weighted average value
        total_weight = sum(result.confidence_score for result in valuation_results)
        weighted_value = sum(
            result.value * (result.confidence_score / total_weight)
            for result in valuation_results
        )
        
        # Calculate range
        std_dev = np.std([result.value for result in valuation_results])
        value_range = {
            "low": max(0, weighted_value - std_dev),
            "mid": weighted_value,
            "high": weighted_value + std_dev
        }
        
        # Get key factors and risks using LLM
        key_factors = self._analyze_key_factors(business_profile)
        risk_factors = self._analyze_risk_factors(business_profile)
        
        return ValuationSummary(
            business_profile=business_profile,
            results=valuation_results,
            recommended_range=value_range,
            key_factors=key_factors,
            risk_factors=risk_factors
        )
        
    def _adjust_multiple(
        self,
        base_multiple: float,
        business_profile: BusinessProfile,
        industry_metrics: Dict[str, Any],
        is_ebitda: bool = False
    ) -> float:
        """Adjust valuation multiple based on business metrics."""
        adjustment = 1.0
        
        # Growth rate adjustment
        if business_profile.financial_metrics.growth_rate > industry_metrics.get('avg_growth_rate', 0.1):
            adjustment += 0.2
        
        # Margin adjustment
        if is_ebitda:
            ebitda_margin = (
                business_profile.financial_metrics.ebitda /
                business_profile.financial_metrics.revenue
            )
            if ebitda_margin > industry_metrics.get('avg_ebitda_margin', 0.15):
                adjustment += 0.15
        else:
            profit_margin = (
                business_profile.financial_metrics.profit /
                business_profile.financial_metrics.revenue
            )
            if profit_margin > industry_metrics.get('avg_profit_margin', 0.1):
                adjustment += 0.15
        
        # Market position adjustment
        market_share = (
            business_profile.financial_metrics.revenue /
            business_profile.market_metrics.total_market_size
        )
        if market_share > industry_metrics.get('avg_market_share', 0.05):
            adjustment += 0.1
            
        return base_multiple * adjustment
        
    def _calculate_confidence_score(
    self,
    business_profile: BusinessProfile,
    method: ValuationMethod
) -> float:
    """Calculate confidence score for valuation method."""
    base_score = 0.7
    adjustments = 0.0
    
    try:
        if method == ValuationMethod.REVENUE_MULTIPLE:
            # Check revenue growth
            revenue_growth = getattr(business_profile.financial_metrics, 'revenue_growth', None)
            if revenue_growth is not None and isinstance(revenue_growth, (int, float)) and revenue_growth > 0:
                adjustments += 0.1
            
            # Check profit
            profit = getattr(business_profile.financial_metrics, 'profit', None)
            if profit is not None and isinstance(profit, (int, float)) and profit > 0:
                adjustments += 0.1
                
        elif method == ValuationMethod.EBITDA_MULTIPLE:
            # Check EBITDA
            ebitda = getattr(business_profile.financial_metrics, 'ebitda', None)
            if ebitda is not None and isinstance(ebitda, (int, float)) and ebitda > 0:
                adjustments += 0.15
            
            # Check EBITDA margin
            ebitda_margin = getattr(business_profile.financial_metrics, 'ebitda_margin', None)
            if ebitda_margin is not None and isinstance(ebitda_margin, (int, float)) and ebitda_margin > 0.15:
                adjustments += 0.1
                
        elif method == ValuationMethod.DCF:
            # Check revenue growth
            revenue_growth = getattr(business_profile.financial_metrics, 'revenue_growth', None)
            if revenue_growth is not None and isinstance(revenue_growth, (int, float)) and revenue_growth > 0:
                adjustments += 0.1
            
            # Check profit
            profit = getattr(business_profile.financial_metrics, 'profit', None)
            if profit is not None and isinstance(profit, (int, float)) and profit > 0:
                adjustments += 0.1
    except (AttributeError, TypeError) as e:
        print(f"Warning: Error calculating confidence score: {str(e)}")
        return base_score
            
    return min(0.95, base_score + adjustments)

def _adjust_multiple(
    self,
    base_multiple: float,
    business_profile: BusinessProfile,
    industry_metrics: Dict[str, Any],
    is_ebitda: bool = False
) -> float:
    """Adjust valuation multiple based on business metrics."""
    adjustment = 1.0
    
    try:
        # Growth rate adjustment
        growth_rate = getattr(business_profile.financial_metrics, 'growth_rate', None)
        avg_growth_rate = industry_metrics.get('avg_growth_rate', 0.1)
        
        if (growth_rate is not None and 
            isinstance(growth_rate, (int, float)) and 
            isinstance(avg_growth_rate, (int, float)) and 
            growth_rate > avg_growth_rate):
            adjustment += 0.2
        
        # Margin adjustment
        if is_ebitda:
            ebitda = getattr(business_profile.financial_metrics, 'ebitda', None)
            revenue = getattr(business_profile.financial_metrics, 'revenue', None)
            
            if (ebitda is not None and revenue is not None and 
                isinstance(ebitda, (int, float)) and 
                isinstance(revenue, (int, float)) and 
                revenue > 0):
                ebitda_margin = ebitda / revenue
                avg_ebitda_margin = industry_metrics.get('avg_ebitda_margin', 0.15)
                
                if isinstance(avg_ebitda_margin, (int, float)) and ebitda_margin > avg_ebitda_margin:
                    adjustment += 0.15
        else:
            profit = getattr(business_profile.financial_metrics, 'profit', None)
            revenue = getattr(business_profile.financial_metrics, 'revenue', None)
            
            if (profit is not None and revenue is not None and 
                isinstance(profit, (int, float)) and 
                isinstance(revenue, (int, float)) and 
                revenue > 0):
                profit_margin = profit / revenue
                avg_profit_margin = industry_metrics.get('avg_profit_margin', 0.1)
                
                if isinstance(avg_profit_margin, (int, float)) and profit_margin > avg_profit_margin:
                    adjustment += 0.15
        
        # Market position adjustment
        revenue = getattr(business_profile.financial_metrics, 'revenue', None)
        market_size = getattr(business_profile.market_metrics, 'total_market_size', None)
        
        if (revenue is not None and market_size is not None and 
            isinstance(revenue, (int, float)) and 
            isinstance(market_size, (int, float)) and 
            market_size > 0):
            market_share = revenue / market_size
            avg_market_share = industry_metrics.get('avg_market_share', 0.05)
            
            if isinstance(avg_market_share, (int, float)) and market_share > avg_market_share:
                adjustment += 0.1
                
    except (AttributeError, TypeError, ZeroDivisionError) as e:
        print(f"Warning: Error adjusting multiple: {str(e)}")
        return base_multiple
            
    return base_multiple * adjustment

def calculate_ebitda_multiple_valuation(
    self,
    business_profile: BusinessProfile,
    industry_metrics: Dict[str, Any]
) -> Optional[ValuationResult]:
    """Calculate valuation using EBITDA multiple method."""
    try:
        ebitda = getattr(business_profile.financial_metrics, 'ebitda', None)
        if not ebitda or not isinstance(ebitda, (int, float)) or ebitda <= 0:
            return None
            
        # Get industry average multiple
        industry_multiple = industry_metrics.get('ebitda_multiple', 8.0)
        
        # Adjust multiple based on margins and growth
        adjusted_multiple = self._adjust_multiple(
            base_multiple=industry_multiple,
            business_profile=business_profile,
            industry_metrics=industry_metrics,
            is_ebitda=True
        )
        
        value = ebitda * adjusted_multiple
        
        confidence_score = self._calculate_confidence_score(
            business_profile=business_profile,
            method=ValuationMethod.EBITDA_MULTIPLE
        )
        
        return ValuationResult(
            method=ValuationMethod.EBITDA_MULTIPLE,
            value=value,
            confidence_score=confidence_score,
            multiplier_used=adjusted_multiple,
            assumptions={
                "industry_multiple": industry_multiple,
                "adjusted_multiple": adjusted_multiple,
                "annual_ebitda": ebitda
            }
        )
    except Exception as e:
        print(f"Error calculating EBITDA multiple valuation: {str(e)}")
        return None
        
    def _calculate_wacc(self, business_profile: BusinessProfile) -> float:
        """Calculate Weighted Average Cost of Capital."""
        # Simplified WACC calculation
        risk_free_rate = 0.03  # 3% risk-free rate
        market_risk_premium = 0.06  # 6% market risk premium
        beta = 1.2  # Assumed beta
        
        # Cost of equity
        cost_of_equity = risk_free_rate + (beta * market_risk_premium)
        
        # Simplified WACC (assuming 100% equity financing for early-stage companies)
        return cost_of_equity
        
    def _get_revenue_range(self, revenue: float) -> str:
        """Get revenue range category."""
        if revenue < 1_000_000:
            return "0-1M"
        elif revenue < 10_000_000:
            return "1M-10M"
        elif revenue < 50_000_000:
            return "10M-50M"
        else:
            return "50M+"
            
    def _analyze_key_factors(self, business_profile: BusinessProfile) -> List[str]:
        """Analyze key value drivers using LLM."""
        prompt = f"""
        Analyze the following business profile and list 3-5 key value drivers:
        - Industry: {business_profile.industry}
        - Stage: {business_profile.business_stage}
        - Revenue: ${business_profile.financial_metrics.revenue:,.2f}
        - Growth Rate: {business_profile.financial_metrics.growth_rate}
        - Market Size: ${business_profile.market_metrics.total_market_size:,.2f}
        """
        
        response = self.llm.generate_response(prompt)
        return [factor.strip() for factor in response.split('\n') if factor.strip()]
        
    def _analyze_risk_factors(self, business_profile: BusinessProfile) -> List[str]:
        """Analyze risk factors using LLM."""
        prompt = f"""
        Analyze the following business profile and list 3-5 key risk factors:
        - Industry: {business_profile.industry}
        - Stage: {business_profile.business_stage}
        - Revenue: ${business_profile.financial_metrics.revenue:,.2f}
        - Growth Rate: {business_profile.financial_metrics.growth_rate}
        - Market Size: ${business_profile.market_metrics.total_market_size:,.2f}
        """
        
        response = self.llm.generate_response(prompt)
        return [risk.strip() for risk in response.split('\n') if risk.strip()]
