from typing import Dict, Any, Optional, List
from models.business_profile import BusinessProfile
from models.valuation import ValuationMethod, ValuationResult, ValuationSummary
from core.database import DatabaseConnection

class ValuationService:
    def __init__(self, db: DatabaseConnection):
        self.db = db

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

    def calculate_revenue_multiple_valuation(
        self,
        business_profile: BusinessProfile,
        industry_metrics: Dict[str, Any]
    ) -> Optional[ValuationResult]:
        """Calculate valuation using revenue multiple method."""
        try:
            revenue = getattr(business_profile.financial_metrics, 'revenue', None)
            if not revenue or not isinstance(revenue, (int, float)) or revenue <= 0:
                return None
                
            # Get industry average multiple
            industry_multiple = industry_metrics.get('revenue_multiple', 4.0)
            
            # Adjust multiple based on margins and growth
            adjusted_multiple = self._adjust_multiple(
                base_multiple=industry_multiple,
                business_profile=business_profile,
                industry_metrics=industry_metrics
            )
            
            value = revenue * adjusted_multiple
            
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
                    "annual_revenue": revenue
                }
            )
        except Exception as e:
            print(f"Error calculating revenue multiple valuation: {str(e)}")
            return None

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

    def calculate_dcf_valuation(
        self,
        business_profile: BusinessProfile,
        industry_metrics: Dict[str, Any]
    ) -> Optional[ValuationResult]:
        """Calculate valuation using DCF method."""
        try:
            cash_flow = getattr(business_profile.financial_metrics, 'free_cash_flow', None)
            growth_rate = getattr(business_profile.financial_metrics, 'growth_rate', None)
            
            if (not cash_flow or not growth_rate or
                not isinstance(cash_flow, (int, float)) or
                not isinstance(growth_rate, (int, float)) or
                cash_flow <= 0):
                return None
            
            # Use industry metrics or defaults
            discount_rate = industry_metrics.get('discount_rate', 0.15)
            terminal_growth = industry_metrics.get('terminal_growth', 0.03)
            projection_years = 5
            
            # Project cash flows
            projected_cash_flows = []
            current_cf = cash_flow
            
            for year in range(projection_years):
                current_cf *= (1 + growth_rate)
                projected_cash_flows.append(current_cf)
            
            # Calculate terminal value
            terminal_value = (projected_cash_flows[-1] * (1 + terminal_growth) /
                            (discount_rate - terminal_growth))
            
            # Calculate present value
            present_value = 0
            for i, cf in enumerate(projected_cash_flows):
                present_value += cf / ((1 + discount_rate) ** (i + 1))
            
            # Add terminal value
            present_value += terminal_value / ((1 + discount_rate) ** projection_years)
            
            confidence_score = self._calculate_confidence_score(
                business_profile=business_profile,
                method=ValuationMethod.DCF
            )
            
            return ValuationResult(
                method=ValuationMethod.DCF,
                value=present_value,
                confidence_score=confidence_score,
                multiplier_used=None,
                assumptions={
                    "discount_rate": discount_rate,
                    "terminal_growth": terminal_growth,
                    "projection_years": projection_years,
                    "initial_cash_flow": cash_flow,
                    "growth_rate": growth_rate
                }
            )
        except Exception as e:
            print(f"Error calculating DCF valuation: {str(e)}")
            return None

    def get_valuation_summary(
        self,
        business_profile: BusinessProfile
    ) -> ValuationSummary:
        """Generate a comprehensive valuation summary using multiple methods."""
        try:
            # Get industry metrics from database
            industry_metrics = self.db.get_industry_metrics(business_profile.industry)
            
            # Calculate valuations using different methods
            valuations = []
            
            revenue_val = self.calculate_revenue_multiple_valuation(
                business_profile,
                industry_metrics
            )
            if revenue_val:
                valuations.append(revenue_val)
            
            ebitda_val = self.calculate_ebitda_multiple_valuation(
                business_profile,
                industry_metrics
            )
            if ebitda_val:
                valuations.append(ebitda_val)
            
            dcf_val = self.calculate_dcf_valuation(
                business_profile,
                industry_metrics
            )
            if dcf_val:
                valuations.append(dcf_val)
            
            if not valuations:
                raise ValueError("No valid valuation methods available")
            
            # Calculate weighted average value
            total_confidence = sum(v.confidence_score for v in valuations)
            weighted_value = sum(
                v.value * (v.confidence_score / total_confidence)
                for v in valuations
            )
            
            # Calculate range
            values = [v.value for v in valuations]
            value_range = (min(values), max(values))
            
            # Get comparable companies
            comparable_companies = self.db.find_similar_businesses(
                industry=business_profile.industry,
                revenue_range=self._get_revenue_range(business_profile.financial_metrics.revenue),
                business_stage=business_profile.business_stage
            )
            
            return ValuationSummary(
                valuation_results=valuations,
                weighted_average_value=weighted_value,
                value_range=value_range,
                comparable_companies=comparable_companies,
                key_factors=self._get_key_factors(business_profile),
                recommendations=self._generate_recommendations(business_profile, valuations)
            )
        except Exception as e:
            print(f"Error generating valuation summary: {str(e)}")
            return None

    def _get_revenue_range(self, revenue: float) -> str:
        """Determine revenue range category."""
        if revenue < 1000000:  # Less than 1M
            return "0-1M"
        elif revenue < 10000000:  # 1M-10M
            return "1M-10M"
        elif revenue < 50000000:  # 10M-50M
            return "10M-50M"
        else:  # 50M+
            return "50M+"

    def _get_key_factors(self, business_profile: BusinessProfile) -> List[str]:
        """Identify key factors affecting valuation."""
        factors = []
        
        try:
            # Growth factors
            if business_profile.financial_metrics.growth_rate > 0.3:
                factors.append("High growth rate")
            
            # Profitability factors
            if business_profile.financial_metrics.profit_margin > 0.2:
                factors.append("Strong profit margins")
            
            # Market factors
            if business_profile.market_metrics.market_share > 0.1:
                factors.append("Significant market share")
            
            # Add other relevant factors
            if business_profile.operational_metrics.customer_count > 1000:
                factors.append("Large customer base")
                
        except Exception as e:
            print(f"Error getting key factors: {str(e)}")
            
        return factors

    def _generate_recommendations(
        self,
        business_profile: BusinessProfile,
        valuations: List[ValuationResult]
    ) -> List[str]:
        """Generate recommendations based on valuation analysis."""
        recommendations = []
        
        try:
            # Value improvement recommendations
            if business_profile.financial_metrics.profit_margin < 0.15:
                recommendations.append(
                    "Focus on improving profit margins through cost optimization"
                )
            
            if business_profile.market_metrics.market_share < 0.05:
                recommendations.append(
                    "Consider strategies to increase market share"
                )
            
            # Valuation method recommendations
            if len(valuations) < 3:
                recommendations.append(
                    "Improve financial data quality to enable more valuation methods"
                )
                
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            
        return recommendations
