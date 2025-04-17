from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from core.database import DatabaseConnection
from core.llm_service import LLMService
from models.business_profile import BusinessProfile

class AnalyticsService:
    def __init__(self):
        self.db = DatabaseConnection()
        self.llm = LLMService()

    def get_industry_metrics(self, industry: str) -> Dict[str, Any]:
        """Get industry-specific metrics and benchmarks."""
        try:
            # Default metrics in case of database errors
            default_metrics = {
                'revenue_multiple': 3.0,
                'ebitda_multiple': 8.0,
                'avg_growth_rate': 0.15,
                'avg_profit_margin': 0.12,
                'avg_ebitda_margin': 0.18,
                'avg_market_share': 0.05,
                'avg_revenue': 1_000_000,
                'median_deal_size': 5_000_000,
                'avg_valuation_multiple': 4.0,
                'avg_competitor_count': 5
            }

            # Get metrics from database
            pipeline = [
                {"$match": {"industry": industry}},
                {
                    "$group": {
                        "_id": None,
                        "revenue_multiple": {"$avg": "$revenue_multiple"},
                        "ebitda_multiple": {"$avg": "$ebitda_multiple"},
                        "avg_growth_rate": {"$avg": "$growth_rate"},
                        "avg_profit_margin": {"$avg": "$profit_margin"},
                        "avg_ebitda_margin": {"$avg": "$ebitda_margin"},
                        "avg_market_share": {"$avg": "$market_share"},
                        "avg_revenue": {"$avg": "$revenue"},
                        "median_deal_size": {"$avg": "$valuation"},
                        "avg_valuation_multiple": {"$avg": "$valuation_multiple"},
                        "avg_competitor_count": {"$avg": {"$size": "$competitors"}}
                    }
                }
            ]

            results = list(self.db.businesses.aggregate(pipeline))
            
            if results:
                metrics = results[0]
                # Remove MongoDB _id field
                metrics.pop('_id', None)
                # Fill in any missing metrics with defaults
                for key, value in default_metrics.items():
                    if key not in metrics or metrics[key] is None:
                        metrics[key] = value
                return metrics
            
            return default_metrics

        except Exception as e:
            print(f"Error fetching industry metrics: {str(e)}")
            return default_metrics

    def analyze_business_trends(
        self,
        industry: str,
        timeframe_months: int = 12
    ) -> Dict[str, Any]:
        """Analyze business trends over time."""
        pipeline = [
            {
                "$match": {
                    "industry": industry,
                    "timestamp": {
                        "$gte": datetime.now().replace(
                            day=1,
                            hour=0,
                            minute=0,
                            second=0,
                            microsecond=0
                        ) - pd.DateOffset(months=timeframe_months)
                    }
                }
            },
            {
                "$group": {
                    "_id": {
                        "year": {"$year": "$timestamp"},
                        "month": {"$month": "$timestamp"}
                    },
                    "avg_valuation": {"$avg": "$valuation"},
                    "deal_count": {"$sum": 1},
                    "success_rate": {
                        "$avg": {
                            "$cond": [
                                {"$eq": ["$deal_outcome.status", "success"]},
                                1,
                                0
                            ]
                        }
                    }
                }
            },
            {"$sort": {"_id.year": 1, "_id.month": 1}}
        ]

        try:
            results = list(self.db.businesses.aggregate(pipeline))
        except Exception as e:
            print(f"Error fetching business trends: {str(e)}")
            results = []
        
        # Process results into time series
        timeline = []
        valuations = []
        deal_counts = []
        success_rates = []
        
        for result in results:
            date_str = f"{result['_id']['year']}-{result['_id']['month']:02d}"
            timeline.append(date_str)
            valuations.append(result['avg_valuation'])
            deal_counts.append(result['deal_count'])
            success_rates.append(result['success_rate'])

        return {
            "timeline": timeline,
            "trends": {
                "valuations": self._calculate_trend(valuations),
                "deal_counts": self._calculate_trend(deal_counts),
                "success_rates": self._calculate_trend(success_rates)
            },
            "raw_data": {
                "valuations": valuations,
                "deal_counts": deal_counts,
                "success_rates": success_rates
            }
        }

    def get_comparable_businesses(
        self,
        business_profile: BusinessProfile,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get comparable businesses based on industry and metrics."""
        revenue_range = self._get_revenue_range(
            business_profile.financial_metrics.revenue
        )
        
        pipeline = [
            {
                "$match": {
                    "industry": business_profile.industry,
                    "business_stage": business_profile.business_stage,
                    "revenue_range": revenue_range
                }
            },
            {
                "$project": {
                    "business_name": 1,
                    "valuation": 1,
                    "revenue": 1,
                    "profit_margin": 1,
                    "growth_rate": 1,
                    "deal_outcome": 1
                }
            },
            {"$limit": limit}
        ]
        
        try:
            return list(self.db.businesses.aggregate(pipeline))
        except Exception as e:
            print(f"Error fetching comparable businesses: {str(e)}")
            return []

    def generate_performance_metrics(
        self,
        business_profile: BusinessProfile
    ) -> Dict[str, Any]:
        """Generate comprehensive performance metrics and benchmarks."""
        # Get industry benchmarks
        benchmarks = self.get_industry_metrics(business_profile.industry)
        
        # Calculate financial metrics
        financial_metrics = self._calculate_financial_metrics(
            business_profile,
            benchmarks
        )
        
        # Calculate operational metrics
        operational_metrics = self._calculate_operational_metrics(
            business_profile,
            benchmarks
        )
        
        # Calculate market metrics
        market_metrics = self._calculate_market_metrics(
            business_profile,
            benchmarks
        )
        
        return {
            "financial_metrics": financial_metrics,
            "operational_metrics": operational_metrics,
            "market_metrics": market_metrics,
            "industry_benchmarks": benchmarks
        }

    def _calculate_trend(self, data: List[float]) -> Dict[str, Any]:
        """Calculate trend statistics for a time series."""
        if not data:
            return {
                "direction": "neutral",
                "magnitude": 0,
                "volatility": 0
            }
            
        # Calculate period-over-period changes
        changes = [
            (data[i] - data[i-1]) / data[i-1]
            for i in range(1, len(data))
            if data[i-1] != 0
        ]
        
        if not changes:
            return {
                "direction": "neutral",
                "magnitude": 0,
                "volatility": 0
            }
            
        avg_change = np.mean(changes)
        volatility = np.std(changes)
        
        return {
            "direction": "up" if avg_change > 0.05 else "down" if avg_change < -0.05 else "neutral",
            "magnitude": abs(avg_change),
            "volatility": volatility
        }

    def _get_revenue_range(self, revenue: float) -> str:
        """Determine revenue range category."""
        if revenue < 1_000_000:
            return "0-1M"
        elif revenue < 10_000_000:
            return "1M-10M"
        elif revenue < 50_000_000:
            return "10M-50M"
        else:
            return "50M+"

    def _calculate_financial_metrics(
        self,
        business_profile: BusinessProfile,
        benchmarks: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate financial performance metrics."""
        metrics = {}
        
        # Revenue metrics
        metrics["revenue"] = {
            "value": business_profile.financial_metrics.revenue,
            "percentile": self._calculate_percentile(
                "revenue",
                business_profile.financial_metrics.revenue,
                business_profile.industry
            )
        }
        
        # Profit metrics
        if business_profile.financial_metrics.profit is not None:
            profit_margin = (
                business_profile.financial_metrics.profit /
                business_profile.financial_metrics.revenue
            )
            metrics["profit_margin"] = {
                "value": profit_margin,
                "percentile": self._calculate_percentile(
                    "profit_margin",
                    profit_margin,
                    business_profile.industry
                )
            }
        
        # Growth metrics
        if business_profile.financial_metrics.growth_rate is not None:
            metrics["growth_rate"] = {
                "value": business_profile.financial_metrics.growth_rate,
                "percentile": self._calculate_percentile(
                    "growth_rate",
                    business_profile.financial_metrics.growth_rate,
                    business_profile.industry
                )
            }
        
        return metrics

    def _calculate_operational_metrics(
        self,
        business_profile: BusinessProfile,
        benchmarks: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate operational performance metrics."""
        metrics = {}
        
        # Efficiency metrics
        if business_profile.financial_metrics.operating_costs is not None:
            operating_margin = (
                (business_profile.financial_metrics.revenue -
                 business_profile.financial_metrics.operating_costs) /
                business_profile.financial_metrics.revenue
            )
            metrics["operating_margin"] = {
                "value": operating_margin,
                "percentile": self._calculate_percentile(
                    "operating_margin",
                    operating_margin,
                    business_profile.industry
                )
            }
        
        # Scale metrics
        if business_profile.operational_metrics.customer_count is not None:
            revenue_per_customer = (
                business_profile.financial_metrics.revenue /
                business_profile.operational_metrics.customer_count
            )
            metrics["revenue_per_customer"] = {
                "value": revenue_per_customer,
                "percentile": self._calculate_percentile(
                    "revenue_per_customer",
                    revenue_per_customer,
                    business_profile.industry
                )
            }
        
        return metrics

    def _calculate_market_metrics(
        self,
        business_profile: BusinessProfile,
        benchmarks: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate market performance metrics."""
        metrics = {}
        
        # Market share
        if business_profile.market_metrics.total_market_size:
            market_share = (
                business_profile.financial_metrics.revenue /
                business_profile.market_metrics.total_market_size
            )
            metrics["market_share"] = {
                "value": market_share,
                "percentile": self._calculate_percentile(
                    "market_share",
                    market_share,
                    business_profile.industry
                )
            }
        
        # Competitive position
        if business_profile.market_metrics.competitor_count:
            metrics["competitive_position"] = {
                "value": business_profile.market_metrics.competitor_count,
                "benchmark": benchmarks.get("avg_competitor_count")
            }
        
        return metrics

    def _calculate_percentile(
        self,
        metric: str,
        value: float,
        industry: str
    ) -> float:
        """Calculate percentile rank for a metric within the industry."""
        try:
            pipeline = [
                {"$match": {"industry": industry}},
                {"$sort": {metric: 1}},
                {
                    "$group": {
                        "_id": None,
                        "values": {"$push": f"${metric}"}
                    }
                }
            ]
            
            result = list(self.db.businesses.aggregate(pipeline))
            if not result:
                return 0.5
                
            values = result[0]["values"]
            return np.percentile(values, value) / 100
        except Exception as e:
            print(f"Error calculating percentile: {str(e)}")
            return 0.5
