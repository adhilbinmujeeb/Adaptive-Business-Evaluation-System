from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from ..core.database import DatabaseConnection

class AnalyticsService:
    def __init__(self):
        self.db = DatabaseConnection()

    def get_industry_metrics(self, industry: str) -> Dict[str, Any]:
        """Get comprehensive industry metrics."""
        return self.db.get_industry_metrics(industry)

    def analyze_market_trends(
        self,
        industry: str,
        timeframe: str = "1y"
    ) -> Dict[str, Any]:
        """Analyze market trends for an industry."""
        # Convert timeframe to datetime
        end_date = datetime.now()
        if timeframe == "1y":
            start_date = end_date.replace(year=end_date.year - 1)
        elif timeframe == "3y":
            start_date = end_date.replace(year=end_date.year - 3)
        else:
            start_date = end_date.replace(year=end_date.year - 5)

        # Get historical data
        pipeline = [
            {
                "$match": {
                    "business_basics.industry_category": industry,
                    "metadata.timestamp": {
                        "$gte": start_date,
                        "$lte": end_date
                    }
                }
            },
            {
                "$group": {
                    "_id": {
                        "year": {"$year": "$metadata.timestamp"},
                        "month": {"$month": "$metadata.timestamp"}
                    },
                    "avg_valuation": {"$avg": "$pitch_metrics.implied_valuation"},
                                        "deal_count": {"$sum": 1},
                    "success_rate": {
                        "$avg": {
                            "$cond": [
                                {"$ne": ["$deal_outcome.final_result", "no deal"]},
                                1,
                                0
                            ]
                        }
                    }
                }
            },
            {"$sort": {"_id.year": 1, "_id.month": 1}}
        ]

        results = list(self.db.get_collection("business_listings").aggregate(pipeline))
        
        # Process results
        trend_data = {
            "timeline": [],
            "valuations": [],
            "deal_counts": [],
            "success_rates": []
        }
        
        for r in results:
            date_str = f"{r['_id']['year']}-{r['_id']['month']:02d}"
            trend_data["timeline"].append(date_str)
            trend_data["valuations"].append(r["avg_valuation"])
            trend_data["deal_counts"].append(r["deal_count"])
            trend_data["success_rates"].append(r["success_rate"])

        # Calculate trend indicators
        trend_data["metrics"] = {
            "valuation_trend": self._calculate_trend(trend_data["valuations"]),
            "deal_volume_trend": self._calculate_trend(trend_data["deal_counts"]),
            "success_rate_trend": self._calculate_trend(trend_data["success_rates"])
        }

        return trend_data

    def get_comparable_businesses(
        self,
        business_profile: Dict[str, Any],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get comparable businesses with similar characteristics."""
        industry = business_profile.get("industry")
        stage = business_profile.get("business_stage")
        revenue_range = self._get_revenue_range(business_profile.get("financial_metrics", {}).get("revenue"))

        pipeline = [
            {
                "$match": {
                    "business_basics.industry_category": industry,
                    "business_basics.stage": stage,
                    "business_metrics.basic_metrics.revenue": {
                        "$gte": revenue_range["min"],
                        "$lte": revenue_range["max"]
                    }
                }
            },
            {
                "$project": {
                    "business_name": "$business_basics.business_name",
                    "valuation": "$pitch_metrics.implied_valuation",
                    "revenue": "$business_metrics.basic_metrics.revenue",
                    "deal_outcome": 1,
                    "key_metrics": "$business_metrics.basic_metrics"
                }
            },
            {"$limit": limit}
        ]

        return list(self.db.get_collection("business_listings").aggregate(pipeline))

    def generate_performance_metrics(
        self,
        business_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate key performance metrics and benchmarks."""
        industry = business_profile.get("industry")
        stage = business_profile.get("business_stage")
        
        # Get industry benchmarks
        benchmarks = self._get_industry_benchmarks(industry, stage)
        
        # Calculate company metrics
        metrics = {
            "financial": self._calculate_financial_metrics(business_profile, benchmarks),
            "operational": self._calculate_operational_metrics(business_profile, benchmarks),
            "market": self._calculate_market_metrics(business_profile, benchmarks)
        }
        
        # Add percentile rankings
        metrics["rankings"] = self._calculate_percentile_rankings(metrics, benchmarks)
        
        return metrics

    def _calculate_trend(self, data: List[float]) -> Dict[str, Any]:
        """Calculate trend indicators for a time series."""
        if not data:
            return {"direction": "neutral", "strength": 0}
            
        # Calculate simple moving average
        window = min(3, len(data))
        sma = pd.Series(data).rolling(window=window).mean().tolist()[-3:]
        
        # Calculate trend direction and strength
        if len(sma) >= 2:
            change = (sma[-1] - sma[0]) / sma[0] if sma[0] else 0
            direction = "up" if change > 0.05 else "down" if change < -0.05 else "neutral"
            strength = abs(change)
        else:
            direction = "neutral"
            strength = 0
            
        return {
            "direction": direction,
            "strength": strength,
            "recent_values": sma
        }

    def _get_revenue_range(self, revenue: float) -> Dict[str, float]:
        """Get revenue range for comparable company search."""
        if not revenue:
            return {"min": 0, "max": float('inf')}
            
        # Create range of ±50% of the company's revenue
        return {
            "min": revenue * 0.5,
            "max": revenue * 1.5
        }

    def _get_industry_benchmarks(
        self,
        industry: str,
        stage: str
    ) -> Dict[str, Any]:
        """Get industry benchmarks from the database."""
        pipeline = [
            {
                "$match": {
                    "business_basics.industry_category": industry,
                    "business_basics.stage": stage
                }
            },
            {
                "$group": {
                    "_id": None,
                    "revenue_stats": {
                        "$push": "$business_metrics.basic_metrics.revenue"
                    },
                    "margin_stats": {
                        "$push": "$business_metrics.basic_metrics.margins"
                    },
                    "valuation_stats": {
                        "$push": "$pitch_metrics.implied_valuation"
                    }
                }
            },
            {
                "$project": {
                    "revenue": {
                        "median": {"$median": "$revenue_stats"},
                        "p75": {"$percentile": ["$revenue_stats", 75]},
                        "p25": {"$percentile": ["$revenue_stats", 25]}
                    },
                    "margins": {
                        "median": {"$median": "$margin_stats"},
                        "p75": {"$percentile": ["$margin_stats", 75]},
                        "p25": {"$percentile": ["$margin_stats", 25]}
                    },
                    "valuation": {
                        "median": {"$median": "$valuation_stats"},
                        "p75": {"$percentile": ["$valuation_stats", 75]},
                        "p25": {"$percentile": ["$valuation_stats", 25]}
                    }
                }
            }
        ]

        result = list(self.db.get_collection("business_listings").aggregate(pipeline))
        return result[0] if result else {}

    def _calculate_financial_metrics(
        self,
        profile: Dict[str, Any],
        benchmarks: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate financial performance metrics."""
        financials = profile.get("financial_metrics", {})
        
        metrics = {
            "revenue_growth": self._calculate_growth_rate(
                financials.get("revenue_history", [])
            ),
            "gross_margin": financials.get("gross_margin"),
            "burn_rate": financials.get("burn_rate"),
            "runway": self._calculate_runway(
                financials.get("cash", 0),
                financials.get("burn_rate", 0)
            )
        }
        
        # Add benchmark comparisons
        if benchmarks.get("revenue"):
            metrics["revenue_percentile"] = self._calculate_percentile(
                financials.get("revenue", 0),
                benchmarks["revenue"]
            )
            
        return metrics

    def _calculate_operational_metrics(
        self,
        profile: Dict[str, Any],
        benchmarks: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate operational performance metrics."""
        operations = profile.get("operational_metrics", {})
        
        return {
            "customer_acquisition_cost": operations.get("cac"),
            "lifetime_value": operations.get("ltv"),
            "ltv_cac_ratio": self._calculate_ratio(
                operations.get("ltv"),
                operations.get("cac")
            ),
            "retention_rate": operations.get("retention_rate"),
            "efficiency_score": self._calculate_efficiency_score(operations)
        }

    def _calculate_market_metrics(
        self,
        profile: Dict[str, Any],
        benchmarks: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate market performance metrics."""
        market = profile.get("market_metrics", {})
        
        return {
            "market_share": market.get("market_share"),
            "market_growth": market.get("market_growth_rate"),
            "competitive_position": self._assess_competitive_position(
                market,
                benchmarks
            ),
            "market_penetration": self._calculate_penetration(
                market.get("current_customers"),
                market.get("total_addressable_market")
            )
        }

    def _calculate_percentile_rankings(
        self,
        metrics: Dict[str, Any],
        benchmarks: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate percentile rankings for key metrics."""
        rankings = {}
        
        for category, category_metrics in metrics.items():
            for metric, value in category_metrics.items():
                if isinstance(value, (int, float)) and metric in benchmarks:
                    rankings[metric] = self._calculate_percentile(
                        value,
                        benchmarks[metric]
                    )
                    
        return rankings

    def _calculate_growth_rate(self, history: List[float]) -> Optional[float]:
        """Calculate compound annual growth rate."""
        if not history or len(history) < 2:
            return None
            
        start_value = history[0]
        end_value = history[-1]
        periods = len(history) - 1
        
        if start_value <= 0:
            return None
            
        return (end_value / start_value) ** (1/periods) - 1

    def _calculate_runway(
        self,
        cash: float,
        burn_rate: float
    ) -> Optional[float]:
        """Calculate cash runway in months."""
        if not burn_rate:
            return None
        return cash / burn_rate

    def _calculate_ratio(
        self,
        numerator: Optional[float],
        denominator: Optional[float]
    ) -> Optional[float]:
        """Safely calculate ratio between two numbers."""
        if not numerator or not denominator:
            return None
        return numerator / denominator if denominator != 0 else None

    def _calculate_efficiency_score(
        self,
        operations: Dict[str, Any]
    ) -> Optional[float]:
        """Calculate operational efficiency score."""
        metrics = [
            operations.get("ltv_cac_ratio", 0),
            operations.get("retention_rate", 0),
            operations.get("conversion_rate", 0)
        ]
        
        if not any(metrics):
            return None
            
        weights = [0.4, 0.4, 0.2]
        return sum(m * w for m, w in zip(metrics, weights) if m is not None)

    def _calculate_percentile(
        self,
        value: float,
        distribution: Dict[str, float]
    ) -> float:
        """Calculate percentile of a value within a distribution."""
        if not value or not distribution:
            return 0.0
            
        median = distribution.get("median", 0)
        p75 = distribution.get("p75", median * 1.5)
        p25 = distribution.get("p25", median * 0.5)
        
        if value >= p75:
            return 0.75 + 0.25 * min((value - p75) / (p75 - median), 1)
        elif value >= median:
            return 0.5 + 0.25 * ((value - median) / (p75 - median))
        elif value >= p25:
            return 0.25 + 0.25 * ((value - p25) / (median - p25))
        else:
            return 0.25 * (value / p25)

    def _assess_competitive_position(
        self,
        market: Dict[str, Any],
        benchmarks: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess competitive position in the market."""
        market_share = market.get("market_share", 0)
        growth_rate = market.get("market_growth_rate", 0)
        competitor_count = len(market.get("competitors", []))
        
        position = {
            "market_position": "follower",
            "growth_alignment": "neutral",
            "competitive_pressure": "medium"
        }
        
        # Assess market position
        if market_share > 0.2:
            position["market_position"] = "leader"
        elif market_share > 0.1:
            position["market_position"] = "challenger"
            
        # Assess growth alignment
        market_growth = market.get("market_growth_rate", 0)
        if growth_rate > market_growth * 1.2:
            position["growth_alignment"] = "outperforming"
        elif growth_rate < market_growth * 0.8:
            position["growth_alignment"] = "underperforming"
            
        # Assess competitive pressure
        if competitor_count < 5:
            position["competitive_pressure"] = "low"
        elif competitor_count > 15:
            position["competitive_pressure"] = "high"
            
        return position

    def _calculate_penetration(
        self,
        current_customers: Optional[int],
        total_market: Optional[int]
    ) -> Optional[float]:
        """Calculate market penetration rate."""
        if not current_customers or not total_market:
            return None
        return current_customers / total_market if total_market > 0 else 0
