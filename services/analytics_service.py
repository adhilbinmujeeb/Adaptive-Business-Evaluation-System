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
                    "deal_
