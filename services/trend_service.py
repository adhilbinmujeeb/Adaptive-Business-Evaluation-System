from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

def analyze_industry_trends(industry, timespan, listings_collection):
    end_date = datetime.utcnow()
    days_map = {"1y": 365, "3y": 3*365, "5y": 5*365}
    start_date = end_date - timedelta(days=days_map.get(timespan, 3*365))

    try:
        pipeline = [
            {
                "$match": {
                    "business_basics.industry_category": {"$elemMatch": {"$eq": industry}},
                    "metadata.episode_date": {"$gte": start_date, "$lte": end_date},
                    "pitch_metrics.implied_valuation": {"$gt": 0}
                }
            },
            {
                "$addFields": {
                    "yearQuarter": {"$dateToString": {"format": "%Y-Q%q", "date": "$metadata.episode_date"}}
                }
            },
            {
                "$group": {
                    "_id": "$yearQuarter",
                    "avg_valuation": {"$avg": "$pitch_metrics.implied_valuation"},
                    "sample_size": {"$sum": 1}
                }
            },
            {"$sort": {"_id": 1}},
            {"$project": {"_id": 0, "period": "$_id", "avg_valuation": 1, "sample_size": 1}}
        ]
        trend_data = list(listings_collection.aggregate(pipeline))
        return {"industry": industry, "timespan": timespan, "trend_data": trend_data}
    except Exception as e:
        st.error(f"Error fetching trend data: {e}")
        return {"error": "Failed to retrieve trend data."}
