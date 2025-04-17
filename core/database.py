import pymongo
from pymongo import MongoClient
import time
from typing import Optional, Dict, Any
from .config import MONGO_URI, DB_NAME

class DatabaseConnection:
    _instance = None
    _client = None
    _db = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._client:
            self._initialize_connection()

    def _initialize_connection(self, max_retries: int = 3) -> None:
        """Initialize MongoDB connection with retry logic."""
        for attempt in range(max_retries):
            try:
                self._client = MongoClient(MONGO_URI)
                # Test the connection
                self._client.admin.command('ismaster')
                self._db = self._client[DB_NAME]
                print("MongoDB connection successful.")
                return
            except pymongo.errors.ConnectionFailure as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise Exception("Failed to connect to MongoDB after multiple retries")

    @property
    def db(self):
        """Get the database instance."""
        if not self._client:
            self._initialize_connection()
        return self._db

    def get_collection(self, collection_name: str):
        """Get a specific collection."""
        return self.db[collection_name]

    def find_similar_businesses(self, industry: str, limit: int = 5) -> list:
        """Find similar businesses in the same industry."""
        try:
            return list(self.db.business_listings.find({
                "business_basics.industry_category": {
                    "$elemMatch": {"$regex": f"^{industry}$", "$options": "i"}
                }
            }).limit(limit))
        except Exception as e:
            print(f"Error finding similar businesses: {e}")
            return []

    def get_industry_metrics(self, industry: str) -> Dict[str, Any]:
        """Get industry-specific metrics from historical data."""
        try:
            pipeline = [
                {
                    "$match": {
                        "business_basics.industry_category": {
                            "$elemMatch": {"$regex": f"^{industry}$", "$options": "i"}
                        }
                    }
                },
                {
                    "$group": {
                        "_id": None,
                        "avg_valuation": {"$avg": "$pitch_metrics.implied_valuation"},
                        "avg_revenue_multiple": {
                            "$avg": {
                                "$cond": [
                                    {"$gt": ["$business_metrics.basic_metrics.revenue", 0]},
                                    {"$divide": ["$pitch_metrics.implied_valuation", "$business_metrics.basic_metrics.revenue"]},
                                    None
                                ]
                            }
                        },
                        "deal_count": {"$sum": 1},
                        "successful_deals": {
                            "$sum": {
                                "$cond": [
                                    {"$ne": ["$deal_outcome.final_result", "no deal"]},
                                    1,
                                    0
                                ]
                            }
                        }
                    }
                }
            ]
            
            result = list(self.db.business_listings.aggregate(pipeline))
            if result:
                return result[0]
            return {}
        except Exception as e:
            print(f"Error getting industry metrics: {e}")
            return {}

    def get_relevant_questions(self, business_stage: str, category: str, limit: int = 5) -> list:
        """Get relevant questions based on business stage and category."""
        try:
            return list(self.db.questions.find({
                "category": category,
                "business_stage": business_stage
            }).limit(limit))
        except Exception as e:
            print(f"Error getting relevant questions: {e}")
            return []

    def close(self):
        """Close the database connection."""
        if self._client:
            self._client.close()
            self._client = None
