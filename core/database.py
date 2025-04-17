from typing import List, Dict, Any, Optional
import os
from datetime import datetime
from pymongo import MongoClient

class DatabaseConnection:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Use the provided MongoDB URI
        self.mongo_uri = 'mongodb+srv://adhilbinmujeeb:admin123@cluster0.uz62z.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
        
        try:
            # Initialize MongoDB client
            self.client = MongoClient(self.mongo_uri)
            
            # Select database
            self.db = self.client.business_rag
            
            # Initialize collections
            self.businesses = self.db.business_listings
            self.questions = self.db.questions
            
            # Test connection
            self.client.admin.command('ping')
            print("Successfully connected to MongoDB!")
            
        except Exception as e:
            print(f"Error connecting to MongoDB: {str(e)}")
            # Initialize with empty collections for fallback
            self.businesses = None
            self.questions = None
            
        self._initialized = True
    
    def get_questions_for_stage(self, business_stage: str) -> List[Dict[str, Any]]:
        """Get assessment questions for a specific business stage."""
        try:
            if self.questions:
                questions = list(self.questions.find({"business_stage": business_stage}))
                if questions:
                    return questions
            return self._get_default_questions(business_stage)
        except Exception as e:
            print(f"Error fetching questions: {str(e)}")
            return self._get_default_questions(business_stage)

    def get_question_by_id(self, question_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific question by ID."""
        try:
            if self.questions:
                question = self.questions.find_one({"_id": question_id})
                if question:
                    return question
            # Return from default questions if not found in DB
            for questions in self._get_default_questions("all").values():
                for question in questions:
                    if question["_id"] == question_id:
                        return question
            return None
        except Exception as e:
            print(f"Error fetching question: {str(e)}")
            return None

    def find_similar_businesses(
        self,
        industry: str,
        revenue_range: str,
        business_stage: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar businesses based on criteria."""
        try:
            if not self.businesses:
                return []
                
            min_revenue, max_revenue = self._get_revenue_range_bounds(revenue_range)
            query = {
                "industry": industry,
                "business_stage": business_stage,
                "revenue": {"$gte": min_revenue, "$lt": max_revenue}
            }
            return list(self.businesses.find(query).limit(limit))
        except Exception as e:
            print(f"Error finding similar businesses: {str(e)}")
            return []

    def _get_revenue_range_bounds(self, range_category: str) -> tuple:
        """Get the minimum and maximum revenue values for a range category."""
        ranges = {
            "0-1M": (0, 1_000_000),
            "1M-10M": (1_000_000, 10_000_000),
            "10M-50M": (10_000_000, 50_000_000),
            "50M+": (50_000_000, float('inf'))
        }
        return ranges.get(range_category, (0, float('inf')))

    def _get_default_questions(self, business_stage: str) -> List[Dict[str, Any]]:
        """Provide default questions if database query fails."""
        default_questions = {
            "Startup": [
                {
                    "_id": "q1_startup",
                    "text": "What is your current monthly revenue?",
                    "category": "Financial",
                    "business_stage": "Startup",
                    "follow_up_questions": [
                        "How has this changed over the past 6 months?",
                        "What's your target revenue for the next year?"
                    ]
                },
                {
                    "_id": "q2_startup",
                    "text": "What is your customer acquisition cost (CAC)?",
                    "category": "Marketing",
                    "business_stage": "Startup",
                    "follow_up_questions": [
                        "How does this compare to your lifetime value (LTV)?",
                        "What channels have the lowest CAC?"
                    ]
                }
            ],
            "Growth": [
                {
                    "_id": "q1_growth",
                    "text": "What is your year-over-year growth rate?",
                    "category": "Financial",
                    "business_stage": "Growth",
                    "follow_up_questions": [
                        "What are the main drivers of this growth?",
                        "Is this growth sustainable?"
                    ]
                },
                {
                    "_id": "q2_growth",
                    "text": "What is your market share in your primary market?",
                    "category": "Market",
                    "business_stage": "Growth",
                    "follow_up_questions": [
                        "Who are your main competitors?",
                        "What is your competitive advantage?"
                    ]
                }
            ],
            "Mature": [
                {
                    "_id": "q1_mature",
                    "text": "What is your EBITDA margin?",
                    "category": "Financial",
                    "business_stage": "Mature",
                    "follow_up_questions": [
                        "How has this evolved over the past 3 years?",
                        "What initiatives are in place to improve margins?"
                    ]
                },
                {
                    "_id": "q2_mature",
                    "text": "What is your dividend payout ratio?",
                    "category": "Financial",
                    "business_stage": "Mature",
                    "follow_up_questions": [
                        "How do you balance reinvestment vs. distributions?",
                        "What's your capital allocation strategy?"
                    ]
                }
            ]
        }
        return default_questions.get(business_stage, []) if business_stage != "all" else default_questions

    def close(self):
        """Close the MongoDB connection."""
        if hasattr(self, 'client'):
            try:
                self.client.close()
            except Exception as e:
                print(f"Error closing MongoDB connection: {str(e)}")
