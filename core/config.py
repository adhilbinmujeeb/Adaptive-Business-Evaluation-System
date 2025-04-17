import os
from dotenv import load_dotenv
from enum import Enum

# Load environment variables
load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "business_rag"

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Business Stage Definitions
class BusinessStage(Enum):
    IDEA = "Idea/Concept Stage"
    STARTUP = "Startup/Early Stage"
    GROWTH = "Growth Stage"
    MATURE = "Mature Stage"
    TURNAROUND = "Turnaround/Restructuring Stage"

# Industry Categories
INDUSTRY_CATEGORIES = [
    "Software/SaaS",
    "E-commerce",
    "Manufacturing",
    "Services",
    "Retail",
    "Healthcare",
    "Food & Beverage",
    "Real Estate",
    "Financial Services",
    "Media & Entertainment",
    "Education",
    "Non-profit",
    "Biotech/Pharma"
]

# Functional Areas
FUNCTIONAL_AREAS = [
    "Marketing & Sales",
    "Operations & Logistics",
    "Finance & Accounting",
    "Human Resources",
    "Research & Development",
    "Customer Service",
    "IT & Technology",
    "Legal & Compliance"
]

# Business Models
BUSINESS_MODELS = [
    "Subscription-based",
    "Marketplace/Platform",
    "Direct-to-Consumer",
    "Franchise",
    "Service-based"
]

# Strategic Focus Areas
STRATEGIC_FOCUS = [
    "Innovation-focused",
    "Cost Leadership",
    "Differentiation",
    "Market Expansion",
    "Acquisition/Partnership",
    "Environmental, Social & Governance",
    "Digital Transformation",
    "Crisis Management",
    "International Expansion",
    "Exit Strategy"
]

# Valuation Configuration
VALUATION_METRICS = {
    "Software/SaaS": {
        "revenue_multiple_ranges": {
            "early": {"min": 5, "max": 10},
            "growth": {"min": 10, "max": 15},
            "mature": {"min": 3, "max": 8}
        },
        "ebitda_multiple_ranges": {
            "early": {"min": 12, "max": 15},
            "growth": {"min": 15, "max": 20},
            "mature": {"min": 8, "max": 12}
        }
    },
    "default": {
        "revenue_multiple_ranges": {
            "early": {"min": 2, "max": 4},
            "growth": {"min": 3, "max": 6},
            "mature": {"min": 1, "max": 3}
        },
        "ebitda_multiple_ranges": {
            "early": {"min": 8, "max": 12},
            "growth": {"min": 10, "max": 15},
            "mature": {"min": 6, "max": 10}
        }
    }
}

# Assessment Configuration
MAX_QUESTIONS = 25  # Maximum number of questions in assessment
MIN_QUESTIONS = 10  # Minimum number of questions in assessment

# Question weights by business stage
QUESTION_WEIGHTS = {
    BusinessStage.IDEA.value: {
        "Business Fundamentals": 0.3,
        "Market Analysis": 0.3,
        "Financial Planning": 0.2,
        "Team & Experience": 0.2
    },
    BusinessStage.STARTUP.value: {
        "Business Fundamentals": 0.25,
        "Market Analysis": 0.25,
        "Financial Metrics": 0.3,
        "Team & Experience": 0.2
    },
    BusinessStage.GROWTH.value: {
        "Financial Metrics": 0.35,
        "Market Position": 0.25,
        "Operational Efficiency": 0.25,
        "Team & Leadership": 0.15
    },
    BusinessStage.MATURE.value: {
        "Financial Performance": 0.4,
        "Market Leadership": 0.2,
        "Innovation Strategy": 0.2,
        "Risk Management": 0.2
    },
    BusinessStage.TURNAROUND.value: {
        "Financial Recovery": 0.4,
        "Operational Restructuring": 0.3,
        "Market Repositioning": 0.2,
        "Leadership & Change": 0.1
    }
}

# Confidence Score Thresholds
CONFIDENCE_THRESHOLDS = {
    "high": 0.8,
    "medium": 0.6,
    "low": 0.4
}
