
"""
Core package for business insights hub.
Contains core functionality for database connection, configuration, and LLM services.
"""

from .config import (
    BusinessStage,
    INDUSTRY_CATEGORIES,
    FUNCTIONAL_AREAS,
    BUSINESS_MODELS,
    STRATEGIC_FOCUS
)
from .database import DatabaseConnection
from .llm_service import LLMService

__all__ = [
    'BusinessStage',
    'INDUSTRY_CATEGORIES',
    'FUNCTIONAL_AREAS',
    'BUSINESS_MODELS',
    'STRATEGIC_FOCUS',
    'DatabaseConnection',
    'LLMService'
]
