from typing import Dict, Any, Optional, List
import streamlit as st
from core.database import DatabaseConnection
from core.llm_service import LLMService
from services.valuation_service import ValuationService
from services.assessment_service import AssessmentService
from services.analytics_service import AnalyticsService
from models.business_profile import BusinessProfile, FinancialMetrics, MarketMetrics, OperationalMetrics

# Initialize services with proper dependencies
@st.cache_resource
def init_services():
    # Initialize database connection
    db = DatabaseConnection()
    
    # Initialize services with database connection
    llm_service = LLMService()
    valuation_service = ValuationService(db=db)
    assessment_service = AssessmentService(db=db, llm_service=llm_service)
    analytics_service = AnalyticsService(db=db)
    
    return {
        'db': db,
        'llm_service': llm_service,
        'valuation_service': valuation_service,
        'assessment_service': assessment_service,
        'analytics_service': analytics_service
    }

# Initialize all services
services = init_services()

# Extract services from the initialized dictionary
db = services['db']
llm_service = services['llm_service']
valuation_service = services['valuation_service']
assessment_service = services['assessment_service']
analytics_service = services['analytics_service']

def main():
    st.title("Business Insights Hub")
    
    # Add your Streamlit UI components and logic here
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Business Assessment", "Valuation Analysis", "Market Analytics"]
    )
    
    if page == "Business Assessment":
        show_assessment_page()
    elif page == "Valuation Analysis":
        show_valuation_page()
    else:
        show_analytics_page()

def show_assessment_page():
    st.header("Business Assessment")
    # Add your assessment page logic here
    pass

def show_valuation_page():
    st.header("Valuation Analysis")
    # Add your valuation page logic here
    pass

def show_analytics_page():
    st.header("Market Analytics")
    # Add your analytics page logic here
    pass

if __name__ == "__main__":
    main()
