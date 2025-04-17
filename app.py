import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional

from core.config import (
    BusinessStage, INDUSTRY_CATEGORIES, FUNCTIONAL_AREAS,
    BUSINESS_MODELS, STRATEGIC_FOCUS
)
from core.database import DatabaseConnection
from core.llm_service import LLMService
from services.valuation_service import ValuationService
from services.assessment_service import AssessmentService
from services.analytics_service import AnalyticsService
from models.business_profile import BusinessProfile, FinancialMetrics, MarketMetrics

# Initialize services
db = DatabaseConnection()
llm = LLMService()
valuation_service = ValuationService()
assessment_service = AssessmentService()
analytics_service = AnalyticsService()

# Set page configuration
st.set_page_config(
    page_title="Business Insights Hub",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #2563EB;
        border-color: #2563EB;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        border-radius: 4px 4px 0 0;
        padding: 0.5rem 1rem;
        background-color: #E2E8F0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A !important;
        color: white !important;
    }
    div[data-testid="stSidebar"] {
        background-color: #F8FAFC;
        padding-top: 1.5rem;
    }
    .card {
        background-color: #F8FAFC;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #EFF6FF;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        border: 1px solid #BFDBFE;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #64748B;
    }
    .similar-biz-card {
        border: 1px solid #CBD5E1;
        border-radius: 0.375rem;
        padding: 1rem;
        margin-bottom: 0.75rem;
        background-color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
    st.title("💼 Business Insights Hub")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Navigation")
    page = st.radio("", [
        "💰 Company Valuation",
        "📊 Business Assessment",
    ], key="main_nav")

    st.markdown("---")
    st.info("Using Groq API (Llama 4 Scout)")
    st.markdown(f"<div style='text-align: center; padding: 1rem; font-size: 0.8rem; color: #64748B;'>{datetime.now().strftime('%B %d, %Y')}</div>", unsafe_allow_html=True)

# Initialize session state
if 'valuation_data' not in st.session_state:
    st.session_state.valuation_data = {}
if 'assessment_session' not in st.session_state:
    st.session_state.assessment_session = None
if 'current_question_idx' not in st.session_state:
    st.session_state.current_question_idx = 0

def format_currency(value: float) -> str:
    """Format currency values with appropriate suffix (K, M, B)."""
    if value >= 1e9:
        return f"${value/1e9:.1f}B"
    elif value >= 1e6:
        return f"${value/1e6:.1f}M"
    elif value >= 1e3:
        return f"${value/1e3:.1f}K"
    else:
        return f"${value:.2f}"

def create_trend_chart(data: Dict[str, List], title: str) -> go.Figure:
    """Create a trend chart using Plotly."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data["timeline"],
        y=data["values"],
        mode='lines+markers',
        name=title,
        line=dict(color='#1E3A8A', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        template="plotly_white",
        height=400,
        showlegend=False
    )
    
    return fig

def display_metric_card(label: str, value: Any, suffix: str = "") -> None:
    """Display a metric in a styled card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{value}{suffix}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Company Valuation Page
if "Company Valuation" in page:
    st.markdown("# 💰 Company Valuation Estimator")
    st.markdown("Estimate your company's value and see how it compares to similar pitches.")

    # Business Profile Collection
    st.markdown("## Business Profile")
    with st.form("business_profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            company_name = st.text_input("Company Name", key="company_name")
            industry = st.selectbox("Industry", options=INDUSTRY_CATEGORIES, key="industry")
            business_stage = st.selectbox("Business Stage", options=[stage.value for stage in BusinessStage], key="stage")
            business_model = st.selectbox("Business Model", options=BUSINESS_MODELS, key="business_model")

        with col2:
            revenue = st.number_input("Annual Revenue (USD)", min_value=0.0, step=1000.0, key="revenue")
            profit = st.number_input("Net Profit (USD)", step=1000.0, key="profit")
            ebitda = st.number_input("EBITDA (USD)", step=1000.0, key="ebitda")
            growth_rate = st.selectbox("Growth Rate", options=["High", "Moderate", "Low"], key="growth")

        st.markdown("### Additional Information")
        col3, col4 = st.columns(2)
        
        with col3:
            assets = st.number_input("Total Assets (USD)", min_value=0.0, step=1000.0, key="assets")
            liabilities = st.number_input("Total Liabilities (USD)", min_value=0.0, step=1000.0, key="liabilities")

        with col4:
            market_size = st.number_input("Target Market Size (USD)", min_value=0.0, step=1000.0, key="market_size")
            competitors = st.multiselect("Main Competitors", options=[], key="competitors")

        cash_flows = st.text_input(
            "Projected Cash Flows (next 5 years, comma-separated in USD)",
            help="Enter projected cash flows for the next 5 years, separated by commas",
            key="cash_flows"
        )

        submit_button = st.form_submit_button("Calculate Valuation")

        if submit_button:
            if not company_name:
                st.error("Please enter a company name.")
            else:
                # Create business profile
                financial_metrics = FinancialMetrics(
                    revenue=revenue,
                    profit=profit,
                    ebitda=ebitda
                )
                
                market_metrics = MarketMetrics(
                    total_market_size=market_size,
                    competitors=competitors
                )
                
                business_profile = BusinessProfile(
                    business_name=company_name,
                    industry=industry,
                    business_stage=business_stage,
                    description="",  # Could add a description field if needed
                    business_model=business_model,
                    financial_metrics=financial_metrics,
                    market_metrics=market_metrics
                )

                # Calculate valuation
                with st.spinner("Calculating valuation..."):
                    # Get industry metrics
                    industry_metrics = analytics_service.get_industry_metrics(industry)
                    
                    # Calculate valuations using different methods
                    valuation_results = []
                    
                    # Revenue multiple valuation
                    if revenue > 0:
                        revenue_val = valuation_service.calculate_revenue_multiple_valuation(
                            business_profile,
                            industry_metrics
                        )
                        if revenue_val:
                            valuation_results.append(revenue_val)
                    
                    # EBITDA multiple valuation
                    if ebitda > 0:
                        ebitda_val = valuation_service.calculate_ebitda_multiple_valuation(
                            business_profile,
                            industry_metrics
                        )
                        if ebitda_val:
                            valuation_results.append(ebitda_val)
                    
                    # DCF valuation
                    if cash_flows:
                        try:
                            cf_list = [float(cf.strip()) for cf in cash_flows.split(",")]
                            if len(cf_list) == 5:
                                dcf_val = valuation_service.calculate_dcf_valuation(
                                    business_profile,
                                    cf_list
                                )
                                if dcf_val:
                                    valuation_results.append(dcf_val)
                        except ValueError:
                            st.warning("Invalid cash flow projections format.")

                    # Generate valuation summary
                    if valuation_results:
                        valuation_summary = valuation_service.generate_valuation_summary(
                            business_profile,
                            valuation_results
                        )
                        
                        st.session_state.valuation_data = valuation_summary.to_dict()
                        st.rerun()

    # Display Valuation Results
    if st.session_state.valuation_data:
        st.markdown("## Valuation Results")
        
        # Display recommended range
        range_data = st.session_state.valuation_data["recommended_range"]
        col1, col2, col3 = st.columns(3)
        
        with col1:
            display_metric_card(
                "Conservative Estimate",
                format_currency(range_data["low"])
            )
        with col2:
            display_metric_card(
                "Base Estimate",
                format_currency(range_data["mid"])
            )
        with col3:
            display_metric_card(
                "Optimistic Estimate",
                format_currency(range_data["high"])
            )

        # Display valuation methods
        st.markdown("### Valuation Methods")
        for result in st.session_state.valuation_data["results"]:
            with st.expander(f"{result['method']} Valuation"):
                st.markdown(f"**Value:** {format_currency(result['value'])}")
                st.markdown(f"**Confidence Score:** {result['confidence_score']:.2%}")
                
                if result.get("multiplier_used"):
                    st.markdown(f"**Multiple Used:** {result['multiplier_used']:.2f}x")
                
                if result.get("assumptions"):
                    st.markdown("**Key Assumptions:**")
                    for key, value in result["assumptions"].items():
                        st.markdown(f"- {key.replace('_', ' ').title()}: {value}")

        # Display key factors and risks
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Key Value Drivers")
            for factor in st.session_state.valuation_data["key_factors"]:
                st.markdown(f"- {factor}")
                
        with col2:
            st.markdown("### Risk Factors")
            for risk in st.session_state.valuation_data["risk_factors"]:
                st.markdown(f"- {risk}")

        # Display similar businesses
        st.markdown("## Comparable Companies")
        similar_businesses = valuation_service.get_comparable_companies(
            BusinessProfile.from_dict(st.session_state.valuation_data)
        )
        
        if similar_businesses:
            for biz in similar_businesses:
                with st.container():
                    st.markdown(
                        f"""
                        <div class="similar-biz-card">
                            <h4>{biz['business_name']}</h4>
                            <p><strong>Valuation:</strong> {format_currency(biz['valuation'])}</p>
                            <p><strong>Revenue:</strong> {format_currency(biz['revenue'])}</p>
                            {f"<p><strong>Deal Outcome:</strong> {biz['deal_outcome']['final_result']}</p>" if biz.get('deal_outcome') else ""}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.info("No comparable companies found in the database.")

# Business Assessment Page
elif "Business Assessment" in page:
    st.markdown("# 📊 Interactive Business Assessment")
    st.markdown("Get personalized insights through an adaptive business evaluation.")

    # Initialize or continue assessment session
    if not st.session_state.assessment_session:
        st.markdown("## Start New Assessment")
        
        with st.form("start_assessment"):
            business_name = st.text_input("Business Name")
            industry = st.selectbox("Industry", options=INDUSTRY_CATEGORIES)
            business_stage = st.selectbox("Business Stage", options=[stage.value for stage in BusinessStage])
            
            start_button = st.form_submit_button("Start Assessment")
            
            if start_button and business_name:
                st.session_state.assessment_session = assessment_service.start_assessment(
                    business_name,
                    business_stage,
                    industry
                )
                st.rerun()

    # Continue existing assessment
    elif not st.session_state.assessment_session.completion_status >= 1.0:
        st.progress(st.session_state.assessment_session.completion_status)
        
        # Get next question
        current_question = assessment_service.get_next_question(
            st.session_state.assessment_session,
            st.session_state.assessment_session.questions_answers
        )
        
        if current_question:
            st.markdown(f"### Question {len(st.session_state.assessment_session.questions_answers) + 1}")
            st.markdown(f"**{current_question.text}**")
            
            response = st.text_area("Your Answer", height=100)
            
            if st.button("Submit Answer"):
                if response.strip():
                    # Process answer
                    answer = assessment_service.process_answer(
                        st.session_state.assessment_session,
                        current_question,
                        response
                    )
                    
                    st.rerun()
                else:
                    st.warning("Please provide an answer before submitting.")

    # Display assessment results
    else:
        st.markdown("## Assessment Results")
        
        # Generate final assessment
        assessment_result = assessment_service.generate_assessment_result(
            st.session_state.assessment_session
        )
        
        # Display category scores
        st.markdown("### Performance by Category")
        cols = st.columns(len(assessment_result.scores))
        
        for col, (category, score) in zip(cols, assessment_result.scores.items()):
            with col:
                display_metric_card(
                    category,
                    f"{score:.1%}"
                )

        # Display recommendations
        st.markdown("### Strategic Recommendations")
        for i, rec in enumerate(assessment_result.recommendations, 1):
            st.markdown(f"{i}. {rec}")

        # Display opportunities and risks
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Growth Opportunities")
            for opp in assessment_result.opportunities:
                st.markdown(f"- {opp}")
                
        with col2:
            st.markdown("### Risk Factors")
            for risk in assessment_result.risk_factors:
                st.markdown(f"- {risk}")

        # Option to start new assessment
        if st.button("Start New Assessment"):
            st.session_state.assessment_session = None
            st.rerun()

# Footer
st.markdown("""
<hr style='margin: 2rem 0;'>
<div style='padding: 1rem; text-align: center; font-size: 0.8rem; color: #64748B;'>
    Business Insights Hub © 2024 | Powered by Groq
</div>
""", unsafe_allow_html=True)
