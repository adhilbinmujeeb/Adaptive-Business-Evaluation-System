import streamlit as st
from datetime import datetime
from components.ui_components import render_sidebar, render_valuation_page, render_assessment_page, render_trends_page
from core.llm_service import LLMService
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Page configuration
st.set_page_config(
    page_title="Business Insights Hub",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3 { color: #1E3A8A; }
    .stButton>button { background-color: #1E3A8A; color: white; border-radius: 6px; padding: 0.5rem 1rem; font-weight: 500; }
    .stButton>button:hover { background-color: #2563EB; border-color: #2563EB; }
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; }
    .stTabs [data-baseweb="tab"] { height: 3rem; white-space: pre-wrap; border-radius: 4px 4px 0 0; padding: 0.5rem 1rem; background-color: #E2E8F0; }
    .stTabs [aria-selected="true"] { background-color: #1E3A8A !important; color: white !important; }
    div[data-testid="stSidebar"] { background-color: #F8FAFC; padding-top: 1.5rem; }
    .card { background-color: #F8FAFC; border-radius: 0.5rem; padding: 1.5rem; margin-bottom: 1rem; border: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .metric-card { background-color: #EFF6FF; border-radius: 0.5rem; padding: 1rem; text-align: center; border: 1px solid #BFDBFE; }
    .metric-value { font-size: 1.5rem; font-weight: bold; color: #1E3A8A; }
    .metric-label { font-size: 0.875rem; color: #64748B; }
    .sidebar-header { padding: 0.5rem 1rem; margin-bottom: 1rem; border-bottom: 1px solid #E2E8F0; }
    .similar-biz-card { border: 1px solid #CBD5E1; border-radius: 0.375rem; padding: 1rem; margin-bottom: 0.75rem; background-color: #FFFFFF; }
</style>
""", unsafe_allow_html=True)

# Initialize LLM Service
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please set it in .env.")
    st.stop()
llm_service = LLMService(api_key=GROQ_API_KEY, default_provider="groq")

# Session State Initialization
if 'valuation_data' not in st.session_state:
    st.session_state.valuation_data = {}
if 'assessment_responses' not in st.session_state:
    st.session_state.assessment_responses = {}
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'question_count' not in st.session_state:
    st.session_state.question_count = 0
if 'assessment_completed' not in st.session_state:
    st.session_state.assessment_completed = False
if 'current_assessment_question' not in st.session_state:
    st.session_state.current_assessment_question = "Tell me about your business and what problem you're solving."
if 'assessment_industry' not in st.session_state:
    st.session_state.assessment_industry = "Other"
if 'assessment_stage' not in st.session_state:
    st.session_state.assessment_stage = "Early Revenue"
if 'valuation_step' not in st.session_state:
    st.session_state.valuation_step = 0

# Main Application
def main():
    # Sidebar Navigation
    page = render_sidebar(llm_service)

    # Page Rendering
    if "Company Valuation" in page:
        render_valuation_page(llm_service)
    elif "Business Assessment" in page:
        render_assessment_page(llm_service)
    elif "Industry Trends" in page:
        render_trends_page()

    # Footer
    st.markdown(f"""
    <hr style='margin: 2rem 0;'>
    <div style='padding: 1rem; text-align: center; font-size: 0.8rem; color: #64748B;'>
        Business Insights Hub © {datetime.now().year} | Powered by Groq
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
