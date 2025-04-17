import streamlit as st
from core.llm_service import LLMService
from core.database import get_database
from components.ui_components import render_valuation_page, render_assessment_page
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def main():
    st.set_page_config(page_title="Business Insights Hub", layout="wide")
    
    # Load custom CSS
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "Valuation"
    if 'valuation_step' not in st.session_state:
        st.session_state.valuation_step = 0
    if 'valuation_data' not in st.session_state:
        st.session_state.valuation_data = {}
    if 'assessment_history' not in st.session_state:
        st.session_state.assessment_history = []
    
    # Initialize services
    llm_service = LLMService(GROQ_API_KEY)
    db = get_database()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Valuation", "Assessment"], index=["Valuation", "Assessment"].index(st.session_state.page))
    st.session_state.page = page
    
    # Render pages
    if page == "Valuation":
        render_valuation_page(llm_service)
    elif page == "Assessment":
        render_assessment_page(llm_service)

if __name__ == "__main__":
    main()
