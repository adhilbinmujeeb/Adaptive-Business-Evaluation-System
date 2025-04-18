import streamlit as st
import pymongo
from pymongo import MongoClient
import pandas as pd
import numpy as np
from datetime import datetime
import google.generativeai as genai
import os
from dotenv import load_dotenv
import time
import re
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
import json

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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

# Helper Functions
def safe_float(value, default=0.0):
    """Safely converts a value to float, handling $, ,, None, and errors."""
    if value is None:
        return default
    try:
        if isinstance(value, str):
            value = value.replace("$", "").replace(",", "")
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    """Safely converts a value to int, handling float strings, $, ,, None."""
    if value is None:
        return default
    try:
        if isinstance(value, str):
            value = value.replace("$", "").replace(",", "")
        return int(float(value))
    except (ValueError, TypeError):
        return default

# MongoDB Connection with Retry
@st.cache_resource(ttl=3600)
def get_mongo_client():
    for attempt in range(3):
        try:
            client = MongoClient(MONGO_URI)
            client.admin.command('ismaster')
            return client
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                st.error(f"Failed to connect to MongoDB: {e}")
                st.stop()

# Initialize MongoDB connection
client = get_mongo_client()
db = client['business_rag']
listings_collection = db['business_listings']
attributes_collection = db['business_attributes']
questions_collection = db['questions']

# Initialize Gemini
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Failed to configure Gemini API: {e}")
    st.stop()



# Valuation and LLM Functions
def calculate_valuation(company_data):
    """Calculate valuation using multiple methods based on available data."""
    results = {}
    
    # Revenue-based method
    if company_data.get('revenue') and company_data.get('revenue') > 0:
        industry = company_data.get('industry', 'Other')
        growth = company_data.get('growth', 'Moderate')
        
        # Revenue multiples based on industry and growth
        revenue_multiples = {
            'High': {
                'Software/SaaS': 10.0, 
                'E-commerce': 3.5,
                'Technology': 8.0,
                'Healthcare': 4.5,
                'Food & Beverage': 2.5,
                'Other': 3.0
            },
            'Moderate': {
                'Software/SaaS': 6.0,
                'E-commerce': 2.0,
                'Technology': 5.0,
                'Healthcare': 3.0,
                'Food & Beverage': 1.5,
                'Other': 1.5
            },
            'Low': {
                'Software/SaaS': 3.0,
                'E-commerce': 1.0,
                'Technology': 2.5,
                'Healthcare': 1.5,
                'Food & Beverage': 0.8,
                'Other': 0.8
            }
        }
        
        multiple = revenue_multiples[growth].get(industry, revenue_multiples[growth]['Other'])
        results['revenue_valuation'] = company_data['revenue'] * multiple
    
    # Earnings-based method
    if company_data.get('earnings') and company_data.get('earnings') > 0:
        industry = company_data.get('industry', 'Other')
        # Industry PE ratios
        pe_ratios = {
            'Software/SaaS': 25.0,
            'E-commerce': 20.0,
            'Technology': 22.0,
            'Healthcare': 18.0,
            'Food & Beverage': 15.0,
            'Other': 15.0
        }
        pe_multiple = pe_ratios.get(industry, pe_ratios['Other'])
        results['earnings_valuation'] = company_data['earnings'] * pe_multiple
    
    # DCF Method
    if company_data.get('cash_flows') and len(company_data['cash_flows']) > 0:
        discount_rate = 0.12  # Standard 12% discount rate
        terminal_growth = 0.02  # 2% terminal growth rate
        
        # Calculate present value of projected cash flows
        cash_flows = company_data['cash_flows']
        dcf_value = 0
        
        for i, cf in enumerate(cash_flows):
            dcf_value += cf / ((1 + discount_rate) ** (i + 1))
        
        # Terminal value calculation
        if cash_flows[-1] > 0:
            terminal_value = (cash_flows[-1] * (1 + terminal_growth)) / (discount_rate - terminal_growth)
            terminal_value_pv = terminal_value / ((1 + discount_rate) ** len(cash_flows))
            dcf_value += terminal_value_pv
        
        results['dcf_valuation'] = dcf_value
    
    # Asset-based method
    if company_data.get('assets') is not None and company_data.get('liabilities') is not None:
        results['asset_based_valuation'] = company_data['assets'] - company_data['liabilities']
    
    return results

def get_similar_businesses(industry, listings_collection, limit=5):
    """Find similar businesses from historical data."""
    try:
        similar_businesses = list(listings_collection.find({
            "business_basics.industry_category": {
                "$elemMatch": {"$regex": f"^{re.escape(industry)}$", "$options": "i"}
            }
        }).limit(limit))
        return similar_businesses
    except Exception as e:
        print(f"Error finding similar businesses: {e}")
        return []

def gemini_qna(query, context=None):
    """Enhanced question answering using Gemini."""
    try:
        # Create the system prompt
        system_prompt = """
       Expert Business Investor Interview System
System Role Definition
You are an expert business analyst and investor interviewer, combining the analytical precision of Kevin O'Leary, the technical insight of Mark Cuban, and the strategic vision of other top investors from "Shark Tank" and "Dragon's Den" while maintaining a professional, neutral tone. Your purpose is to conduct in-depth interviews with business owners to comprehensively evaluate their companies for potential investment or acquisition.
Interview Context & Objectives
You have access to a database of approximately 1021 unique questions from investor shows like Shark Tank and Dragon's Den. Your goal is to leverage these questions strategically while adapting them to each specific business. The interview should gather all information necessary to:
    1. Build a complete business profile 
    2. Assess viability and growth potential 
    3. Identify strengths, weaknesses, and opportunities 
    4. Determine appropriate valuation methods and ranges 
    5. Generate an investor-ready business summary 
Adaptive Interview Methodology
Phase 1: Initial Discovery (3-5 questions)
Begin with general questions to identify fundamental business parameters:
- "Tell me about your business and what problem you're solving."
- "How long have you been operating and what's your current stage?"
- "What industry are you in and who are your target customers?"
- "What's your revenue model and current traction?"

Phase 2: Business Model Deep Dive (5-7 questions)
Tailor questions based on the business model identified in Phase 1:
For Digital/SaaS businesses: Focus on metrics like MRR/ARR, churn rate, CAC, LTV, and scalability

- "What's your monthly recurring revenue and growth rate?"
- "What's your customer acquisition cost compared to lifetime value?"
- "What's your churn rate and retention strategy?"

For Physical Product businesses: Focus on production, supply chain, margins, and distribution
- "What are your production costs and gross margins?"
- "How do you manage your supply chain and inventory?"
- "What are your distribution channels and retail strategy?"

For Service businesses: Focus on scalability, capacity utilization, pricing models

- "How do you scale your service delivery beyond your personal time?"
- "What's your hourly/project rate structure and utilization rate?"
- "How do you maintain quality as you expand your team?"

Phase 3: Market & Competition Analysis (4-6 questions)
Adapt questions based on market maturity and competitive landscape:

- "What's your total addressable market size and how did you calculate it?"
- "Who are your top 3 competitors and how do you differentiate?"
- "What barriers to entry exist in your market?"
- "What market trends are impacting your growth potential?"

Phase 4: Financial Performance (5-8 questions)
Tailor financial questions based on business stage:
For Pre-revenue/Early stage:

- "What's your burn rate and runway?"
- "What are your financial projections for the next 24 months?"
- "What assumptions underlie your revenue forecasts?"

For Revenue-generating businesses:
- "What has your year-over-year revenue growth been?"
- "Break down your cost structure between fixed and variable costs."
- "What's your path to profitability and timeline?"
- "What are your gross and net margins?"

For Profitable businesses:
- "What's your EBITDA and how has it evolved over time?"
- "What's your cash conversion cycle?"
- "How do you reinvest profits back into the business?"

Phase 5: Team & Operations (3-5 questions)
- "Tell me about your founding team and key executives."
- "What critical roles are you looking to fill next?"
- "How is equity distributed among founders and employees?"
- "What operational challenges are limiting your growth?"

Phase 6: Investment & Growth Strategy (4-6 questions)
- "How much capital are you raising and at what valuation?"
- "How will you allocate the investment funds?"
- "What specific milestones will this funding help you achieve?"
- "What's your long-term exit strategy?"
Dynamic Adaptation Requirements
Pattern Recognition Flags
Throughout the interview, identify patterns that require deeper investigation:
Red Flags - Require immediate follow-up:
    • Inconsistent financial numbers 
    • Unrealistic market size claims 
    • Vague answers about competition 
    • Excessive founder salaries relative to revenue 
    • Unreasonable valuation expectations 
Opportunity Signals - Areas to explore further:
    • Unusually high margins for the industry 
    • Proprietary technology or IP 
    • Evidence of product-market fit 
    • Strong team with relevant experience 
    • Clear customer acquisition strategy with proven ROI 
Jump Logic Instructions
    • If a response reveals a critical issue or opportunity, immediately pivot to explore that area more deeply before returning to your sequence 
    • If you detect inconsistency between answers, flag it and seek clarification 
    • If the business has unusual characteristics that don't fit standard models, adapt your questioning approach accordingly 
Response Analysis
Continuously evaluate:
    • Answer quality and thoroughness 
    • Internal consistency across topics 
    • Information gaps requiring additional questions 
    • Unique business aspects that warrant customized questions 
Strategic Database Utilization
When selecting or formulating questions:
    1. Start with general questions from your database that match the current business context 
    2. Adapt database questions to the specific business type, size, and stage 
    3. Create logical follow-up questions based on previous answers 
    4. When encountering unique business aspects, formulate new questions inspired by patterns in your database 
Communication Guidelines
Interview Flow
    • Maintain a conversational but purposeful tone 
    • Ask one question at a time to ensure clarity 
    • Begin with open-ended questions before narrowing focus 
    • Acknowledge and build upon previous answers to show active listening 
    • Use transitional phrases when changing topics: "Now I'd like to understand more about..." 
Question Formulation
    • Be direct and specific in your questions 
    • Avoid leading questions that suggest preferred answers 
    • Use neutral language that doesn't assume success or failure 
    • When needed, request quantifiable metrics rather than generalities 
    • Frame follow-up questions that refer to previous answers: "You mentioned X earlier. How does that relate to...?" 
Business Valuation Framework
Apply appropriate valuation methods based on business type and stage:
    1. For Pre-Revenue Companies: 
        ◦ Team and IP assessment 
        ◦ Market opportunity sizing 
        ◦ Comparable early-stage funding rounds 
    2. For Early-Stage Revenue Companies: 
        ◦ Revenue multiples based on growth rate 
        ◦ Customer acquisition economics assessment 
        ◦ Comparable transaction analysis 
    3. For Established Companies: 
        ◦ P/E ratios 
        ◦ EV/EBITDA multiples 
        ◦ Discounted Cash Flow analysis 
        ◦ Book value and asset-based valuations 
Analysis & Deliverables
After completing the interview, prepare:
    1. Business Profile Summary including: 
        ◦ Company overview and value proposition 
        ◦ Market opportunity assessment 
        ◦ Competitive positioning 
        ◦ Team evaluation 
        ◦ Business model analysis 
    2. Financial Analysis including: 
        ◦ Revenue and profitability metrics 
        ◦ Growth trajectory 
        ◦ Unit economics 
        ◦ Capital efficiency 
    3. Valuation Assessment including: 
        ◦ Methodologies applied 
        ◦ Comparable company/transaction benchmarks 
        ◦ Recommended valuation range 
        ◦ Key value drivers and detractors 
    4. Investment Considerations including: 
        ◦ Key strengths and differentiators 
        ◦ Risk factors and mitigation strategies 
        ◦ Growth opportunities 
        ◦ Strategic recommendations
        """
        
        # Combine prompts
        full_prompt = f"{system_prompt}\n\n"
        if context:
            full_prompt += f"Context: {context}\n\n"
        full_prompt += f"Query: {query}"
        
        # Generate response
        response = gemini_model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                candidate_count=1,
                max_output_tokens=7048,
            ),
            safety_settings={
                'HARASSMENT':'block_none',
                'HATE_SPEECH':'block_none',
                'SEXUAL':'block_none',
                'DANGEROUS':'block_none'
            }
        )
        
        # Extract and return the response text
        if hasattr(response, 'text'):
            return response.text
        elif response.parts:
            return "".join(part.text for part in response.parts)
        else:
            return "Error: Could not generate response"
            
    except Exception as e:
        print(f"Error in gemini_qna: {e}")
        return f"An error occurred: {str(e)}"

def generate_next_question(conversation_history, industry=None):
    """Generate the next question based on conversation history."""
    try:
        # First, try to get a relevant question from the database
        if industry:
            db_question = questions_collection.find_one({
                "category": "Core Business Analysis Questions"
            })
            if db_question:
                return db_question.get("question")
        
        # If no database question found or no industry specified,
        # generate a question using Gemini
        conversation_context = "\n\n".join([
            f"Q: {exchange['question']}\nA: {exchange['answer']}"
            for exchange in conversation_history
        ])
        
        prompt = f"""
        Based on this business assessment conversation, generate the most relevant next question.
        Focus on gathering critical business information we don't yet have.
        
        Conversation History:
        {conversation_context}
        
        Generate a single, specific question that will help us better understand the business.
        The question should be clear, concise, and focused on one aspect of the business.
        """
        
        response = gemini_qna(prompt)
        
        # Clean up the response
        question = response.strip().strip('"').strip()
        if question.lower().startswith("next question:"):
            question = question[13:].strip()
        
        return question
    
    except Exception as e:
        print(f"Error generating next question: {e}")
        return "What are your current revenue and growth projections?"



# Main Application Logic
def main():
    # --- Sidebar Navigation ---
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
        st.markdown(f"<div style='text-align: center; padding: 1rem; font-size: 0.8rem; color: #64748B;'>{datetime.now().strftime('%B %d, %Y')}</div>", unsafe_allow_html=True)

    # Initialize session state variables
    if 'valuation_data' not in st.session_state:
        st.session_state.valuation_data = {}
    if 'assessment_responses' not in st.session_state:
        st.session_state.assessment_responses = {}
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'current_question_idx' not in st.session_state:
        st.session_state.current_question_idx = 0
    if 'assessment_completed' not in st.session_state:
        st.session_state.assessment_completed = False

    # --- Company Valuation Page ---
    if "Company Valuation" in page:
        render_valuation_page()

    # --- Business Assessment Page ---
    elif "Business Assessment" in page:
        render_assessment_page()

def render_valuation_page():
    st.markdown("# 💰 Company Valuation Estimator")
    st.markdown("Estimate your company's value using multiple valuation methods and industry comparisons.")

    # Valuation input form
    with st.form("valuation_form"):
        st.markdown("### Company Information")
        col1, col2 = st.columns(2)
        
        with col1:
            company_name = st.text_input("Company Name", key="company_name")
            industry_options = [
                "Software/SaaS", "E-commerce", "Technology", "Healthcare",
                "Food & Beverage", "Manufacturing", "Retail", "Services",
                "Real Estate", "Other"
            ]
            industry = st.selectbox("Industry", industry_options, key="industry")
            revenue = st.number_input("Annual Revenue (USD)", min_value=0.0, format="%f", key="revenue")
            earnings = st.number_input("Annual Earnings/Net Income (USD)", format="%f", key="earnings")
            
        with col2:
            assets = st.number_input("Total Assets (USD)", min_value=0.0, format="%f", key="assets")
            liabilities = st.number_input("Total Liabilities (USD)", min_value=0.0, format="%f", key="liabilities")
            growth_options = ["High", "Moderate", "Low"]
            growth = st.select_slider("Growth Rate", options=growth_options, value="Moderate", key="growth")

        st.markdown("### Cash Flow Projections")
        st.markdown("Enter projected cash flows for the next 5 years (USD)")
        cf_cols = st.columns(5)
        cash_flows = []
        for i, col in enumerate(cf_cols):
            with col:
                cf = st.number_input(f"Year {i+1}", min_value=0.0, format="%f", key=f"cf_{i}")
                cash_flows.append(cf)

        submit_button = st.form_submit_button("Calculate Valuation")

        if submit_button:
            if not company_name:
                st.error("Please enter a company name.")
                return

            # Prepare company data
            company_data = {
                "name": company_name,
                "industry": industry,
                "revenue": revenue,
                "earnings": earnings,
                "assets": assets,
                "liabilities": liabilities,
                "growth": growth,
                "cash_flows": cash_flows
            }

            # Calculate valuation
            valuation_results = calculate_valuation(company_data)
            
            # Store results in session state
            st.session_state.valuation_data = {
                "company_data": company_data,
                "valuation_results": valuation_results
            }

            # Find similar businesses
            similar_businesses = get_similar_businesses(industry, listings_collection)
            st.session_state.similar_businesses = similar_businesses

    # Display valuation results if available
    if 'valuation_data' in st.session_state and st.session_state.valuation_data:
        display_valuation_results(st.session_state.valuation_data, st.session_state.get('similar_businesses', []))

def display_valuation_results(valuation_data, similar_businesses):
    st.markdown("---")
    st.markdown("## Valuation Results")

    company_data = valuation_data["company_data"]
    valuation_results = valuation_data["valuation_results"]

    # Display valuation methods in cards
    st.markdown("### Valuation Methods")
    cols = st.columns(len(valuation_results))
    
    for col, (method, value) in zip(cols, valuation_results.items()):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">${value:,.2f}</div>
                <div class="metric-label">{method.replace('_', ' ').title()}</div>
            </div>
            """, unsafe_allow_html=True)

    # Calculate and display average valuation
    valid_valuations = [v for v in valuation_results.values() if v > 0]
    if valid_valuations:
        avg_valuation = sum(valid_valuations) / len(valid_valuations)
        st.markdown(f"""
        <div style='text-align: center; margin: 2rem 0;'>
            <h3>Estimated Average Valuation</h3>
            <div style='font-size: 2rem; color: #1E3A8A; font-weight: bold;'>
                ${avg_valuation:,.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Display similar businesses
    if similar_businesses:
        st.markdown("### Similar Businesses")
        for biz in similar_businesses:
            biz_basics = biz.get("business_basics", {})
            pitch_metrics = biz.get("pitch_metrics", {})
            
            st.markdown(f"""
            <div class="similar-biz-card">
                <h4>{biz_basics.get('business_name', 'N/A')}</h4>
                <p><strong>Industry:</strong> {', '.join(biz_basics.get('industry_category', ['N/A']))}</p>
                <p><strong>Valuation:</strong> ${safe_int(pitch_metrics.get('implied_valuation', 0)):,}</p>
                <p><strong>Ask Amount:</strong> ${safe_int(pitch_metrics.get('initial_ask_amount', 0)):,} for {pitch_metrics.get('equity_offered', 'N/A')}% equity</p>
            </div>
            """, unsafe_allow_html=True)

    # Generate insights using Gemini
    insights_prompt = f"""
    Analyze this business valuation data and provide strategic insights:
    
    Company: {company_data['name']}
    Industry: {company_data['industry']}
    Revenue: ${company_data['revenue']:,.2f}
    Growth Rate: {company_data['growth']}
    
    Valuation Results:
    {json.dumps(valuation_results, indent=2)}
    
    Provide:
    1. Key valuation drivers
    2. Industry-specific considerations
    3. Growth opportunities
    4. Risk factors
    5. Strategic recommendations
    """
    
    with st.spinner("Generating insights..."):
        insights = gemini_qna(insights_prompt)
        
        st.markdown("### Strategic Insights")
        st.markdown(insights)


def render_assessment_page():
    st.markdown("# 📊 Business Assessment")
    st.markdown("Get personalized insights through an adaptive business evaluation.")

    max_questions = 15  # Maximum number of questions in the assessment

    # Display progress
    progress = min(1.0, st.session_state.current_question_idx / max_questions)
    st.progress(progress)

    # Assessment Logic
    if not st.session_state.assessment_completed and st.session_state.current_question_idx < max_questions:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        # Generate or get current question
        if st.session_state.current_question_idx == 0:
            current_question = "Tell me about your business and what problem you're solving."
        else:
            current_question = generate_next_question(st.session_state.conversation_history)

        st.markdown(f"### Question {st.session_state.current_question_idx + 1} of {max_questions}")
        st.markdown(f"**{current_question}**")

        # Get user's answer
        answer = st.text_area("Your Answer", height=100, key=f"answer_{st.session_state.current_question_idx}")

        if st.button("Submit Answer", use_container_width=True):
            if answer.strip():  # Ensure answer is not empty
                # Store the Q&A in conversation history
                st.session_state.conversation_history.append({
                    "question": current_question,
                    "answer": answer
                })
                st.session_state.assessment_responses[current_question] = answer
                st.session_state.current_question_idx += 1

                if st.session_state.current_question_idx >= max_questions:
                    st.session_state.assessment_completed = True
                
                st.rerun()
            else:
                st.warning("Please provide an answer before proceeding.")

        st.markdown("</div>", unsafe_allow_html=True)

    # Display Assessment Results
    elif st.session_state.assessment_completed:
        display_assessment_results()

def display_assessment_results():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Business Assessment Results")

    # Prepare conversation history for analysis
    conversation_text = "\n\n".join([
        f"Q: {q}\nA: {a}" for q, a in st.session_state.assessment_responses.items()
    ])

    # Generate comprehensive analysis using Gemini
    analysis_prompt = f"""
    Analyze this business assessment interview and provide a comprehensive evaluation:

    Interview Transcript:
    {conversation_text}

    Provide a detailed analysis covering:
    1. Business Profile Summary
    2. SWOT Analysis
    3. Financial Health Assessment
    4. Market Position
    5. Key Strengths & Concerns
    6. Strategic Recommendations
    7. Investment Potential Rating (High/Medium/Low)

    Format the response with clear headings and bullet points.
    Focus on actionable insights and specific recommendations.
    """

    with st.spinner("Generating comprehensive business assessment..."):
        analysis = gemini_qna(analysis_prompt)
        st.markdown(analysis)

    # Add option to start new assessment
    if st.button("Start New Assessment", use_container_width=True):
        # Reset assessment state
        st.session_state.conversation_history = []
        st.session_state.current_question_idx = 0
        st.session_state.assessment_completed = False
        st.session_state.assessment_responses = {}
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

def display_error_message(error_text):
    """Display error message in a consistent format."""
    st.error(f"Error: {error_text}")

def validate_input(value, value_type="string", min_value=None, max_value=None):
    """Validate input values."""
    try:
        if value_type == "string":
            if not isinstance(value, str):
                return False, "Invalid string input"
            if min_value and len(value) < min_value:
                return False, f"Input must be at least {min_value} characters"
            if max_value and len(value) > max_value:
                return False, f"Input must be less than {max_value} characters"
        elif value_type in ["float", "int"]:
            value = float(value) if value_type == "float" else int(value)
            if min_value is not None and value < min_value:
                return False, f"Value must be greater than {min_value}"
            if max_value is not None and value > max_value:
                return False, f"Value must be less than {max_value}"
        return True, None
    except ValueError:
        return False, f"Invalid {value_type} input"

def format_currency(amount):
    """Format currency values consistently."""
    try:
        return f"${float(amount):,.2f}"
    except (ValueError, TypeError):
        return "$0.00"

# --- Footer ---
def render_footer():
    st.markdown("""
    <hr style='margin: 2rem 0;'>
    <div style='padding: 1rem; text-align: center; font-size: 0.8rem; color: #64748B;'>
        Business Insights Hub © 2024 | Powered by Google Gemini
    </div>
    """, unsafe_allow_html=True)

# Main execution
if __name__ == "__main__":
    try:
        main()
        render_footer()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


