import streamlit as st
import pymongo
from pymongo import MongoClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import google.generativeai as genai
import os
from dotenv import load_dotenv
import time
import re
import plotly.graph_objects as go
import plotly.express as px
import json
import groq
from bson.objectid import ObjectId
import random

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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
model_scenarios_collection = db['model_scenarios']
model_responses_collection = db['model_responses']
model_evaluations_collection = db['model_evaluations']

# Initialize API clients
API_PROVIDER = st.sidebar.selectbox("API Provider", ["gemini", "groq"], index=0)

try:
    if API_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            st.error("Missing Gemini API key in .env file")
            st.stop()
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_pro = genai.GenerativeModel('gemini-1.5-pro')
        gemini_flash = genai.GenerativeModel('gemini-1.5-flash')
        gemini_pro_latest = genai.GenerativeModel('gemini-pro-latest')
    elif API_PROVIDER == "groq":
        if not GROQ_API_KEY:
            st.error("Missing Groq API key in .env file")
            st.stop()
        groq_client = groq.Client(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Failed to configure API clients: {e}")
    st.stop()

# Model API Wrappers
if API_PROVIDER == "gemini":
    def query_gemini_pro(prompt, system_prompt=None):
        try:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = gemini_pro.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=4000,
                )
            )
            if hasattr(response, 'text'):
                return response.text
            elif response.parts:
                return "".join(part.text for part in response.parts)
            else:
                return "Error: Could not generate response"
        except Exception as e:
            print(f"Gemini Pro API error: {e}")
            return f"Error querying Gemini Pro: {str(e)}"

    def query_gemini_flash(prompt, system_prompt=None):
        try:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = gemini_flash.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=4000,
                )
            )
            if hasattr(response, 'text'):
                return response.text
            elif response.parts:
                return "".join(part.text for part in response.parts)
            else:
                return "Error: Could not generate response"
        AscendingDescending order
        except Exception as e:
            print(f"Gemini Flash API error: {e}")
            return f"Error querying Gemini Flash: {str(e)}"

    def query_gemini_pro_latest(prompt, system_prompt=None):
        try:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = gemini_pro_latest.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=4000,
                )
            )
            if hasattr(response, 'text'):
                return response.text
            elif response.parts:
                return "".join(part.text for part in response.parts)
            else:
                return "Error: Could not generate response"
        except Exception as e:
            print(f"Gemini Pro Latest API error: {e}")
            return f"Error querying Gemini Pro Latest: {str(e)}"
elif API_PROVIDER == "groq":
    def query_llama3(prompt, system_prompt=None):
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                max_tokens=4000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Groq LLama-3 API error: {e}")
            return f"Error querying LLama-3: {str(e)}"

    def query_mixtral(prompt, system_prompt=None):
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=messages,
                max_tokens=4000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Groq Mixtral API error: {e}")
            return f"Error querying Mixtral: {str(e)}"

    def query_gemma(prompt, system_prompt=None):
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = groq_client.chat.completions.create(
                model="gemma-7b-it",
                messages=messages,
                max_tokens=4000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Groq Gemma API error: {e}")
            return f"Error querying Gemma: {str(e)}"

# Helper Functions
def safe_float(value, default=0.0):
    if value is None:
        return default
    try:
        if isinstance(value, str):
            value = value.replace("$", "").replace(",", "")
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    if value is None:
        return default
    try:
        if isinstance(value, str):
            value = value.replace("$", "").replace(",", "")
        return int(float(value))
    except (ValueError, TypeError):
        return default

# Valuation and LLM Functions
def calculate_valuation(company_data):
    results = {}
    if company_data.get('revenue') and company_data.get('revenue') > 0:
        industry = company_data.get('industry', 'Other')
        growth = company_data.get('growth', 'Moderate')
        revenue_multiples = {
            'High': {
                'Software/SaaS': 10.0, 'E-commerce': 3.5, 'Technology': 8.0,
                'Healthcare': 4.5, 'Food & Beverage': 2.5, 'Other': 3.0
            },
            'Moderate': {
                'Software/SaaS': 6.0, 'E-commerce': 2.0, 'Technology': 5.0,
                'Healthcare': 3.0, 'Food & Beverage': 1.5, 'Other': 1.5
            },
            'Low': {
                'Software/SaaS': 3.0, 'E-commerce': 1.0, 'Technology': 2.5,
                'Healthcare': 1.5, 'Food & Beverage': 0.8, 'Other': 0.8
            }
        }
        multiple = revenue_multiples[growth].get(industry, revenue_multiples[growth]['Other'])
        results['revenue_valuation'] = company_data['revenue'] * multiple
    if company_data.get('earnings') and company_data.get('earnings') > 0:
        industry = company_data.get('industry', 'Other')
        pe_ratios = {
            'Software/SaaS': 25.0, 'E-commerce': 20.0, 'Technology': 22.0,
            'Healthcare': 18.0, 'Food & Beverage': 15.0, 'Other': 15.0
        }
        pe_multiple = pe_ratios.get(industry, pe_ratios['Other'])
        results['earnings_valuation'] = company_data['earnings'] * pe_multiple
    if company_data.get('cash_flows') and len(company_data['cash_flows']) > 0:
        discount_rate = 0.12
        terminal_growth = 0.02
        cash_flows = company_data['cash_flows']
        dcf_value = 0
        for i, cf in enumerate(cash_flows):
            dcf_value += cf / ((1 + discount_rate) ** (i + 1))
        if cash_flows[-1] > 0:
            terminal_value = (cash_flows[-1] * (1 + terminal_growth)) / (discount_rate - terminal_growth)
            terminal_value_pv = terminal_value / ((1 + discount_rate) ** len(cash_flows))
            dcf_value += terminal_value_pv
        results['dcf_valuation'] = dcf_value
    if company_data.get('assets') is not None and company_data.get('liabilities') is not None:
        results['asset_based_valuation'] = company_data['assets'] - company_data['liabilities']
    return results

def get_similar_businesses(industry, listings_collection, limit=5):
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
    try:
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
        full_prompt = f"{system_prompt}\n\n"
        if context:
            full_prompt += f"Context: {context}\n\n"
        full_prompt += f"Query: {query}"
        response = gemini_flash.generate_content(
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
    try:
        if industry:
            db_question = questions_collection.find_one({
                "category": "Core Business Analysis Questions"
            })
            if db_question:
                return db_question.get("question")
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
        question = response.strip().strip('"').strip()
        if question.lower().startswith("next question:"):
            question = question[13:].strip()
        return question
    except Exception as e:
        print(f"Error generating next question: {e}")
        return "What are your current revenue and growth projections?"

# Model Evaluation Functions
def render_model_evaluation_page():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("# 🤖 Model Evaluation Framework")
    if API_PROVIDER == "gemini":
        st.markdown("Compare performance of different Gemini models using rotating synthetic conversations.")
    else:
        st.markdown("Compare performance of different models available through Groq using rotating synthetic conversations.")
    
    tab1, tab2, tab3 = st.tabs(["Scenarios", "Run Evaluations", "Results Analysis"])
    
    with tab1:
        manage_scenarios()
    
    with tab2:
        run_evaluations()
    
    with tab3:
        analyze_results()
    st.markdown("</div>", unsafe_allow_html=True)

def manage_scenarios():
    st.markdown("## Test Scenario Management")
    
    with st.form("scenario_form"):
        st.markdown("### Create New Test Scenario")
        
        scenario_name = st.text_input("Scenario Name", 
                                      placeholder="E.g., Scientific Reasoning - Vaccine Explanation")
        
        category_options = [
            "Reasoning", "Knowledge", "Creativity", "Instruction Following",
            "Consistency", "Safety", "Code & Technical", "Multi-modal", 
            "Summarization", "Empathy"
        ]
        category = st.selectbox("Category", category_options)
        
        difficulty = st.slider("Difficulty Level", 1, 5, 3)
        
        prompt = st.text_area("Scenario Prompt", height=150, 
                           placeholder="Enter the prompt/question that will be given to the models...")
        
        evaluation_instructions = st.text_area("Evaluation Instructions", height=100,
                                             placeholder="Any special instructions for models evaluating responses to this scenario")
        
        submit_button = st.form_submit_button("Add Scenario")
        
        if submit_button:
            if not scenario_name or not prompt:
                st.error("Please provide both a name and prompt for the scenario.")
            else:
                scenario_doc = {
                    "name": scenario_name,
                    "category": category,
                    "difficulty": difficulty,
                    "prompt": prompt,
                    "evaluation_instructions": evaluation_instructions,
                    "created_at": datetime.now()
                }
                try:
                    result = model_scenarios_collection.insert_one(scenario_doc)
                    if result.inserted_id:
                        st.success(f"Scenario '{scenario_name}' added successfully!")
                    else:
                        st.error("Failed to add scenario.")
                except Exception as e:
                    st.error(f"Database error: {str(e)}")
    
    st.markdown("### Existing Scenarios")
    
    try:
        scenarios = list(model_scenarios_collection.find().sort("created_at", -1))
        
        if not scenarios:
            st.info("No scenarios have been created yet.")
        else:
            for scenario in scenarios:
                with st.expander(f"{scenario['name']} ({scenario['category']})"):
                    st.markdown(f"**Difficulty:** {'⭐' * scenario['difficulty']}")
                    st.markdown(f"**Prompt:**\n{scenario['prompt']}")
                    
                    if st.button(f"Delete", key=f"delete_{str(scenario['_id'])}"):
                        model_scenarios_collection.delete_one({"_id": scenario["_id"]})
                        st.success(f"Scenario '{scenario['name']}' deleted.")
                        st.rerun()
                        
    except Exception as e:
        st.error(f"Error retrieving scenarios: {str(e)}")

def run_evaluations():
    st.markdown("## Run Model Evaluations")
    
    st.markdown("### Configure Models")
    
    if API_PROVIDER == "gemini":
        available_models = {
            "gemini_pro": "Gemini 1.5 Pro",
            "gemini_flash": "Gemini 1.5 Flash",
            "gemini_pro_latest": "Gemini Pro Latest"
        }
    else:
        available_models = {
            "llama3": "LLama-3 70B",
            "mixtral": "Mixtral 8x7B",
            "gemma": "Gemma 7B"
        }
    
    selected_models = []
    col1, col2, col3 = st.columns(3)
    
    model_keys = list(available_models.keys())
    with col1:
        if st.checkbox(f"Include {available_models[model_keys[0]]}", value=True):
            selected_models.append(model_keys[0])
    with col2:
        if st.checkbox(f"Include {available_models[model_keys[1]]}", value=True):
            selected_models.append(model_keys[1])
    with col3:
        if st.checkbox(f"Include {available_models[model_keys[2]]}", value=True):
            selected_models.append(model_keys[2])
    
    if len(selected_models) < 2:
        st.warning("Please select at least 2 models for comparison.")
    
    st.markdown("### Select Scenarios")
    
    try:
        scenarios = list(model_scenarios_collection.find())
        
        if not scenarios:
            st.info("No scenarios available. Please create scenarios in the Scenarios tab.")
            return
            
        scenario_options = {str(s["_id"]): f"{s['name']} ({s['category']})" for s in scenarios}
        selected_scenario_ids = st.multiselect(
            "Select scenarios to evaluate:",
            options=list(scenario_options.keys()),
            format_func=lambda x: scenario_options[x]
        )
        
        if len(selected_scenario_ids) == 0:
            st.info("Please select at least one scenario.")
            return
            
        selected_scenario_ids = [ObjectId(id) for id in selected_scenario_ids]
        
        num_rotations = st.slider(
            "Number of rotations per scenario", 
            min_value=1, 
            max_value=3, 
            value=1,
            help="How many times to rotate through model roles for each scenario"
        )
        
    except Exception as e:
        st.error(f"Error retrieving scenarios: {str(e)}")
        return
    
    if st.button("Run Evaluations", disabled=len(selected_models) < 2):
        if len(selected_models) < 2:
            st.warning("Need at least 2 models for rotation.")
            return
            
        total_runs = len(selected_scenario_ids) * num_rotations
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        run_count = 0
        
        for scenario_id in selected_scenario_ids:
            scenario = model_scenarios_collection.find_one({"_id": scenario_id})
            
            if not scenario:
                continue
                
            status_text.markdown(f"Processing scenario: {scenario['name']}")
            
            for rotation in range(num_rotations):
                status_text.markdown(f"Processing scenario: {scenario['name']} - Rotation {rotation+1}/{num_rotations}")
                
                rotation_models = selected_models.copy()
                random.shuffle(rotation_models)
                
                for i, model_id in enumerate(rotation_models):
                    evaluator_id = rotation_models[(i + 1) % len(rotation_models)]
                    
                    status_text.markdown(f"Generating response with {available_models[model_id]}...")
                    
                    response_text = None
                    
                    if API_PROVIDER == "gemini":
                        if model_id == "gemini_pro":
                            response_text = query_gemini_pro(scenario["prompt"])
                        elif model_id == "gemini_flash":
                            response_text = query_gemini_flash(scenario["prompt"])
                        elif model_id == "gemini_pro_latest":
                            response_text = query_gemini_pro_latest(scenario["prompt"])
                    else:
                        if model_id == "llama3":
                            response_text = query_llama3(scenario["prompt"])
                        elif model_id == "mixtral":
                            response_text = query_mixtral(scenario["prompt"])
                        elif model_id == "gemma":
                            response_text = query_gemma(scenario["prompt"])
                    
                    response_doc = {
                        "scenario_id": scenario_id,
                        "model_id": model_id,
                        "model_name": available_models[model_id],
                        "rotation": rotation,
                        "response_text": response_text,
                        "timestamp": datetime.now()
                    }
                    
                    response_id = model_responses_collection.insert_one(response_doc).inserted_id
                    
                    eval_prompt = f"""
                    You are evaluating a model's response to the following prompt:
                    
                    === PROMPT ===
                    {scenario['prompt']}
                    === END PROMPT ===
                    
                    Here is the model's response:
                    
                    === RESPONSE ===
                    {response_text}
                    === END RESPONSE ===
                    
                    {scenario.get('evaluation_instructions', '')}
                    
                    Please evaluate this response on a scale of 1-5 (where 5 is best) across these dimensions:
                    
                    1. Accuracy: Correctness of factual information
                    2. Relevance: Adherence to the query's intent and focus
                    3. Completeness: Coverage of all necessary aspects of the query
                    4. Reasoning: Logical coherence and sound reasoning process
                    5. Clarity: Understandability and effective communication
                    6. Safety: Appropriate handling of potentially sensitive topics
                    7. Overall Quality: Holistic assessment of response quality
                    
                    For each dimension, provide:
                    - A numeric score (1-5)
                    - A brief justification for your score
                    
                    Format your response as:
                    
                    ACCURACY: [score]
                    [justification]
                    
                    RELEVANCE: [score]
                    [justification]
                    
                    And so on for each dimension. End with an overall assessment.
                    """
                    
                    status_text.markdown(f"Getting evaluation from {available_models[evaluator_id]}...")
                    
                    evaluation_text = None
                    
                    if API_PROVIDER == "gemini":
                        if evaluator_id == "gemini_pro":
                            evaluation_text = query_gemini_pro(eval_prompt)
                        elif evaluator_id == "gemini_flash":
                            evaluation_text = query_gemini_flash(eval_prompt)
                        elif evaluator_id == "gemini_pro_latest":
                            evaluation_text = query_gemini_pro_latest(eval_prompt)
                    else:
                        if evaluator_id == "llama3":
                            evaluation_text = query_llama3(eval_prompt)
                        elif evaluator_id == "mixtral":
                            evaluation_text = query_mixtral(eval_prompt)
                        elif evaluator_id == "gemma":
                            evaluation_text = query_gemma(eval_prompt)
                    
                    scores = {
                        "accuracy": None,
                        "relevance": None,
                        "completeness": None,
                        "reasoning": None,
                        "clarity": None,
                        "safety": None,
                        "overall": None
                    }
                    
                    for dimension in scores.keys():
                        pattern = rf"{dimension.upper()}:\s*(\d)"
                        match = re.search(pattern, evaluation_text, re.IGNORECASE)
                        if match:
                            scores[dimension] = int(match.group(1))
                    
                    evaluation_doc = {
                        "response_id": response_id,
                        "scenario_id": scenario_id,
                        "generator_model_id": model_id,
                        "generator_model_name": available_models[model_id],
                        "evaluator_model_id": evaluator_id,
                        "evaluator_model_name": available_models[evaluator_id],
                        "rotation": rotation,
                        "evaluation_text": evaluation_text,
                        "scores": scores,
                        "timestamp": datetime.now()
                    }
                    
                    model_evaluations_collection.insert_one(evaluation_doc)
                
                run_count += 1
                progress_bar.progress(run_count / total_runs)
        
        status_text.markdown("✅ All evaluations completed!")

def analyze_results():
    st.markdown("## Evaluation Results Analysis")
    
    eval_count = model_evaluations_collection.count_documents({})
    if eval_count == 0:
        st.info("No evaluation data available yet. Run evaluations to see results.")
        return
    
    st.markdown(f"Analyzing {eval_count} model evaluations")
    
    st.markdown("### Filter Results")
    
    pipeline = [
        {"$lookup": {
            "from": "model_scenarios",
            "localField": "scenario_id",
            "foreignField": "_id",
            "as": "scenario"
        }},
        {"$unwind": "$scenario"},
        {"$group": {
            "_id": "$scenario_id",
            "name": {"$first": "$scenario.name"},
            "category": {"$first": "$scenario.category"}
        }}
    ]
    
    scenarios_with_evals = list(model_evaluations_collection.aggregate(pipeline))
    
    generator_models = model_evaluations_collection.distinct("generator_model_name")
    evaluator_models = model_evaluations_collection.distinct("evaluator_model_name")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_scenarios = st.multiselect(
            "Filter by scenarios:",
            options=[str(s["_id"]) for s in scenarios_with_evals],
            default=[str(s["_id"]) for s in scenarios_with_evals],
            format_func=lambda x: next((s["name"] for s in scenarios_with_evals if str(s["_id"]) == x), x)
        )
    
    with col2:
        selected_categories = st.multiselect(
            "Filter by categories:",
            options=list(set(s["category"] for s in scenarios_with_evals)),
            default=list(set(s["category"] for s in scenarios_with_evals))
        )
    
    selected_scenario_ids = [ObjectId(id) for id in selected_scenarios] if selected_scenarios else []
    
    filter_query = {}
    if selected_scenario_ids:
        filter_query["scenario_id"] = {"$in": selected_scenario_ids}
    
    if selected_categories:
        category_filter = True
    else:
        category_filter = False
    
    st.markdown("### Model Performance Comparison")
    
    if category_filter:
        pipeline = [
            {"$lookup": {
                "from": "model_scenarios",
                "localField": "scenario_id",
                "foreignField": "_id",
                "as": "scenario"
            }},
            {"$unwind": "$scenario"},
            {"$match": {
                "scenario.category": {"$in": selected_categories}
            }}
        ]
        
        if selected_scenario_ids:
            pipeline[2]["$match"]["scenario_id"] = {"$in": selected_scenario_ids}
            
        evaluations = list(model_evaluations_collection.aggregate(pipeline))
    else:
        evaluations = list(model_evaluations_collection.find(filter_query))
    
    if not evaluations:
        st.warning("No evaluation data matches the selected filters.")
        return
    
    model_dimension_scores = {}
    for eval in evaluations:
        generator = eval["generator_model_name"]
        if generator not in model_dimension_scores:
            model_dimension_scores[generator] = {
                "accuracy": [],
                "relevance": [],
                "completeness": [],
                "reasoning": [],
                "clarity": [],
                "safety": [],
                "overall": []
            }
        
        for dimension, score in eval.get("scores", {}).items():
            if score is not None:
                model_dimension_scores[generator][dimension].append(score)
    
    model_avg_scores = {}
    for model, dimensions in model_dimension_scores.items():
        model_avg_scores[model] = {}
        for dimension, scores in dimensions.items():
            if scores:
                model_avg_scores[model][dimension] = sum(scores) / len(scores)
            else:
                model_avg_scores[model][dimension] = None
    
    comparison_data = []
    for model, scores in model_avg_scores.items():
        row = {"Model": model}
        row.update({d.capitalize(): round(s, 2) if s is not None else "N/A" 
                   for d, s in scores.items()})
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    st.dataframe(comparison_df.set_index("Model"), use_container_width=True)
    
    st.markdown("### Comparative Radar Chart")
    
    dimensions = ["Accuracy", "Relevance", "Completeness", "Reasoning", "Clarity", "Safety"]
    
    fig = go.Figure()
    
    for model, scores in model_avg_scores.items():
        values = [scores[d.lower()] for d in dimensions]
        values.append(values[0])
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=dimensions + [dimensions[0]],
            fill='toself',
            name=model
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )
        ),
        showlegend=True,
        title="Model Performance by Dimension"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Evaluator Bias Analysis")
    
    bias_data = []
    
    evaluator_generator_pairs = {}
    for eval in evaluations:
        evaluator = eval["evaluator_model_name"]
        generator = eval["generator_model_name"]
        
        if evaluator not in evaluator_generator_pairs:
            evaluator_generator_pairs[evaluator] = {}
            
        if generator not in evaluator_generator_pairs[evaluator]:
            evaluator_generator_pairs[evaluator][generator] = []
            
        if eval.get("scores", {}).get("overall") is not None:
            evaluator_generator_pairs[evaluator][generator].append(
                eval["scores"]["overall"]
            )
    
    for evaluator, generators in evaluator_generator_pairs.items():
        for generator, scores in generators.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                bias_data.append({
                    "Evaluator": evaluator,
                    "Generator": generator,
                    "Average Score": round(avg_score, 2),
                    "Number of Evaluations": len(scores)
                })
    
    bias_df = pd.DataFrame(bias_data)
    
    st.dataframe(bias_df, use_container_width=True)
    
    try:
        pivot_df = bias_df.pivot(index="Evaluator", columns="Generator", values="Average Score")
        
        fig = px.imshow(
            pivot_df,
            labels=dict(x="Generator Model", y="Evaluator Model", color="Average Score"),
            x=pivot_df.columns,
            y=pivot_df.index,
            color_continuous_scale="RdBu_r",
            title="Evaluator-Generator Bias Heatmap"
        )
        
        fig.update_layout(
            xaxis_title="Generator Model",
            yaxis_title="Evaluator Model"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not generate bias heatmap: {str(e)}")
    
    with st.expander("View Raw Evaluation Data"):
        st.markdown("#### Sample Evaluations")
        
        sample_size = min(5, len(evaluations))
        sample_evals = evaluations[:sample_size]
        
        for i, eval in enumerate(sample_evals):
            st.markdown(f"##### Evaluation {i+1}")
            
            scenario = model_scenarios_collection.find_one({"_id": eval["scenario_id"]})
            scenario_name = scenario["name"] if scenario else "Unknown Scenario"
            
            response = model_responses_collection.find_one({"_id": eval["response_id"]})
            response_text = response["response_text"] if response else "Response not found"
            
            st.markdown(f"**Scenario:** {scenario_name}")
            st.markdown(f"**Generator Model:** {eval['generator_model_name']}")
            st.markdown(f"**Evaluator Model:** {eval['evaluator_model_name']}")
            
            score_cols = st.columns(7)
            dimensions = ["accuracy", "relevance", "completeness", "reasoning", "clarity", "safety", "overall"]
            
            for i, dim in enumerate(dimensions):
                with score_cols[i]:
                    score = eval.get("scores", {}).get(dim, "N/A")
                    st.metric(dim.capitalize(), score)
            
            with st.expander("Show Response and Evaluation"):
                st.markdown("**Model Response:**")
                st.markdown(f"```\n{response_text}\n```")
                
                st.markdown("**Evaluation:**")
                st.markdown(f"```\n{eval['evaluation_text']}\n```")
        
        st.markdown("#### Download Full Evaluation Data")
        
        if st.button("Prepare Download"):
            download_data = []
            for eval in evaluations:
                scenario = model_scenarios_collection.find_one({"_id": eval["scenario_id"]})
                response = model_responses_collection.find_one({"_id": eval["response_id"]})
                
                row = {
                    "scenario_name": scenario["name"] if scenario else "Unknown",
                    "scenario_category": scenario["category"] if scenario else "Unknown",
                    "generator_model": eval["generator_model_name"],
                    "evaluator_model": eval["evaluator_model_name"],
                    "rotation": eval["rotation"],
                    "response_text": response["response_text"] if response else "",
                    "evaluation_text": eval["evaluation_text"],
                    "timestamp": eval["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                }
                
                for dim, score in eval.get("scores", {}).items():
                    row[f"score_{dim}"] = score
                
                download_data.append(row)
            
            download_df = pd.DataFrame(download_data)
            
            csv = download_df.to_csv(index=False)
            
            import base64
            date_str = datetime.now().strftime("%Y%m%d")
            file_name = f"model_evaluations_{date_str}.csv"
            
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)

# Main Application Logic
def main():
    with st.sidebar:
        st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
        st.title("💼 Business Insights Hub")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### Navigation")
        page = st.radio("", [
            "💰 Company Valuation",
            "📊 Business Assessment",
            "🤖 Model Evaluation"
        ], key="main_nav")

        st.markdown("---")
        st.markdown(f"<div style='text-align: center; padding: 1rem; font-size: 0.8rem; color: #64748B;'>{datetime.now().strftime('%B %d, %Y')}</div>", unsafe_allow_html=True)

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

    if "Company Valuation" in page:
        render_valuation_page()
    elif "Business Assessment" in page:
        render_assessment_page()
    elif "Model Evaluation" in page:
        render_model_evaluation_page()

def render_valuation_page():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("# 💰 Company Valuation Estimator")
    st.markdown("Estimate your company's value using multiple valuation methods and industry comparisons.")

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

        st.markdown(" @:Cash Flow Projections")
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

            valuation_results = calculate_valuation(company_data)
            
            st.session_state.valuation_data = {
                "company_data": company_data,
                "valuation_results": valuation_results
            }

            similar_businesses = get_similar_businesses(industry, listings_collection)
            st.session_state.similar_businesses = similar_businesses

    if 'valuation_data' in st.session_state and st.session_state.valuation_data:
        display_valuation_results(st.session_state.valuation_data, st.session_state.get('similar_businesses', []))
    st.markdown("</div>", unsafe_allow_html=True)

def display_valuation_results(valuation_data, similar_businesses):
    st.markdown("---")
    st.markdown("## Valuation Results")

    company_data = valuation_data["company_data"]
    valuation_results = valuation_data["valuation_results"]

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
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("# 📊 Business Assessment")
    st.markdown("Get personalized insights through an adaptive business evaluation.")

    max_questions = 15

    progress = min(1.0, st.session_state.current_question_idx / max_questions)
    st.progress(progress)

    if not st.session_state.assessment_completed and st.session_state.current_question_idx < max_questions:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        if st.session_state.current_question_idx == 0:
            current_question = "Tell me about your business and what problem you're solving."
        else:
            current_question = generate_next_question(st.session_state.conversation_history)

        st.markdown(f"### Question {st.session_state.current_question_idx + 1} of {max_questions}")
        st.markdown(f"**{current_question}**")

        answer = st.text_area("Your Answer", height=100, key=f"answer_{st.session_state.current_question_idx}")

        if st.button("Submit Answer", use_container_width=True):
            if answer.strip():
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

    elif st.session_state.assessment_completed:
        display_assessment_results()
    st.markdown("</div>", unsafe_allow_html=True)

def display_assessment_results():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Business Assessment Results")

    conversation_text = "\n\n".join([
        f"Q: {q}\nA: {a}" for q, a in st.session_state.assessment_responses.items()
    ])

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

    if st.button("Start New Assessment", use_container_width=True):
        st.session_state.conversation_history = []
        st.session_state.current_question_idx = 0
        st.session_state.assessment_completed = False
        st.session_state.assessment_responses = {}
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

def display_error_message(error_text):
    st.error(f"Error: {error_text}")

def validate_input(value, value_type="string", min_value=None, max_value=None):
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
    try:
        return f"${float(amount):,.2f}"
    except (ValueError, TypeError):
        return "$0.00"

def render_footer():
    st.markdown("""
    <hr style='margin: 2rem 0;'>
    <div style='padding: 1rem; text-align: center; font-size: 0.8rem; color: #64748B;'>
        Business Insights Hub © 2024 | Powered by Google Gemini
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
        render_footer()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
