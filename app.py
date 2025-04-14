import streamlit as st
import pymongo
from pymongo import MongoClient
import pandas as pd
import numpy as np
from datetime import datetime
# Removed google.generativeai import
# import google.generativeai as genai
from groq import Groq, APIError, AuthenticationError # Import Groq SDK and specific errors
import os
from dotenv import load_dotenv
import time
import re

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Remove Gemini key
GROQ_API_KEY = os.getenv("GROQ_API_KEY") # Add Groq key

# Set page configuration
st.set_page_config(
    page_title="Business Insights Hub",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (Keep existing CSS)
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
    .sidebar-header {
        padding: 0.5rem 1rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #E2E8F0;
    }
    .similar-biz-card {
        border: 1px solid #CBD5E1;
        border-radius: 0.375rem; /* rounded-md */
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
            print("MongoDB connection successful.")
            return client
        except pymongo.errors.ConnectionFailure as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(2)
            else:
                st.error("Failed to connect to MongoDB after retries. Please check your connection details and ensure the server is running.")
                st.stop()
        except Exception as e:
             st.error(f"An error occurred during MongoDB connection: {e}")
             st.stop()

client = get_mongo_client()
db = client['business_rag']
question_collection = db['questions']
listings_collection = db['business_listings']

# --- Groq API Setup ---
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop()

try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    # Optional: Test connection briefly if needed, but client initialization itself is a good check
    # print("Groq client initialized successfully.")
except AuthenticationError:
    st.error("Groq API authentication failed. Please check your GROQ_API_KEY.")
    st.stop()
except Exception as e:
    st.error(f"Failed to initialize Groq client: {e}")
    st.stop()

# Helper Functions (Keep existing safe_float, safe_int)
def safe_float(value, default=0.0):
    """Safely converts a value to float, handling $, ,, None, and errors."""
    if value is None:
        return default
    try:
        str_value = str(value).replace("$", "").replace(",", "")
        return float(str_value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    """Safely converts a value to int, handling potential float strings, $, ,, None."""
    if value is None:
        return default
    try:
        str_value = str(value).replace("$", "").replace(",", "")
        return int(float(str_value))
    except (ValueError, TypeError):
        return default


# --- Groq QnA Function ---
def groq_qna(query, context=None, model="meta-llama/llama-4-scout-17b-16e-instruct"): # Default to a fast model
    """
    Sends a query to the Groq API with optional context and a system prompt.

    Args:
        query (str): The user's query.
        context (str, optional): Additional context for the query. Defaults to None.
        model (str): The Groq model to use (e.g., "meta-llama/llama-4-scout-17b-16e-instruct", "mixtral-8x7b-32768").

    Returns:
        str: The AI's response text, or an error message.
    """
    # Keep the detailed system prompt
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
[... Keep the detailed methodology phases (Initial Discovery to Growth Strategy) ...]

Dynamic Adaptation Requirements
[... Keep Pattern Recognition, Jump Logic, Response Analysis ...]

Strategic Database Utilization
[... Keep Database Utilization guidance ...]

Communication Guidelines
[... Keep Interview Flow and Question Formulation guidance ...]

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

    messages = [
        {"role": "system", "content": system_prompt}
    ]
    user_content = f"Query: {query}"
    if context:
        user_content = f"Context: {context}\n\n{user_content}"
    messages.append({"role": "user", "content": user_content})

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model=model,
            # temperature=0.7, # Optional: Adjust creativity
            # max_tokens=4096, # Optional: Limit output length, ensure it's within model limits
        )
        response_text = chat_completion.choices[0].message.content
        return response_text

    except AuthenticationError:
        st.error("Groq API authentication failed. Please check your GROQ_API_KEY.")
        return "Error: Groq Authentication Failed."
    except APIError as e:
        st.error(f"Groq API error: {e}")
        return f"Error: Groq API Error ({e.status_code})"
    except Exception as e:
        st.error(f"An unexpected error occurred while calling Groq API: {e}")
        import traceback
        traceback.print_exc()
        return "An unexpected error occurred fetching the AI response."


# --- Sidebar Navigation (Keep As Is) ---
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
    st.info("Using Groq API (Llama 3 8B)") # Indicate which API is active
    st.markdown(f"<div style='text-align: center; padding: 1rem; font-size: 0.8rem; color: #64748B;'>{datetime.now().strftime('%B %d, %Y')}</div>", unsafe_allow_html=True)

# Session State Initialization (Keep As Is)
if 'valuation_data' not in st.session_state:
    st.session_state.valuation_data = {}
if 'assessment_responses' not in st.session_state:
    st.session_state.assessment_responses = {}
if 'current_question_idx' not in st.session_state:
    st.session_state.current_question_idx = 0
if 'valuation_step' not in st.session_state:
    st.session_state.valuation_step = 0

# ==============================================================================
# 💰 Company Valuation Estimator (UPDATED to use groq_qna)
# ==============================================================================
if "Company Valuation" in page:
    st.markdown("# 💰 Company Valuation Estimator")
    st.markdown("Estimate your company's value and see how it compares to similar pitches.")

    # Valuation Questions (Keep As Is)
    valuation_questions = [
        "What is your company's name?", # Step 0
        "What is your company's annual revenue (in USD)?", # Step 1
        "What are your company's annual earnings (net income, in USD)?", # Step 2
        "What is your company's EBITDA (Earnings Before Interest, Taxes, Depreciation, and Amortization, in USD)?", # Step 3
        "What industry does your company operate in?", # Step 4
        "What is your company's total assets value (in USD)?", # Step 5
        "What is your company's total liabilities (in USD)?", # Step 6
        "What are your projected cash flows for the next 5 years (comma-separated, in USD)?", # Step 7
        "What is your company's growth rate (e.g., High, Moderate, Low)?" # Step 8
    ]

    total_steps = len(valuation_questions)
    current_step = st.session_state.valuation_step

    st.progress(min(1.0, current_step / total_steps))
    st.markdown(f"##### Step {current_step + 1} of {total_steps}")

    if current_step < total_steps:
        # --- Input Section (Keep As Is) ---
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        current_question = valuation_questions[current_step]
        st.markdown(f"### {current_question}")

        help_texts = {
            0: "Enter the legal or operating name of your company.",
            1: "Enter your total annual revenue before expenses.",
            2: "Enter your annual profit after all expenses and taxes.",
            3: "EBITDA = Earnings Before Interest, Taxes, Depreciation, and Amortization.",
            4: "Select the industry that best describes your business. This helps find comparable pitches.",
            5: "Total value of all assets owned by your company.",
            6: "Total of all debts and obligations owed by your company.",
            7: "Estimate your net cash flows (inflows - outflows) for each of the next 5 years, separated by commas.",
            8: "Assess your company's expected revenue growth trend."
        }

        if current_step in help_texts:
            st.markdown(f"*{help_texts[current_step]}*")

        input_key = f"val_step_{current_step}"

        if current_step == 0:
            answer = st.text_input("Company Name", key=input_key, value=st.session_state.valuation_data.get(current_question, ""))
        elif current_step in [1, 2, 3, 5, 6]:
            default_val = safe_float(st.session_state.valuation_data.get(current_question, 0))
            answer = st.number_input("USD", min_value=0.0, step=1000.0, format="%.2f", key=input_key, value=default_val)
            answer = str(answer)
        elif current_step == 4:
             # Dynamically fetch industries from listings if possible, otherwise use static list
            try:
                distinct_industries = listings_collection.distinct("business_basics.industry_category")
                # Filter out potential None or empty strings and flatten if nested lists exist
                industries = sorted(list(set(
                    item for sublist in distinct_industries if sublist for item in (sublist if isinstance(sublist, list) else [sublist]) if item
                )))
                if not industries: # Fallback if no industries found or all are invalid
                    raise ValueError("No valid industries found in database")
            except Exception as e:
                print(f"Could not fetch distinct industries from MongoDB: {e}. Using static list.")
                industries = sorted([
                    "Software/SaaS", "E-commerce", "Manufacturing", "Retail", "Food & Beverage",
                    "Healthcare", "Financial Services", "Real Estate", "Hospitality",
                    "Technology (General)", "Consumer Goods", "Services (Business)", "Services (Consumer)",
                    "Fashion/Apparel", "Tools/DIY", "Automotive", "Energy", "Education", "Entertainment/Media",
                    "Fitness/Wellness", "Pets", "Children/Baby", "Other"
                ])

            default_industry = st.session_state.valuation_data.get(current_question, industries[0] if industries else "Other")
            answer = st.selectbox("Select Industry", industries, key=input_key, index=industries.index(default_industry) if default_industry in industries else 0)

        elif current_step == 7:
            default_flows_str = st.session_state.valuation_data.get(current_question, "0,0,0,0,0")
            default_flows = [safe_float(cf) for cf in default_flows_str.split(",")]
            year_cols = st.columns(5)
            cash_flows_input = []
            for i, col in enumerate(year_cols):
                with col:
                    default_cf = default_flows[i] if i < len(default_flows) else 0.0
                    cf = col.number_input(f"Year {i+1}", min_value=None, step=1000.0, format="%.2f", key=f"cf_{i}_{current_step}", value=default_cf)
                    cash_flows_input.append(str(cf))
            answer = ",".join(cash_flows_input)
        elif current_step == 8:
            growth_options = ["Low", "Moderate", "High"]
            default_growth = st.session_state.valuation_data.get(current_question, "Moderate")
            answer = st.select_slider("Select Growth Rate", options=growth_options, key=input_key, value=default_growth)

        col_back, col_next = st.columns([1, 5])
        with col_back:
            if current_step > 0:
                if st.button("⬅️ Back", key=f"back_{current_step}"):
                    st.session_state.valuation_step -= 1
                    st.rerun()
        with col_next:
            is_valid = True
            if current_step == 0 and not (answer and answer.strip()): # Check if answer exists and is not just whitespace
                 is_valid = False
                 st.warning("Company name cannot be empty.")

            # Ensure 'answer' exists before proceeding (relevant for text input)
            if is_valid and 'answer' in locals() and st.button("Next ➡️", use_container_width=True, key=f"next_{current_step}"):
                st.session_state.valuation_data[current_question] = answer
                st.session_state.valuation_step += 1
                st.rerun()
            elif is_valid and 'answer' not in locals() and st.button("Next ➡️", use_container_width=True, key=f"next_{current_step}"):
                 # This case might occur if number_input returns None somehow, though unlikely with default
                 st.warning("Please enter a valid value.")


        st.markdown("</div>", unsafe_allow_html=True)


    # --- Calculation and Comparison Phase ---
    if current_step >= total_steps:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Company Information Summary")

        # Extract data (Keep As Is)
        company_name = st.session_state.valuation_data.get(valuation_questions[0], "N/A")
        revenue = safe_float(st.session_state.valuation_data.get(valuation_questions[1], "0"))
        earnings = safe_float(st.session_state.valuation_data.get(valuation_questions[2], "0"))
        ebitda = safe_float(st.session_state.valuation_data.get(valuation_questions[3], "0"))
        user_industry = st.session_state.valuation_data.get(valuation_questions[4], "Other")
        assets = safe_float(st.session_state.valuation_data.get(valuation_questions[5], "0"))
        liabilities = safe_float(st.session_state.valuation_data.get(valuation_questions[6], "0"))
        cash_flows_str = st.session_state.valuation_data.get(valuation_questions[7], "0,0,0,0,0")
        cash_flows = [safe_float(cf) for cf in cash_flows_str.split(",")]
        growth = st.session_state.valuation_data.get(valuation_questions[8], "Low")

        # Display summary (Keep As Is)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Company Name:**")
            st.markdown(f"**Industry:**")
            st.markdown(f"**Annual Revenue:**")
            st.markdown(f"**Net Income:**")
            st.markdown(f"**EBITDA:**")
        with col2:
            st.markdown(f"{company_name}")
            st.markdown(f"{user_industry}")
            st.markdown(f"${revenue:,.2f}")
            st.markdown(f"${earnings:,.2f}")
            st.markdown(f"${ebitda:,.2f}")

        st.markdown("</div>", unsafe_allow_html=True) # Close summary card

        # --- Valuation Calculation (Using groq_qna) ---
        industry_avg_pe = 15.0
        industry_avg_ebitda_multiple = 8.0
        # Add logic here to potentially calculate these based on 'listings_collection' if desired

        with st.spinner(f"Calculating valuation for {company_name} using Groq..."):
            valuation_prompt = f"""
            Analyze the following company data and provide a valuation assessment using standard methods.

            Company Data:
              - Company Name: {company_name}
              - Annual Revenue: ${revenue:,.2f}
              - Annual Earnings (Net Income): ${earnings:,.2f}
              - EBITDA: ${ebitda:,.2f}
              - Industry: {user_industry}
              - Total Assets: ${assets:,.2f}
              - Total Liabilities: ${liabilities:,.2f}
              - Projected Cash Flows (5 years): {', '.join([f'${cf:,.2f}' for cf in cash_flows])}
              - Growth Rate Assessment: {growth}

            Industry Benchmarks (Use as reference, state assumptions if using):
              - Average P/E Ratio: ~{industry_avg_pe:.1f}
              - Average EV/EBITDA Multiple: ~{industry_avg_ebitda_multiple:.1f}
              - Typical Discount Rate (WACC) for DCF: ~10-15% (adjust based on risk/growth)

            Valuation Methods to Consider:
            1. Market-Based (if applicable):
               - Comparable Company Analysis (CCA): P/E Multiple (Value = Earnings × P/E), EV/EBITDA Multiple (EV = EBITDA × Multiple). Justify chosen multiples.
            2. Income-Based:
               - Discounted Cash Flow (DCF): Use projected cash flows. State assumed discount rate and terminal value assumptions (e.g., Gordon Growth Model or Exit Multiple). Formula: Σ [CF_t / (1 + r)^t] + [Terminal Value / (1 + r)^n].
            3. Asset-Based:
               - Book Value: Assets - Liabilities. Discuss its relevance (often a floor value).

            Output Requirements:
            - Calculate valuation using *at least two* relevant methods based on the data provided (e.g., P/E, EV/EBITDA if profitable; DCF if cash flows provided; Book Value). Clearly state which methods were used and the result for each.
            - Explain the rationale for choosing the methods and any key assumptions made (multiples, discount rate, growth rate for terminal value).
            - Provide a concluding recommended valuation range, synthesizing the results from the different methods.
            - Format with clear headings (e.g., "Valuation Methods Used", "Assumptions", "Calculations", "Recommended Range").
            """
            # *** Use groq_qna instead of gemini_qna ***
            valuation_result = groq_qna(valuation_prompt, context="Company Valuation Task")

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## Valuation Assessment")
        st.markdown(valuation_result)
        st.markdown("</div>", unsafe_allow_html=True) # Close valuation card

        # --- Find and Display Similar Historical Businesses (Keep As Is) ---
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## Comparable Pitches from Similar Industries")

        try:
            # Case-insensitive regex search within the industry_category array
            similar_businesses = list(listings_collection.find({
                "business_basics.industry_category": {
                    "$elemMatch": {"$regex": f"^{re.escape(user_industry)}$", "$options": "i"}
                }
            }).limit(5))

            if similar_businesses:
                st.markdown(f"Found {len(similar_businesses)} historical pitches in or related to the **'{user_industry}'** category:")
                for biz in similar_businesses:
                    biz_basics = biz.get("business_basics", {})
                    pitch_metrics = biz.get("pitch_metrics", {})
                    deal_outcome = biz.get("deal_outcome", {})

                    name = biz_basics.get("business_name", "N/A")
                    ask = safe_int(pitch_metrics.get("initial_ask_amount"))
                    equity = safe_int(pitch_metrics.get("equity_offered"))
                    valuation = safe_int(pitch_metrics.get("implied_valuation"))
                    result = deal_outcome.get("final_result", "N/A")
                    investors = pitch_metrics.get("participating_investors", [])
                    final_terms_raw = deal_outcome.get("final_terms")

                    final_terms_display = "N/A"
                    if isinstance(final_terms_raw, dict):
                         term_amount = safe_int(final_terms_raw.get('amount'))
                         term_equity = safe_int(final_terms_raw.get('equity'))
                         term_investors = ", ".join(final_terms_raw.get('investors', []))
                         if term_amount > 0 and term_equity > 0:
                             final_terms_display = f"${term_amount:,} for {term_equity}%"
                             if term_investors:
                                 final_terms_display += f" from {term_investors}"
                         else:
                             final_terms_display = str(final_terms_raw) # Fallback
                    elif isinstance(final_terms_raw, str) and final_terms_raw.lower() not in ['none', 'n/a', 'no deal', '']:
                        final_terms_display = final_terms_raw

                    st.markdown(f"<div class='similar-biz-card'>", unsafe_allow_html=True)
                    st.markdown(f"**{name}**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Pitch Ask:**")
                        st.markdown(f"**Implied Valuation:**")
                    with col2:
                         st.markdown(f"${ask:,} for {equity}%")
                         st.markdown(f"${valuation:,}")

                    st.markdown(f"**Outcome:** {str(result).replace('_', ' ').capitalize()}")
                    if result and str(result).lower() != 'no deal' and final_terms_display != 'N/A':
                         st.markdown(f"**Final Deal:** {final_terms_display}")
                    elif investors:
                         investor_names = [inv.get('name', 'Unknown') if isinstance(inv, dict) else str(inv) for inv in investors]
                         st.markdown(f"**Investors Involved:** {', '.join(filter(None, investor_names))}")


                    st.markdown("</div>", unsafe_allow_html=True) # Close similar-biz-card

            else:
                st.info(f"No comparable historical pitches found for the specific industry '{user_industry}' in the database.")

        except Exception as e:
            st.error(f"An error occurred while fetching similar businesses: {e}")
            import traceback
            traceback.print_exc()

        st.markdown("</div>", unsafe_allow_html=True) # Close comparison card

        # --- Reset Button (Keep As Is) ---
        if st.button("Start New Valuation", use_container_width=True, key="reset_valuation"):
            st.session_state.valuation_step = 0
            st.session_state.valuation_data = {}
            st.rerun()


# ==============================================================================
# 📊 Interactive Business Assessment (UPDATED to use groq_qna)
# ==============================================================================
elif "Business Assessment" in page:
    st.markdown("# 📊 Interactive Business Assessment")
    st.markdown("Get personalized insights through an adaptive business evaluation.")

    # Initialize session state variables (Keep As Is)
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'question_count' not in st.session_state:
        st.session_state.question_count = 0
    if 'assessment_completed' not in st.session_state:
        st.session_state.assessment_completed = False
    if 'assessment_responses' not in st.session_state:
        st.session_state.assessment_responses = {}
    if 'current_assessment_question' not in st.session_state:
        st.session_state.current_assessment_question = "Tell me about your business and what problem you're solving."

    max_questions = 15
    st.progress(min(1.0, st.session_state.question_count / max_questions))

    if not st.session_state.assessment_completed and st.session_state.question_count < max_questions:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.markdown(f"### Question {st.session_state.question_count + 1} of {max_questions}")
        st.markdown(f"**{st.session_state.current_assessment_question}**")

        response = st.text_area("Your Answer", height=100, key=f"assess_q_{st.session_state.question_count}")

        if st.button("Submit Answer", use_container_width=True, key=f"submit_assess_{st.session_state.question_count}"):
            if response.strip():
                st.session_state.assessment_responses[st.session_state.current_assessment_question] = response
                st.session_state.conversation_history.append({
                    "question": st.session_state.current_assessment_question,
                    "answer": response
                })
                st.session_state.question_count += 1

                if st.session_state.question_count >= max_questions:
                    st.session_state.assessment_completed = True
                    st.rerun()
                else:
                    # Generate next question (using groq_qna)
                    with st.spinner("Analyzing your response and preparing next question using Groq..."):
                        conversation_context = "\n\n".join([
                            f"Q: {exchange['question']}\nA: {exchange['answer']}"
                            for exchange in st.session_state.conversation_history
                        ])

                        next_question_prompt = f"""
                        Based on the following ongoing business assessment interview, ask the single most insightful follow-up question.
                        Prioritize questions that delve deeper into financials, market strategy, competition, or operational challenges based on the last response.
                        Avoid repeating similar questions. Aim for a logical progression in the interview.

                        Conversation History:
                        {conversation_context}

                        What is the next question? (Return only the question text)
                        """
                        # *** Use groq_qna instead of gemini_qna ***
                        next_question_raw = groq_qna(query=next_question_prompt, context="Continuing business assessment interview")
                        st.session_state.current_assessment_question = next_question_raw.strip().strip('"')

                    st.rerun()
            else:
                st.warning("Please provide an answer before submitting.")

        st.markdown("</div>", unsafe_allow_html=True)

    elif st.session_state.assessment_completed:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## Business Assessment Results")

        assessment_data = "\n\n".join([
            f"Q: {q}\nA: {a}"
            for q, a in st.session_state.assessment_responses.items() if a
        ])

        # Comprehensive analysis prompt (using groq_qna)
        analysis_prompt = f"""
        Perform a comprehensive business assessment based on the following interview transcript.
        Act as an expert investor panel (like Shark Tank/Dragon's Den).

        Interview Transcript:
        {assessment_data}

        Provide the following analysis, formatted clearly with headings and bullet points:
        1.  **Business Profile Summary:** (Overview, value proposition, market, competition, team, model)
        2.  **SWOT Analysis:** (Strengths, Weaknesses, Opportunities, Threats based *only* on the provided text)
        3.  **Financial Health Check:** (Comment on revenue, profit, costs, growth aspects mentioned)
        4.  **Key Strengths & Red Flags:** (Highlight major positives and concerns revealed)
        5.  **Strategic Recommendations:** (Suggest 2-3 actionable next steps for the business owner)
        6.  **Further Questions:** (List 2-3 critical questions still needed for a full evaluation)
        7.  **Overall Investment Potential:** (Brief qualitative assessment - e.g., High Potential, Needs Validation, Significant Concerns)
        """

        with st.spinner("Generating comprehensive business assessment report using Groq..."):
            # *** Use groq_qna instead of gemini_qna ***
            analysis_result = groq_qna(query=analysis_prompt, context="Final assessment report generation")

        st.markdown(analysis_result)

        if st.button("Start New Assessment", use_container_width=True, key="reset_assessment"):
            # Reset assessment state variables (Keep As Is)
            st.session_state.conversation_history = []
            st.session_state.question_count = 0
            st.session_state.assessment_completed = False
            st.session_state.assessment_responses = {}
            st.session_state.current_assessment_question = "Tell me about your business and what problem you're solving."
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# --- Footer (Updated to mention Groq) ---
st.markdown("""
<hr style='margin: 2rem 0;'>
<div style='padding: 1rem; text-align: center; font-size: 0.8rem; color: #64748B;'>
    Business Insights Hub © 2024 | Powered by Groq
</div>
""", unsafe_allow_html=True)
