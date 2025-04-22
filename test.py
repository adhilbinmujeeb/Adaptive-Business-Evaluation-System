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
import traceback

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
            print("MongoDB connection successful")  # Debug statement
            return client
        except Exception as e:
            print(f"MongoDB connection attempt {attempt + 1} failed: {e}")  # Debug statement
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
API_PROVIDER = st.sidebar.selectbox(
    "Select API Provider",
    ["gemini", "groq"],
    index=0,
    label_visibility="visible"
)

try:
    print(f"Configuring API provider: {API_PROVIDER}")  # Debug statement
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
    print(f"API configuration error: {e}")  # Debug statement
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
        similar_businesses = list(list
