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
from groq import Groq  # Import Groq SDK
import itertools  # For permutations
import random  # For random delays (though sleep is used directly)
# import scipy.stats as stats # Import commented out as not used yet, but available if needed

# --- Load Environment Variables ---
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Load Groq Key

# --- Page Config and CSS ---
st.set_page_config(
    page_title="Business Insights Hub",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Sidebar Styles */
    [data-testid="stSidebar"] {
        background-color: #f8fafc; /* Light gray background */
        border-right: 1px solid #e2e8f0;
    }
    .sidebar-header {
        padding: 1rem;
        background-color: #0ea5e9; /* Sky blue */
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        color: white; /* White text */
        text-align: center;
    }
    .sidebar-header h1 {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0;
    }
    [data-testid="stSidebar"] .stRadio > label {
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0.375rem;
        transition: background-color 0.2s ease-in-out;
        font-weight: 500; /* Medium font weight */
    }
    [data-testid="stSidebar"] .stRadio > label:hover {
        background-color: #e0f2fe; /* Lighter blue on hover */
        color: #0c4a6e; /* Darker blue text on hover */
    }
    /* Active radio button */
    [data-testid="stSidebar"] .stRadio [type="radio"]:checked + div {
        background-color: #e0f2fe;
        border-left: 4px solid #0ea5e9; /* Blue indicator line */
        color: #0c4a6e;
        font-weight: 600; /* Bold active selection */
        padding-left: calc(1rem - 4px); /* Adjust padding for border */
    }
    /* General Card Style */
    .card {
        background-color: #ffffff;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
    }
    /* Specific Card Style for Evaluation Page */
    .card-evaluation {
        background-color: #f0f9ff; /* Light blue background */
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #BEE3F8; /* Blue border */
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    /* Ensure DataFrames fit container */
    .stDataFrame {
        width: 100%;
    }
    /* Button Styling */
    .stButton>button {
        border-radius: 0.375rem;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: background-color 0.2s ease-in-out, border-color 0.2s ease-in-out;
        border: 1px solid #0ea5e9;
        background-color: #0ea5e9;
        color: white;
    }
    .stButton>button:hover {
        background-color: #0284c7; /* Darker blue on hover */
        border-color: #0284c7;
        color: white;
    }
    .stButton>button:focus {
        box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.3); /* Focus ring */
        outline: none;
    }
    .stButton>button:disabled {
        background-color: #cbd5e1; /* Gray when disabled */
        border-color: #cbd5e1;
        color: #64748b;
    }
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background-color: #0ea5e9; /* Blue progress bar */
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
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
        # Convert to float first to handle strings like "10.0"
        return int(float(value))
    except (ValueError, TypeError):
        return default

# --- MongoDB Connection (Placeholder Implementation) ---
@st.cache_resource(ttl=3600)
def get_mongo_client():
    """Attempts to connect to MongoDB."""
    try:
        # This assumes MONGO_URI is correctly set in your .env file or environment
        if not MONGO_URI:
            st.warning("MONGO_URI not found. MongoDB features will be disabled.", icon="⚠️")
            return None
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000) # 5 second timeout
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        print("MongoDB connection successful.")
        return client
    except pymongo.errors.ConnectionFailure as e:
        st.error(f"MongoDB Connection Failed: {e}. Check your MONGO_URI and network settings.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during MongoDB connection: {e}")
        return None

# Initialize MongoDB connection (will be None if connection fails)
# client = get_mongo_client()
# db = client['business_rag'] if client else None
# listings_collection = db['business_listings'] if db else None
# attributes_collection = db['business_attributes'] if db else None
# questions_collection = db['questions'] if db else None
# Note: The provided code doesn't actively use the DB, so this is placeholder.

# --- Initialize AI Clients ---
# Gemini
gemini_client = None
gemini_model_name = None
try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model_name = 'gemini-1.5-flash'  # Or choose another available model like 'gemini-pro'
        gemini_client = genai.GenerativeModel(gemini_model_name)
        print(f"Gemini configured with model: {gemini_model_name}")
    else:
        st.warning("GEMINI_API_KEY not found. Gemini features disabled.", icon="⚠️")
except Exception as e:
    st.error(f"Failed to configure Gemini API: {e}")
    gemini_client = None

# Groq
groq_client = None
try:
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("Groq client configured.")
    else:
        st.warning("GROQ_API_KEY not found. Groq models disabled.", icon="⚠️")
except Exception as e:
    st.error(f"Failed to configure Groq API: {e}")
    groq_client = None

# Combine available models
available_models = {}
if gemini_client:
    available_models['gemini'] = {'client': gemini_client, 'name': gemini_model_name}
if groq_client:
    # List of commonly available / known stable Groq models
    groq_model_names = ["gemma2-9b-it", "llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "llama-3.1-8b-instant", "llama-3.1-70b-versatile"]
    for name in groq_model_names:
        # Simple key generation (e.g., groq_gemma2, groq_llama3)
        key_name = f"groq_{name.split('-')[0]}"
        # Handle potential duplicate keys (e.g., llama3-8b vs llama3-70b)
        if key_name in available_models:
             key_name += f"_{name.split('-')[1]}" # e.g., groq_llama3_8b, groq_llama3_70b
        # Handle potential duplicate keys (e.g. llama-3.1-8b vs llama-3.1-70b)
        if key_name in available_models:
            key_name += f"_{name.split('-')[2]}" # e.g., groq_llama_3.1_8b

        available_models[key_name] = {'client': groq_client, 'name': name}


if not available_models:
    st.error("🚨 No AI models could be initialized! Please check your API keys (.env file or environment variables) and configurations. The application cannot function without at least one AI model.", icon="🛑")
    st.stop()

# --- Business Scenarios ---
BUSINESS_SCENARIOS = [
    {
        "name": "Early-Stage SaaS",
        "prompt": "You are interviewing the founder of 'CloudFlow', a 1-year-old B2B SaaS startup providing workflow automation tools for small marketing agencies. They have 15 paying customers, $5k MRR, growing 20% MoM. Seeking $250k seed funding."
    },
    {
        "name": "Established Restaurant",
        "prompt": "You are interviewing the owner of 'The Corner Bistro', a popular neighborhood restaurant operating for 8 years. Stable revenue around $800k/year, 15% net margin. Owner wants to open a second location and needs $150k."
    },
    {
        "name": "Pre-Revenue Hardware",
        "prompt": "You are interviewing the inventor of 'AquaPure Home', a smart home water purification device. They have a working prototype and patents pending. No sales yet. Planning a Kickstarter campaign. Need $50k for tooling and initial production run."
    },
    {
        "name": "E-commerce DTC Brand",
        "prompt": "You are interviewing the founder of 'Zenith Watches', a 3-year-old direct-to-consumer watch brand. $1.2M annual revenue, 40% gross margin, 10% net margin. Facing increased ad costs. Seeking $300k for inventory and marketing expansion."
    },
    {
        "name": "Local Service Business",
        "prompt": "You are interviewing the owner of 'GreenThumbs Landscaping', a 5-year-old local landscaping service. $300k annual revenue, mostly seasonal. Owner manages 3 crews. Wants $75k for new equipment to handle more clients."
    },
    {
        "name": "Mobile Gaming App",
        "prompt": "You are interviewing the developers of 'Pixel Quest', a free-to-play mobile RPG launched 6 months ago. 50k downloads, 5k DAU, $1k/month revenue from in-app purchases. Needs $100k for user acquisition and feature development."
    },
    {
        "name": "Biotech Research",
        "prompt": "You are interviewing the lead scientist of 'NeuroGen Labs', a pre-clinical biotech company developing a novel drug for Alzheimer's. Strong pre-clinical data. Needs $2M Series A for Phase 1 trials."
    },
    {
        "name": "Subscription Box",
        "prompt": "You are interviewing the founder of 'CraftBox Monthly', a 2-year-old subscription box for craft supplies. 1,500 subscribers at $30/month. 50% gross margin, facing churn issues (10%/month). Seeking $120k for product sourcing and retention marketing."
    },
    {
        "name": "Consulting Firm",
        "prompt": "You are interviewing the partners of 'StratAlign Consulting', a 10-year-old boutique management consulting firm. $5M annual revenue, 5 partners, 20 consultants. Stable but wants to develop a proprietary software platform. Seeking $750k."
    },
    {
        "name": "Non-Profit Organization",
        "prompt": "You are interviewing the director of 'CodeFuture Kids', a 4-year-old non-profit teaching coding to underserved youth. Serves 500 kids annually, $200k budget primarily from grants. Seeks $50k bridge funding to cover operational costs while securing larger grants."
    }
]

# --- LLM API Call Wrappers ---
def gemini_qna(query, context=None, model_client=None):
    """Question answering using Gemini."""
    if model_client is None:
        st.error("Gemini client not provided to gemini_qna")
        return "Error: Gemini client missing."
    try:
        system_prompt = """
        You are an expert business analyst and investor interviewer.
        Your goal is to conduct an in-depth interview to evaluate a business OR act as the entrepreneur being interviewed, providing plausible answers based on the given scenario.
        - If asked to ask a question: Ask relevant, probing questions using the context. Follow the Business Assessment Framework if provided.
        - If asked to provide an answer: Act as the entrepreneur. Provide a plausible, concise answer based ONLY on the provided scenario and conversation history. Do not invent completely new major facts not hinted at in the scenario.
        - If asked to evaluate: Be objective and follow instructions precisely.
        Maintain a professional tone. Keep responses concise unless detail is requested.
        """

        full_prompt = f"{system_prompt}\n\n"
        if context:
            full_prompt += f"Context/History:\n{context}\n\n"
        full_prompt += f"Request:\n{query}"

        response = model_client.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7, # Balance creativity and consistency
                candidate_count=1,
                max_output_tokens=1024,
            ),
            safety_settings={ # Relax safety settings if appropriate for business context, be mindful of implications
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
            }
        )

        # Handle potential lack of 'text' attribute and check for parts
        if hasattr(response, 'text') and response.text:
             return response.text
        elif response.parts:
             return "".join(part.text for part in response.parts)
        else:
             # Check for blocking reason
             try:
                 block_reason = response.prompt_feedback.block_reason
                 block_message = response.prompt_feedback.block_reason_message if hasattr(response.prompt_feedback, 'block_reason_message') else 'No specific message.'
                 print(f"Gemini Warning: Response blocked. Reason: {block_reason}. Message: {block_message}")
                 return f"Error: Response blocked by safety filter ({block_reason})."
             except Exception:
                 print(f"Gemini Error: Could not extract text or blocking reason. Response: {response}")
                 return "Error: Could not generate response (Unknown Gemini issue)."

    except Exception as e:
        print(f"Error in gemini_qna: {e}")
        st.error(f"Gemini API Error: {e}") # Show error in UI as well
        return f"An error occurred during Gemini generation: {str(e)}"

def groq_qna(query, context=None, model_client=None, model_name=None):
    """Question answering using Groq."""
    if model_client is None:
        st.error("Groq client not provided to groq_qna")
        return "Error: Groq client missing."
    if model_name is None:
        st.error("Groq model name not provided to groq_qna")
        return "Error: Groq model name missing."

    try:
        system_prompt = """
        You are an expert business analyst and investor interviewer.
        Your goal is to conduct an in-depth interview to evaluate a business OR act as the entrepreneur being interviewed, providing plausible answers based on the given scenario.
        - If asked to ask a question: Ask relevant, probing questions using the context. Follow the Business Assessment Framework if provided.
        - If asked to provide an answer: Act as the entrepreneur. Provide a plausible, concise answer based ONLY on the provided scenario and conversation history. Do not invent completely new major facts not hinted at in the scenario.
        - If asked to evaluate: Be objective and follow instructions precisely.
        Maintain a professional tone. Keep responses concise unless detail is requested.
        """

        messages = [{"role": "system", "content": system_prompt}]
        if context:
            # Combine context into a single system message for simplicity if needed, or keep separate
            messages.append({"role": "system", "content": f"Context/History:\n{context}"})
        messages.append({"role": "user", "content": f"Request:\n{query}"})

        chat_completion = model_client.chat.completions.create(
            messages=messages,
            model=model_name,
            temperature=0.7,
            max_tokens=1024,
            # top_p=1 # Default is usually fine
            # stop=None # No specific stop sequences needed typically
            # stream=False # Not streaming for this use case
        )

        response_content = chat_completion.choices[0].message.content
        return response_content

    except Exception as e:
        print(f"Error in groq_qna for model {model_name}: {e}")
        st.error(f"Groq API Error ({model_name}): {e}") # Show error in UI
        return f"An error occurred during Groq generation ({model_name}): {str(e)}"

# Generic function to call the right API
def ask_model(model_key, query, context=None):
    """Calls the appropriate Q&A function based on the model key."""
    if model_key not in available_models:
        st.error(f"Model key '{model_key}' not found in available models.")
        return f"Error: Unknown model key '{model_key}'"

    model_info = available_models[model_key]
    client = model_info['client']
    name = model_info['name']
    # Add a small random delay before calling API to help avoid strict rate limits
    # time.sleep(random.uniform(0.1, 0.5))

    start_time = time.time()
    if model_key.startswith('gemini'):
        response = gemini_qna(query, context=context, model_client=client)
    elif model_key.startswith('groq'):
        response = groq_qna(query, context=context, model_client=client, model_name=name)
    else:
        st.error(f"Handler not implemented for model key type: {model_key}")
        response = f"Error: No handler for {model_key}"
    end_time = time.time()
    print(f"Model '{model_key}' responded in {end_time - start_time:.2f} seconds.")

    return response


# --- Business Assessment Simulation and Evaluation ---
BUSINESS_TOPICS = [
    {"name": "financial", "keywords": ["revenue", "profit", "cash flow", "funding", "investment", "margins", "burn rate", "runway", "valuation", "cap table"]},
    {"name": "market", "keywords": ["competitors", "market size", "tam", "sam", "som", "customers", "demand", "industry", "trends", "segmentation", "go-to-market", "gtm"]},
    {"name": "team", "keywords": ["founder", "team", "experience", "skills", "leadership", "hiring", "advisors", "culture", "roles"]},
    {"name": "product", "keywords": ["product", "service", "features", "development", "technology", "roadmap", "ip", "patent", "prototype", "mvp", "differentiation", "value proposition"]},
    {"name": "operations", "keywords": ["operations", "supply chain", "manufacturing", "logistics", "scalability", "processes", "infrastructure"]},
    {"name": "growth", "keywords": ["growth", "expansion", "scaling", "strategy", "roadmap", "marketing", "sales", "acquisition", "cac", "ltv", "retention", "churn", "partnerships"]},
    {"name": "legal", "keywords": ["legal", "incorporation", "contracts", "compliance", "regulatory", "ip"]},
    {"name": "risks", "keywords": ["risk", "challenge", "threat", "weakness", "mitigation"]}
]

# --- Placeholder Quantitative Metrics ---
def calculate_topic_coverage(conversation, topics):
    """Calculate what percentage of key business topics are mentioned in questions."""
    if not conversation: return 0.0
    all_questions_text = " ".join([qa.get("question", "").lower() for qa in conversation if qa.get("question")])
    if not all_questions_text: return 0.0

    topics_covered = 0
    for topic in topics:
        if any(keyword in all_questions_text for keyword in topic["keywords"]):
            topics_covered += 1
    return (topics_covered / len(topics)) * 100 if topics else 0.0

def calculate_question_complexity(conversation):
    """Placeholder: Simple measure based on average question length."""
    if not conversation: return 0.0
    questions = [qa.get("question", "") for qa in conversation if qa.get("question")]
    if not questions: return 0.0
    avg_len = np.mean([len(q.split()) for q in questions])
    # Normalize roughly to 0-100 scale (e.g., assume avg length 5-25 is common)
    score = min(100, max(0, (avg_len - 5) * 5))
    return score

def calculate_lexical_diversity(conversation):
    """Placeholder: Simple measure of unique words used in questions."""
    if not conversation: return 0.0
    all_questions_text = " ".join([qa.get("question", "").lower() for qa in conversation if qa.get("question")])
    words = re.findall(r'\b\w+\b', all_questions_text)
    if not words: return 0.0
    ttr = len(set(words)) / len(words) if len(words) > 0 else 0
    # Normalize roughly to 0-100 scale
    score = min(100, max(0, ttr * 150)) # Adjust multiplier as needed
    return score

def calculate_conversation_coherence(conversation):
    """Placeholder: Needs more sophisticated NLP. Returns dummy value."""
    # Real implementation might involve embedding similarity between Q/A pairs or topic modeling.
    return random.uniform(30, 70) # Return a random plausible score

# --- Simulation Function ---
def simulate_assessment_conversation(model_key, scenario, num_turns=5):
    """Simulates a business assessment conversation."""
    print(f"Simulating conversation for Scenario: '{scenario['name']}', Model: {model_key}, Turns: {num_turns}")
    conversation = []
    history = f"Scenario Context:\n{scenario['prompt']}\n\n---\n\nConversation History:"
    last_response_type = "answer"  # Start by needing a question from the interviewer model

    # Business Assessment Framework Guidance for the Interviewer Model
    business_framework = """
    Business Assessment Framework Guide:
    1. Problem & Solution: Understand the core problem, the proposed solution, and the target customer.
    2. Market Opportunity: Assess market size (TAM/SAM/SOM), growth trends, and key competitors.
    3. Product/Service Details: Evaluate differentiation, technology, IP, development stage, and roadmap.
    4. Go-to-Market Strategy: Understand customer acquisition, sales channels, and marketing plans.
    5. Business Model & Financials: Analyze revenue streams, pricing, unit economics (CAC/LTV), key metrics (MRR/ARR if applicable), funding needs, and financial projections.
    6. Team: Assess founder/team experience, expertise, and ability to execute.
    7. Traction & Milestones: Review progress, key achievements, customer feedback, and future milestones.
    8. Risks & Challenges: Identify potential obstacles and mitigation plans.

    Instructions for Interviewer:
    - Ask specific, insightful questions relevant to the scenario and framework.
    - Build upon previous answers to probe deeper. Avoid generic questions.
    - Adapt your questions based on the entrepreneur's responses.
    - Aim for a logical flow through the assessment areas.
    """

    for turn in range(num_turns):
        print(f"  Turn {turn + 1}/{num_turns}...")
        current_question = ""
        current_answer = ""

        # 1. Generate Interviewer Question
        if last_response_type == "answer":
            q_prompt = f"""You are the expert investor interviewing the entrepreneur.
            {business_framework}

            Based on the Scenario Context and the Conversation History below, ask the *single most important and logical next question* to evaluate this business effectively. Ensure your question is specific and builds on the conversation if possible.

            {history}

            Next Interviewer Question:"""

            question_start_time = time.time()
            current_question = ask_model(model_key, q_prompt, context=None) # Context is already in the prompt
            print(f"    Q{turn+1} ({model_key}): '{current_question[:100]}...' ({(time.time() - question_start_time):.2f}s)")

            if current_question.startswith("Error:") or not current_question.strip():
                error_msg = current_question if current_question.startswith("Error:") else "Empty response"
                print(f"    WARNING: Model {model_key} failed to generate question for turn {turn + 1}. Error: {error_msg}")
                current_question = "[Model failed to generate question]"
                history += f"\nInterviewer Q{turn + 1}: {current_question}"
                # Decide how to handle failure: stop turn, stop simulation, or skip answer?
                # Let's skip the answer generation for this turn if the question failed.
                conversation.append({"question": current_question, "answer": "[Skipped due to question generation failure]"})
                last_response_type = "question_failed" # Mark failure
                # Continue to next turn - maybe the model recovers? Or break? Let's continue.
                time.sleep(1) # Small delay even on failure
                continue # Skip answer generation
            else:
                history += f"\nInterviewer Q{turn + 1}: {current_question}"
                last_response_type = "question"

        # 2. Generate Entrepreneur Answer (using the same model for consistency in simulation)
        if last_response_type == "question":
            a_prompt = f"""You are the entrepreneur being interviewed.
            Based on the Scenario Context and Conversation History (especially the last question asked by the interviewer), provide a plausible, concise, and consistent answer. Stick to the information implied by the scenario.

            {history}

            Your (Entrepreneur's) Answer to Q{turn + 1}:"""

            answer_start_time = time.time()
            current_answer = ask_model(model_key, a_prompt, context=None) # Context is in prompt
            print(f"    A{turn+1} ({model_key}): '{current_answer[:100]}...' ({(time.time() - answer_start_time):.2f}s)")

            if current_answer.startswith("Error:") or not current_answer.strip():
                error_msg = current_answer if current_answer.startswith("Error:") else "Empty response"
                print(f"    WARNING: Model {model_key} failed to generate answer for turn {turn + 1}. Error: {error_msg}")
                current_answer = "[Model failed to generate answer]"
                history += f"\nEntrepreneur A{turn + 1}: {current_answer}\n---" # Add separator even if answer failed
                last_response_type = "answer_failed"
            else:
                history += f"\nEntrepreneur A{turn + 1}: {current_answer}\n---" # Add separator after successful answer
                last_response_type = "answer"

            conversation.append({"question": current_question, "answer": current_answer})

        # Add a delay between turns to avoid hammering APIs
        time.sleep(1.5) # Sleep 1.5 seconds

    print(f"  Finished simulation for {model_key} on scenario '{scenario['name']}'.")
    return conversation

# --- Evaluation Functions ---
def evaluate_conversations(evaluator_model_key, scenario, conversation_a, model_a_key, conversation_b, model_b_key):
    """Uses an evaluator model to compare two conversations based on specified criteria."""
    print(f"Evaluating {model_a_key} vs {model_b_key} for scenario '{scenario['name']}' using evaluator {evaluator_model_key}")
    eval_results = {
        "evaluation_type": "pairwise_comparison",
        "evaluator_model": evaluator_model_key,
        "scenario": scenario['name'],
        "model_a": model_a_key,
        "model_b": model_b_key,
        "preference": "N/A",
        "reason_for_preference": "Evaluation incomplete.",
        "score_a_relevance": 0, "score_a_depth": 0, "score_a_consistency": 0, "score_a_coverage": 0,
        "score_b_relevance": 0, "score_b_depth": 0, "score_b_consistency": 0, "score_b_coverage": 0,
        "a_strengths": [], "a_weaknesses": [],
        "b_strengths": [], "b_weaknesses": [],
        "rationale": "Evaluation not performed or failed.",
        "raw_eval_output": ""
    }

    try:
        # Format conversations for the prompt
        conv_a_text = "\n".join([f" Q{i+1}: {qa.get('question', '[No Q]')}\n A{i+1}: {qa.get('answer', '[No A]')}" for i, qa in enumerate(conversation_a)])
        conv_b_text = "\n".join([f" Q{i+1}: {qa.get('question', '[No Q]')}\n A{i+1}: {qa.get('answer', '[No A]')}" for i, qa in enumerate(conversation_b)])

        # Improved Evaluation Prompt with Clear JSON Structure Request
        eval_prompt = f"""
        **Evaluation Task:** You are an expert Venture Capital analyst. Evaluate two simulated business assessment interviews based on the provided scenario. Your goal is to determine which interview simulation (questions and answers) was more effective and realistic in evaluating the business opportunity.

        **Business Scenario:**
        {scenario['prompt']}

        ---
        **Interview Simulation A (Model: {model_a_key}):**
        {conv_a_text}
        ---
        **Interview Simulation B (Model: {model_b_key}):**
        {conv_b_text}
        ---

        **Evaluation Criteria & Scoring (Score each from 1 to 5):**
        1.  **Question Relevance:** (1=Generic, 5=Highly Specific & Scenario-Aware) How relevant and tailored were the interviewer's questions to this specific business scenario?
        2.  **Inquiry Depth & Progression:** (1=Surface-level, 5=Systematic Deep Dive) Did the questions probe key areas effectively and build logically on previous answers?
        3.  **Answer Plausibility & Consistency:** (1=Contradictory/Unrealistic, 5=Highly Plausible & Consistent) How well did the entrepreneur's answers align with the scenario and maintain internal consistency?
        4.  **Business Topic Coverage:** (1=Major Gaps, 5=Comprehensive & Balanced) How well did the interview cover critical business aspects (Market, Product, Financials, Team, Traction, Risks)?

        **Instructions:**
        1.  First, provide a brief textual analysis comparing the two simulations point-by-point based on the criteria above. Highlight specific examples.
        2.  Then, provide your final structured evaluation **ONLY** in the following JSON format. Do not include any text before or after the JSON block.

        ```json
        {{
          "preference": "Model A" | "Model B" | "Neither",
          "reason_for_preference": "Concise justification for the preference based on the criteria.",
          "score_a_relevance": <score_1_to_5>,
          "score_a_depth": <score_1_to_5>,
          "score_a_consistency": <score_1_to_5>,
          "score_a_coverage": <score_1_to_5>,
          "score_b_relevance": <score_1_to_5>,
          "score_b_depth": <score_1_to_5>,
          "score_b_consistency": <score_1_to_5>,
          "score_b_coverage": <score_1_to_5>,
          "a_strengths": ["<Brief strength 1>", "<Brief strength 2>"],
          "a_weaknesses": ["<Brief weakness 1>", "<Brief weakness 2>"],
          "b_strengths": ["<Brief strength 1>", "<Brief strength 2>"],
          "b_weaknesses": ["<Brief weakness 1>", "<Brief weakness 2>"]
        }}
        ```
        """

        eval_start_time = time.time()
        evaluation_response = ask_model(evaluator_model_key, eval_prompt)
        print(f"    Evaluation response received from {evaluator_model_key} ({(time.time() - eval_start_time):.2f}s)")
        eval_results["raw_eval_output"] = evaluation_response

        if evaluation_response.startswith("Error:"):
            eval_results["rationale"] = f"Evaluator model failed: {evaluation_response}"
            return eval_results

        # Enhanced JSON Parsing
        json_data = None
        try:
            # Regex to find JSON block, accepting potential markdown backticks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```|(\{.*?\})', evaluation_response, re.DOTALL | re.IGNORECASE)
            if json_match:
                json_str = json_match.group(1) or json_match.group(2)
                # Attempt to clean potential artifacts if necessary (e.g., trailing commas)
                # json_str = json_str.strip() # Basic strip
                json_data = json.loads(json_str)
            else:
                # Fallback: try parsing the entire response if no block found (less reliable)
                try:
                    json_data = json.loads(evaluation_response)
                except json.JSONDecodeError:
                    eval_results["rationale"] = "Failed to find or parse JSON block in the evaluator's response."
                    print(f"    WARNING: Could not parse JSON from: {evaluation_response[:300]}...") # Log snippet

        except json.JSONDecodeError as json_e:
            eval_results["rationale"] = f"JSON Parsing Error: {json_e}. Response snippet: {evaluation_response[:300]}..."
            print(f"    ERROR: JSON Decode Error: {json_e}")
        except Exception as parse_e:
            eval_results["rationale"] = f"Error processing evaluation response structure: {parse_e}"
            print(f"    ERROR: Evaluation Processing Error: {parse_e}")

        # Populate results from parsed JSON if successful
        if json_data and isinstance(json_data, dict):
            eval_results["preference"] = json_data.get("preference", "Parse Error")
            eval_results["reason_for_preference"] = json_data.get("reason_for_preference", "N/A")
            # Use safe_int for score parsing
            eval_results["score_a_relevance"] = safe_int(json_data.get("score_a_relevance"), default=0)
            eval_results["score_a_depth"] = safe_int(json_data.get("score_a_depth"), default=0)
            eval_results["score_a_consistency"] = safe_int(json_data.get("score_a_consistency"), default=0)
            eval_results["score_a_coverage"] = safe_int(json_data.get("score_a_coverage"), default=0) # Added coverage
            eval_results["score_b_relevance"] = safe_int(json_data.get("score_b_relevance"), default=0)
            eval_results["score_b_depth"] = safe_int(json_data.get("score_b_depth"), default=0)
            eval_results["score_b_consistency"] = safe_int(json_data.get("score_b_consistency"), default=0)
            eval_results["score_b_coverage"] = safe_int(json_data.get("score_b_coverage"), default=0) # Added coverage
            # Get strengths/weaknesses (ensure they are lists)
            eval_results["a_strengths"] = json_data.get("a_strengths", []) if isinstance(json_data.get("a_strengths"), list) else []
            eval_results["a_weaknesses"] = json_data.get("a_weaknesses", []) if isinstance(json_data.get("a_weaknesses"), list) else []
            eval_results["b_strengths"] = json_data.get("b_strengths", []) if isinstance(json_data.get("b_strengths"), list) else []
            eval_results["b_weaknesses"] = json_data.get("b_weaknesses", []) if isinstance(json_data.get("b_weaknesses"), list) else []
            # Use the preference reason as the primary rationale if parsing was successful
            eval_results["rationale"] = eval_results["reason_for_preference"]
            print(f"    Successfully parsed evaluation for {model_a_key} vs {model_b_key}.")
        else:
             # Keep the earlier rationale if JSON parsing failed
             pass


    except Exception as e:
        eval_results["rationale"] = f"An unexpected error occurred during evaluation: {str(e)}"
        print(f"    ERROR: Unexpected Evaluation Error: {e}")
        st.error(f"Evaluation failed unexpectedly: {e}") # Show critical errors in UI

    # Add a delay after evaluation call
    time.sleep(1.0) # Small delay after evaluation

    return eval_results

# --- Quantitative Evaluation Runner ---
def run_quantitative_evaluation(conversations_dict, scenarios, models_to_test_keys):
    """Evaluates conversations using objective linguistic metrics"""
    print("\n--- Running Quantitative Evaluation ---")
    results = []
    total_tasks = len(scenarios) * len(models_to_test_keys)
    progress_bar = st.progress(0, text="Running quantitative metrics...")
    completed_tasks = 0

    for scenario in scenarios:
        for model_key in models_to_test_keys:
            conv_key = f"{scenario['name']}_{model_key}"
            if conv_key not in conversations_dict:
                print(f"  Skipping quantitative metrics for {model_key} on '{scenario['name']}' (conversation missing).")
                results.append({
                    "evaluation_type": "quantitative",
                    "model": model_key,
                    "scenario": scenario['name'],
                    "metrics": {
                        "topic_coverage_score": 0,
                        "question_complexity": 0,
                        "lexical_diversity": 0,
                        "conversation_coherence": 0
                    },
                    "status": "skipped_missing_conversation"
                })
                continue

            print(f"  Calculating metrics for {model_key} on '{scenario['name']}'...")
            conversation = conversations_dict[conv_key]

            # Calculate metrics
            metrics = {
                "topic_coverage_score": calculate_topic_coverage(conversation, BUSINESS_TOPICS),
                "question_complexity": calculate_question_complexity(conversation),
                "lexical_diversity": calculate_lexical_diversity(conversation),
                "conversation_coherence": calculate_conversation_coherence(conversation) # Placeholder
            }

            # Store results
            results.append({
                "evaluation_type": "quantitative",
                "model": model_key,
                "scenario": scenario['name'],
                "metrics": metrics,
                "status": "completed"
            })
            completed_tasks += 1
            progress = completed_tasks / total_tasks
            progress_bar.progress(progress, text=f"Running quantitative metrics... ({completed_tasks}/{total_tasks})")

    progress_bar.empty() # Clear progress bar
    print("--- Finished Quantitative Evaluation ---")
    return results

# --- Placeholder Evaluation Runners ---
def run_self_evaluation(conversations_dict, scenarios, models_to_test_keys, num_turns):
    """Placeholder for self-evaluation"""
    print("\n--- Running Self Evaluation (Placeholder) ---")
    # In a real implementation, each model would get a prompt asking it to evaluate
    # its own generated conversation (conversations_dict[f"{scenario['name']}_{model_key}"])
    # based on certain criteria (e.g., coherence, adherence to persona).
    print("  Skipping self-evaluation (not implemented).")
    return [] # Return empty list as it's not implemented

def run_expert_comparison(conversations_dict, scenarios, models_to_test_keys, expert_scenarios, num_turns):
    """Placeholder for comparison against expert/reference conversations"""
    print("\n--- Running Expert Comparison (Placeholder) ---")
    # In a real implementation, you would compare the generated conversations
    # (from conversations_dict) against pre-defined "golden" conversations (expert_scenarios)
    # for the same scenarios, perhaps using embedding similarity or other metrics.
    print("  Skipping expert comparison (not implemented).")
    return [] # Return empty list as it's not implemented

# --- Main Evaluation Framework Runner ---
def run_enhanced_evaluation_framework(scenarios, models_to_test_keys, num_turns=5):
    """Runs the full evaluation suite: simulations, pairwise comparisons, and quantitative metrics."""
    print(f"\n--- Starting Enhanced Evaluation Framework ---")
    print(f"Scenarios: {[s['name'] for s in scenarios]}")
    print(f"Models: {models_to_test_keys}")
    print(f"Turns per conversation: {num_turns}")

    results = []
    conversations_cache = {} # Cache conversations: key = f"{scenario_name}_{model_key}"

    # --- Step 1: Simulate all conversations ---
    st.info(f"Simulating conversations for {len(models_to_test_keys)} models across {len(scenarios)} scenarios...")
    total_simulations = len(scenarios) * len(models_to_test_keys)
    sim_progress = st.progress(0, text=f"Running simulations (0/{total_simulations})...")
    completed_simulations = 0

    for scenario in scenarios:
        for model_key in models_to_test_keys:
            conv_key = f"{scenario['name']}_{model_key}"
            print(f"\nGenerating conversation for {conv_key}")
            conversation = simulate_assessment_conversation(model_key, scenario, num_turns)
            conversations_cache[conv_key] = conversation
            completed_simulations += 1
            sim_progress.progress(completed_simulations / total_simulations,
                                  text=f"Running simulations ({completed_simulations}/{total_simulations})...")
    sim_progress.empty()
    st.success("All conversation simulations completed.")

    # --- Step 2: Pairwise Evaluation (A/B/C Comparison) ---
    if len(models_to_test_keys) >= 3:
        st.info("Running pairwise comparisons (A/B testing evaluated by a third model)...")
        # Generate all permutations of (Model A, Model B, Evaluator) where A, B, E are distinct
        model_permutations = list(itertools.permutations(models_to_test_keys, 3))
        print(f"Total pairwise permutations: {len(model_permutations)}")

        total_pairwise_evals = len(scenarios) * len(model_permutations)
        pairwise_progress = st.progress(0, text=f"Running pairwise evaluations (0/{total_pairwise_evals})...")
        completed_pairwise_evals = 0

        pairwise_comparison_results = []
        for scenario in scenarios:
            for model_a, model_b, evaluator_model in model_permutations:
                print(f"\nEvaluating Scenario: '{scenario['name']}'")
                print(f"  Model A: {model_a}, Model B: {model_b}, Evaluator: {evaluator_model}")

                conv_a_key = f"{scenario['name']}_{model_a}"
                conv_b_key = f"{scenario['name']}_{model_b}"

                # Retrieve conversations from cache
                conv_a = conversations_cache.get(conv_a_key)
                conv_b = conversations_cache.get(conv_b_key)

                if conv_a is None or conv_b is None:
                    print(f"  Skipping evaluation - conversation missing for {model_a} or {model_b}")
                    # Optionally log skipped evaluation
                    skipped_eval = {
                         "evaluation_type": "pairwise_comparison", "evaluator_model": evaluator_model,
                         "scenario": scenario['name'], "model_a": model_a, "model_b": model_b,
                         "preference": "N/A", "rationale": "Skipped due to missing conversation data.",
                         "raw_eval_output": "N/A - Skipped"
                    }
                    pairwise_comparison_results.append(skipped_eval)
                else:
                    eval_result = evaluate_conversations(evaluator_model, scenario, conv_a, model_a, conv_b, model_b)
                    pairwise_comparison_results.append(eval_result)

                completed_pairwise_evals += 1
                pairwise_progress.progress(completed_pairwise_evals / total_pairwise_evals,
                                           text=f"Running pairwise evaluations ({completed_pairwise_evals}/{total_pairwise_evals})...")

        results.extend(pairwise_comparison_results)
        pairwise_progress.empty()
        st.success("Pairwise comparisons completed.")
    else:
        st.warning("Skipping pairwise evaluation: requires at least 3 models selected.")

    # --- Step 3: Quantitative Metrics ---
    st.info("Calculating quantitative metrics...")
    quant_evals = run_quantitative_evaluation(conversations_cache, scenarios, models_to_test_keys)
    results.extend(quant_evals)
    st.success("Quantitative metrics calculated.")

    # --- Step 4 & 5: Self-Evaluation and Expert Comparison (Placeholders) ---
    st.info("Running placeholder self-evaluation...")
    self_evals = run_self_evaluation(conversations_cache, scenarios, models_to_test_keys, num_turns)
    results.extend(self_evals)
    # if "expert_scenarios" in st.session_state: # Check if expert data is available
    #     st.info("Running placeholder expert comparison...")
    #     expert_evals = run_expert_comparison(conversations_cache, scenarios, models_to_test_keys,
    #                                         st.session_state.expert_scenarios, num_turns)
    #     results.extend(expert_evals)
    # else:
    #     print("Skipping expert comparison (no expert data found).")
    print("--- Enhanced Evaluation Framework Finished ---")
    return results


# --- Main Application Logic ---
def main():
    with st.sidebar:
        # Use the custom class for the header
        st.markdown('<div class="sidebar-header"><h1>💼 Business Insights Hub</h1></div>', unsafe_allow_html=True)

        st.markdown("### Navigation")
        page_options = [
            # "💰 Company Valuation", # Placeholder - commented out as not implemented
            "📊 Business Assessment",
            "🔬 Model Evaluation Framework"
        ]
        # Use index=1 to default to "Business Assessment" if Valuation is commented out
        default_index = 1 if "💰 Company Valuation" not in page_options else 0
        page = st.radio("Choose a tool:", page_options, index=default_index, key="main_nav", label_visibility="collapsed")

        st.markdown("---")
        # Add API Key status
        st.markdown("#### API Status")
        if 'gemini' in available_models:
            st.success(f"Gemini ({available_models['gemini']['name']}) Initialized", icon="✅")
        else:
            st.warning("Gemini API Failed/Missing Key", icon="⚠️")

        groq_models_count = sum(1 for k in available_models if k.startswith('groq'))
        if groq_models_count > 0:
             st.success(f"Groq ({groq_models_count} models) Initialized", icon="✅")
        elif GROQ_API_KEY is not None: # Key exists but client failed
             st.error("Groq API Key Found but Client Failed", icon="🚨")
        else: # No key
             st.warning("Groq API Failed/Missing Key", icon="⚠️")

        st.markdown("---")
        st.markdown(f"<div style='text-align: center; padding: 1rem 0; font-size: 0.8rem; color: #64748B;'>Powered by AI<br>{datetime.now().strftime('%B %d, %Y')}</div>", unsafe_allow_html=True)

    # Initialize session state variables
    if 'valuation_data' not in st.session_state:
        st.session_state.valuation_data = {} # Keep even if page is commented out
    if 'assessment_responses' not in st.session_state: # Legacy? Not directly used now
        st.session_state.assessment_responses = {}
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'current_question_idx' not in st.session_state:
        st.session_state.current_question_idx = 0
    if 'assessment_completed' not in st.session_state:
        st.session_state.assessment_completed = False
    if 'evaluation_results' not in st.session_state: # For eval page
        st.session_state.evaluation_results = None
    if 'assessment_model_key' not in st.session_state: # Model used for current assessment
        st.session_state.assessment_model_key = 'gemini' if 'gemini' in available_models else (list(available_models.keys())[0] if available_models else None)


    # --- Page Routing ---
    if "Company Valuation" in page:
        render_valuation_page()
    elif "Business Assessment" in page:
        render_assessment_page()
    elif "Model Evaluation Framework" in page:
        render_evaluation_page()

# --- Page Rendering Functions ---

def render_valuation_page():
    st.markdown("# 💰 Company Valuation Estimator (Placeholder)")
    st.markdown("_(This section is under development)_")
    st.warning("Valuation features are not yet implemented.")
    # Placeholder content:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Enter Company Financials")
    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Annual Revenue ($)", min_value=0, step=10000, disabled=True)
        st.number_input("Annual Profit/Loss ($)", step=5000, disabled=True)
    with col2:
        st.selectbox("Industry", ["SaaS", "E-commerce", "Restaurant", "Service"], disabled=True)
        st.slider("Growth Rate (% YoY)", 0, 200, 20, disabled=True)
    st.button("Calculate Valuation (Disabled)", disabled=True)
    st.markdown("</div>", unsafe_allow_html=True)
    pass

def display_valuation_results(valuation_data, similar_businesses):
    # Placeholder function
    pass

def render_assessment_page():
    st.markdown("# 📊 Business Assessment")
    st.markdown("Get personalized insights through an AI-driven business evaluation interview.")

    max_questions = 10 # Reduced max questions for quicker interaction
    default_model = 'gemini' if 'gemini' in available_models else list(available_models.keys())[0]

    # Allow user to choose model for the assessment
    model_options = list(available_models.keys())
    if st.session_state.assessment_model_key not in model_options: # Handle case where previous model is no longer available
        st.session_state.assessment_model_key = default_model

    st.session_state.assessment_model_key = st.selectbox(
        "Select AI Interviewer Model:",
        options=model_options,
        index=model_options.index(st.session_state.assessment_model_key) if st.session_state.assessment_model_key in model_options else 0,
        key="assessment_model_selector",
        help="Choose the AI model that will ask you questions."
    )
    assessment_model_key = st.session_state.assessment_model_key

    if not assessment_model_key:
        st.error("No AI model available for assessment. Please check API configurations.")
        return

    # Reset button
    if st.button("🔄 Start New Assessment", key="reset_assessment"):
        st.session_state.conversation_history = []
        st.session_state.current_question_idx = 0
        st.session_state.assessment_completed = False
        st.rerun()

    # Display progress
    progress_value = min(1.0, st.session_state.current_question_idx / max_questions) if max_questions > 0 else 0
    st.progress(progress_value, text=f"Question {st.session_state.current_question_idx + 1} of {max_questions}")

    if not st.session_state.assessment_completed and st.session_state.current_question_idx < max_questions:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        # Generate or get current question
        current_question = ""
        question_generated = False
        with st.spinner("AI Interviewer is thinking..."):
            if not st.session_state.conversation_history:
                # Initial question
                current_question = "Great, let's start. Could you please give me a brief overview of your business and the primary problem you're aiming to solve?"
                question_generated = True
            elif 'last_question' not in st.session_state or not st.session_state.last_question:
                # Generate next question using the chosen model
                history_context = "Conversation History:\n" + "\n".join([
                    f"Interviewer Q{i+1}: {exchange['question']}\nYou A{i+1}: {exchange['answer']}"
                    for i, exchange in enumerate(st.session_state.conversation_history)
                ])
                next_q_prompt = f"""Based on the conversation history below, act as an insightful investor and ask the single most relevant follow-up question to evaluate this business further. Focus on uncovering key missing information or probing deeper into a previous point. Avoid generic questions.

                {history_context}

                Next Interviewer Question:"""
                current_question = ask_model(assessment_model_key, next_q_prompt)

                if current_question.startswith("Error:"):
                    st.error(f"Failed to generate the next question: {current_question}")
                    # Provide a generic fallback question
                    current_question = "Could you elaborate on your financial projections or key metrics?"
                else:
                    # Clean up potential artifacts from model response
                    current_question = current_question.strip().strip('"').strip()
                question_generated = True
                st.session_state.last_question = current_question # Store the generated question
            else:
                # Use the stored question if we are re-running after submit failed validation etc.
                current_question = st.session_state.last_question
                question_generated = True # It was generated previously

        if question_generated and current_question:
            st.markdown(f"#### Question {st.session_state.current_question_idx + 1}")
            st.markdown(f"**{current_question}**")

            # Use a form for the answer submission
            with st.form(key=f"answer_form_{st.session_state.current_question_idx}"):
                answer = st.text_area("Your Answer:", height=150, key=f"answer_input_{st.session_state.current_question_idx}")
                submitted = st.form_submit_button("Submit Answer", use_container_width=True)

                if submitted:
                    if answer.strip():
                        st.session_state.conversation_history.append({
                            "question": current_question,
                            "answer": answer
                        })
                        st.session_state.current_question_idx += 1
                        st.session_state.last_question = None # Clear last question to force generation of next one

                        if st.session_state.current_question_idx >= max_questions:
                            st.session_state.assessment_completed = True

                        st.rerun() # Rerun to display next question or results
                    else:
                        st.warning("Please provide an answer before submitting.", icon="⚠️")
        elif not question_generated:
            st.warning("Waiting for the AI interviewer to ask a question...")
        else: # Question generated but was empty/error after fallback
             st.error("Could not proceed with the assessment due to an issue generating the question.")


        st.markdown("</div>", unsafe_allow_html=True)

    elif st.session_state.assessment_completed:
        st.success("Business Assessment Interview Completed!", icon="✅")
        display_assessment_results(st.session_state.assessment_model_key) # Pass model used for the interview

def display_assessment_results(model_key_used):
    """Displays the results and analysis after the assessment interview is complete."""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 📈 Business Assessment Analysis")
    st.markdown(f"Based on your conversation with the **{model_key_used}** model.")

    # Display Conversation History
    with st.expander("View Full Interview Transcript"):
        for i, qa in enumerate(st.session_state.conversation_history):
            st.markdown(f"**Q{i+1}:** {qa['question']}")
            st.text_area(f"A{i+1}", value=qa['answer'], height=100, disabled=True, key=f"transcript_a_{i}")
            st.markdown("---")


    conversation_text = "\n\n".join([
        f"Interviewer Q{i+1}: {qa['question']}\nUser A{i+1}: {qa['answer']}"
        for i, qa in enumerate(st.session_state.conversation_history)
    ])

    analysis_prompt = f"""
    Analyze the following business assessment interview transcript. The user provided the answers.
    Your goal is to provide a structured analysis based *only* on the information present in the transcript. Do not invent information.

    Interview Transcript:
    ---
    {conversation_text}
    ---

    Provide a detailed analysis covering the following sections:
    1.  **Business Profile Summary:** Briefly summarize the business based on the user's answers.
    2.  **Key Strengths:** Identify potential strengths mentioned or implied in the conversation.
    3.  **Potential Weaknesses/Risks:** Identify potential weaknesses, risks, or areas needing clarification based on the conversation.
    4.  **Market & Competition:** Comment on what was revealed about the market, customers, and competition. Note any gaps.
    5.  **Financial Picture:** Summarize any financial details mentioned. Highlight missing financial information crucial for assessment.
    6.  **Actionable Insights & Next Steps:** Suggest 2-3 key areas the user might need to elaborate on or develop further based *only* on the interview content. Phrase these as constructive suggestions.
    7.  **Overall Impression (Based on Transcript):** Give a brief concluding thought on the clarity and completeness of the information provided *in this interview*.

    Format the response using Markdown with clear headings (e.g., `### Business Profile Summary`) and bullet points for lists. Focus on objective analysis of the provided text.
    """
    # Use a capable model for analysis (e.g., Gemini or a larger Groq model if available)
    analysis_model_key = 'gemini' if 'gemini' in available_models else list(available_models.keys())[0] # Default
    if not analysis_model_key:
         st.error("No model available to perform the analysis.")
         st.markdown("</div>", unsafe_allow_html=True)
         return

    st.info(f"Generating analysis using: **{analysis_model_key}**...")

    with st.spinner("🤖 Analyzing conversation and generating insights..."):
        analysis = ask_model(analysis_model_key, analysis_prompt)
        if analysis.startswith("Error:"):
            st.error(f"Failed to generate analysis: {analysis}")
        else:
            st.markdown(analysis)

    # Removed the second "Start New Assessment" button here, rely on the one at the top.
    st.markdown("</div>", unsafe_allow_html=True)


# --- NEW: Model Evaluation Page ---
def render_evaluation_page():
    st.markdown("# 🔬 Conversational AI Evaluation Framework")
    st.markdown("""
        Compare the performance of different AI models in simulating a business assessment interview.
        This tool runs simulations where AI models act as both the interviewer and the entrepreneur.
        It then uses AI models to evaluate these simulations based on relevance, depth, consistency, and coverage.
        Finally, it presents aggregated scores and qualitative feedback.
    """)
    st.markdown("---")

    st.markdown('<div class="card-evaluation">', unsafe_allow_html=True)
    st.markdown("### Evaluation Configuration")

    # Model Selection
    available_model_keys = list(available_models.keys())
    if not available_model_keys:
        st.error("No models available for evaluation.", icon="🚨")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Default selection logic: Try to select Gemini and two Groq models if possible
    default_selection = []
    if 'gemini' in available_model_keys:
        default_selection.append('gemini')
    groq_keys = [k for k in available_model_keys if k.startswith('groq')]
    default_selection.extend(groq_keys[:max(0, 3 - len(default_selection))]) # Add up to 2 groq keys
    # If still less than 3, add any remaining models
    default_selection.extend([k for k in available_model_keys if k not in default_selection][:max(0, 3 - len(default_selection))])


    selected_models = st.multiselect(
        "Select Models to Evaluate (Requires at least 3 for Pairwise Comparison)",
        options=available_model_keys,
        default=default_selection if len(available_model_keys) >=3 else available_model_keys,
        help="Choose the models you want to compare. Select 3+ for the A/B/C evaluation."
    )

    # Scenario Selection
    scenario_options = [s['name'] for s in BUSINESS_SCENARIOS]
    selected_scenario_names = st.multiselect(
        "Select Business Scenarios",
        options=scenario_options,
        default=scenario_options[:min(3, len(scenario_options))], # Default to first 3 scenarios
        help="Choose the business contexts for the simulated interviews."
    )
    selected_scenarios = [s for s in BUSINESS_SCENARIOS if s['name'] in selected_scenario_names]

    # Conversation Turns
    num_turns = st.slider(
        "Number of Turns per Simulated Conversation",
        min_value=3, max_value=8, value=5,
        help="Each 'turn' consists of one question from the AI interviewer and one answer from the AI entrepreneur."
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # --- Run Evaluation ---
    run_button_disabled = len(selected_models) < 3 or not selected_scenarios
    run_tooltip = ""
    if not selected_scenarios:
        run_tooltip = "Please select at least one scenario."
    elif len(selected_models) < 3:
         run_tooltip = "Please select at least 3 models for pairwise comparison."


    if st.button("▶️ Run Evaluation Framework", disabled=run_button_disabled, help=run_tooltip, use_container_width=True):
        if len(selected_models) >= 3 and selected_scenarios:
            st.session_state.evaluation_results = None # Clear previous results
            st.info("Starting evaluation process... This may take several minutes depending on the number of models, scenarios, and turns.", icon="⏳")
            # Use with st.spinner for the entire block
            with st.spinner(f"Running evaluations across {len(selected_scenarios)} scenarios and {len(list(itertools.permutations(selected_models, 3)))} pairwise comparisons..."):
                 try:
                     evaluation_start_time = time.time()
                     st.session_state.evaluation_results = run_enhanced_evaluation_framework(
                         selected_scenarios,
                         selected_models,
                         num_turns
                     )
                     evaluation_duration = time.time() - evaluation_start_time
                     st.success(f"✅ Evaluation complete! Took {evaluation_duration:.2f} seconds.")
                 except Exception as e:
                     st.error(f"An error occurred during the evaluation process: {e}")
                     st.exception(e) # Show traceback
                     st.session_state.evaluation_results = [] # Indicate failure but avoid NoneType errors later

            st.rerun() # Rerun to display results now stored in session state
        # No need for else block here as button is disabled if conditions not met

    # --- Display Results ---
    if st.session_state.evaluation_results is not None:
        st.markdown("---")
        st.markdown("### 📊 Evaluation Results Analysis")

        if not st.session_state.evaluation_results:
            st.warning("Evaluation ran but produced no results. Check logs or potential API issues.", icon="⚠️")
            return # Stop further processing if results are empty

        # Convert results to DataFrame
        try:
            results_df = pd.DataFrame(st.session_state.evaluation_results)
        except Exception as e:
            st.error(f"Failed to create DataFrame from results: {e}")
            st.write(st.session_state.evaluation_results) # Print raw results for debugging
            return

        # --- Data Preparation and Aggregation ---
        pairwise_df = results_df[results_df['evaluation_type'] == 'pairwise_comparison'].copy()
        quant_df = results_df[results_df['evaluation_type'] == 'quantitative'].copy()
        # Add placeholders for other types if/when implemented
        # self_eval_df = results_df[results_df['evaluation_type'] == 'self_eval'].copy()

        if not pairwise_df.empty:
            pairwise_df['avg_score_a'] = pairwise_df[['score_a_relevance', 'score_a_depth', 'score_a_consistency', 'score_a_coverage']].mean(axis=1)
            pairwise_df['avg_score_b'] = pairwise_df[['score_b_relevance', 'score_b_depth', 'score_b_consistency', 'score_b_coverage']].mean(axis=1)

            # Calculate aggregate scores received by each model
            scores_a = pairwise_df[['model_a', 'avg_score_a']].rename(columns={'model_a': 'model', 'avg_score_a': 'avg_score_received'})
            scores_b = pairwise_df[['model_b', 'avg_score_b']].rename(columns={'model_b': 'model', 'avg_score_b': 'avg_score_received'})
            all_scores = pd.concat([scores_a, scores_b]).dropna(subset=['avg_score_received'])
            avg_scores_received = all_scores.groupby('model')['avg_score_received'].mean().reset_index().sort_values('avg_score_received', ascending=False)

            # Calculate preference counts
            # Count preferences where the model was either A or B and was preferred
            prefs_a = pairwise_df[pairwise_df['preference'] == 'Model A']['model_a']
            prefs_b = pairwise_df[pairwise_df['preference'] == 'Model B']['model_b']
            all_prefs = pd.concat([prefs_a, prefs_b])
            preference_counts = all_prefs.value_counts().reset_index()
            preference_counts.columns = ['model', 'preference_wins']

            # Calculate evaluator bias (average score given by each model when acting as evaluator)
            evaluator_scores_given = pairwise_df.groupby('evaluator_model')[['avg_score_a', 'avg_score_b']].mean()
            evaluator_scores_given['avg_score_given'] = evaluator_scores_given.mean(axis=1)
            evaluator_bias = evaluator_scores_given[['avg_score_given']].reset_index().sort_values('avg_score_given', ascending=False)

        if not quant_df.empty:
             # Extract nested metrics into columns
             quant_metrics_df = pd.json_normalize(quant_df['metrics'])
             quant_df = pd.concat([quant_df.drop(columns=['metrics']), quant_metrics_df], axis=1)
             # Calculate average quantitative scores per model
             avg_quant_scores = quant_df.groupby('model')[[
                 'topic_coverage_score', 'question_complexity', 'lexical_diversity', 'conversation_coherence'
                 ]].mean().reset_index()


        # --- Display Aggregated Charts ---
        st.markdown("#### Summary Performance Metrics")
        if not pairwise_df.empty:
            cols = st.columns(2)
            with cols[0]:
                 st.markdown("**Average Score Received (Pairwise)**")
                 fig_scores = px.bar(avg_scores_received, x='model', y='avg_score_received', title="Average Score Received (When Evaluated)",
                                     labels={'avg_score_received': 'Avg. Score (1-5)', 'model': 'Model Evaluated'},
                                     text_auto='.2f', height=400)
                 fig_scores.update_layout(yaxis_range=[0, 5]) # Set y-axis from 0 to 5
                 st.plotly_chart(fig_scores, use_container_width=True)

            with cols[1]:
                 st.markdown("**Preference Wins (Pairwise)**")
                 fig_pref = px.bar(preference_counts, x='model', y='preference_wins', title="Times Model Was Preferred",
                                    labels={'preference_wins': '# Times Preferred', 'model': 'Preferred Model'},
                                    text_auto=True, height=400)
                 st.plotly_chart(fig_pref, use_container_width=True)

            # Consider adding evaluator bias chart if interesting
            # st.markdown("**Evaluator Tendencies**")
            # fig_bias = px.bar(evaluator_bias, x='evaluator_model', y='avg_score_given', title="Average Score Given by Evaluator", ...)
            # st.plotly_chart(fig_bias, use_container_width=True)

        if not quant_df.empty:
            st.markdown("**Average Quantitative Metrics**")
            # Melt for grouped bar chart
            avg_quant_melt = avg_quant_scores.melt(id_vars='model', var_name='metric', value_name='score')
            fig_quant = px.bar(avg_quant_melt, x='model', y='score', color='metric', barmode='group',
                              title="Average Quantitative Scores by Model",
                              labels={'score': 'Avg. Score (0-100 Scale)', 'model': 'Model', 'metric': 'Metric'},
                              height=450)
            st.plotly_chart(fig_quant, use_container_width=True)


        # --- Display Detailed Results Table ---
        st.markdown("#### Detailed Evaluation Data")
        tab1, tab2 = st.tabs(["Pairwise Comparison Details", "Quantitative Metrics Details"])

        with tab1:
             if not pairwise_df.empty:
                 st.markdown("Details from A/B comparisons evaluated by a third model.")
                 # Select and reorder columns for better readability
                 display_cols_pairwise = [
                     'scenario', 'model_a', 'model_b', 'evaluator_model', 'preference',
                     'avg_score_a', 'avg_score_b', 'reason_for_preference',
                     'score_a_relevance', 'score_a_depth', 'score_a_consistency', 'score_a_coverage',
                     'score_b_relevance', 'score_b_depth', 'score_b_consistency', 'score_b_coverage',
                     'a_strengths', 'a_weaknesses', 'b_strengths', 'b_weaknesses',
                     'raw_eval_output' # Keep raw output for inspection
                 ]
                 # Filter columns that actually exist in the dataframe
                 display_cols_pairwise = [col for col in display_cols_pairwise if col in pairwise_df.columns]
                 st.dataframe(pairwise_df[display_cols_pairwise], use_container_width=True)
             else:
                 st.info("No pairwise comparison results to display (requires 3+ models).")

        with tab2:
             if not quant_df.empty:
                 st.markdown("Objective metrics calculated from simulated conversations.")
                 display_cols_quant = [
                     'scenario', 'model', 'topic_coverage_score', 'question_complexity',
                     'lexical_diversity', 'conversation_coherence', 'status'
                 ]
                 display_cols_quant = [col for col in display_cols_quant if col in quant_df.columns]
                 st.dataframe(quant_df[display_cols_quant], use_container_width=True)
             else:
                 st.info("No quantitative metric results to display.")


        # --- Download Button ---
        st.markdown("#### Download Full Results")
        try:
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
               label="💾 Download Full Evaluation Results as CSV",
               data=csv,
               file_name=f'model_evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
               mime='text/csv',
               use_container_width=True
            )
        except Exception as e:
            st.error(f"Failed to generate CSV for download: {e}")

    elif st.session_state.evaluation_results == []: # Handle case where evaluation ran but yielded no results (e.g., errors during run)
         st.warning("Evaluation ran but produced no results or failed. Check console logs or model statuses.", icon="⚠️")


# --- Main Execution ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected critical error occurred in the application: {str(e)}", icon="🚨")
        st.exception(e) # Displays the full traceback in the Streamlit app for debugging
        print("\n--- CRITICAL ERROR ---")
        import traceback
        traceback.print_exc() # Prints traceback to console
        print("---------------------\n")
