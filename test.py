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
from groq import Groq # Import Groq SDK
import itertools # For permutations

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY") # Load Groq Key

# --- Page Config and CSS (Keep as is) ---
st.set_page_config(
    page_title="Business Insights Hub",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS (Keep as is)
st.markdown("""
<style>
    /* ... (keep existing CSS) ... */
    .stDataFrame {
        width: 100%;
    }
    .card-evaluation {
        background-color: #F0F9FF; /* Light blue background */
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #BEE3F8; /* Blue border */
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions (Keep safe_float, safe_int) ---
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

# --- MongoDB Connection (Keep as is) ---
@st.cache_resource(ttl=3600)
def get_mongo_client():
    # ... (keep existing code) ...
    pass # Placeholder, keep original implementation

# Initialize MongoDB connection (Keep as is)
# client = get_mongo_client()
# db = client['business_rag']
# listings_collection = db['business_listings']
# attributes_collection = db['business_attributes']
# questions_collection = db['questions']


# --- Initialize AI Clients ---
# Gemini
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model_name = 'gemini-1.5-flash' # Specify model name
    gemini_client = genai.GenerativeModel(gemini_model_name)
except Exception as e:
    st.error(f"Failed to configure Gemini API: {e}")
    gemini_client = None
    # st.stop() # Don't stop if only one fails

# Groq
try:
    if not GROQ_API_KEY:
        st.warning("GROQ_API_KEY not found in environment variables. Groq models will not be available.")
        groq_client = None
    else:
        groq_client = Groq(api_key=GROQ_API_KEY)
        # Define Groq models you want to use
        groq_model_names = ["llama3-8b-8192", "mixtral-8x7b-32768"]
        # You might want to quickly test if the client works
        # try:
        #     groq_client.chat.completions.create(messages=[{"role": "user", "content": "Test"}], model="llama3-8b-8192", max_tokens=10)
        # except Exception as groq_test_e:
        #     st.error(f"Failed to connect to Groq API or test model: {groq_test_e}")
        #     groq_client = None

except Exception as e:
    st.error(f"Failed to configure Groq API: {e}")
    groq_client = None

# Combine available models
available_models = {}
if gemini_client:
    available_models['gemini'] = {'client': gemini_client, 'name': gemini_model_name}
if groq_client:
    for name in groq_model_names:
         # Use a key friendly for dicts/display
        key_name = f"groq_{name.split('-')[0]}" # e.g., groq_llama3, groq_mixtral
        available_models[key_name] = {'client': groq_client, 'name': name}

if not available_models:
    st.error("No AI models could be initialized. Please check your API keys and configurations.")
    st.stop()

# --- LLM API Call Wrappers ---

# Keep original gemini_qna but add model selection if needed
def gemini_qna(query, context=None, model_client=None):
    """Question answering using Gemini."""
    if model_client is None:
         st.error("Gemini client not provided to gemini_qna")
         return "Error: Gemini client missing."
    try:
        system_prompt = """
        You are an expert business analyst and investor interviewer.
        Your goal is to conduct an in-depth interview to evaluate a business.
        Ask relevant, probing questions. If given context, use it.
        If asked to generate an answer as an entrepreneur, provide a plausible, concise answer based on the scenario.
        If asked to evaluate, be objective and follow the instructions precisely.
        Maintain a professional, neutral tone unless specified otherwise.
        """ # Simplified for broader use in simulation/evaluation

        full_prompt = f"{system_prompt}\n\n"
        if context:
            full_prompt += f"Context:\n{context}\n\n"
        full_prompt += f"Request:\n{query}" # Changed 'Query' to 'Request' for clarity

        response = model_client.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                candidate_count=1,
                max_output_tokens=1024, # Adjusted token limit
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
            # Handle potential blocks or errors
            try:
                # Attempt to access potential error information if available
                 print(f"Gemini Response: {response}") # Log the raw response for debugging
                 return f"Error: Could not generate response. Reason: {response.prompt_feedback}"
            except Exception:
                 return "Error: Could not generate response. Unknown reason."

    except Exception as e:
        print(f"Error in gemini_qna: {e}")
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
        Your goal is to conduct an in-depth interview to evaluate a business.
        Ask relevant, probing questions. If given context, use it.
        If asked to generate an answer as an entrepreneur, provide a plausible, concise answer based on the scenario.
        If asked to evaluate, be objective and follow the instructions precisely.
        Maintain a professional, neutral tone unless specified otherwise.
        """ # Consistent system prompt

        messages = [{"role": "system", "content": system_prompt}]
        if context:
            messages.append({"role": "system", "content": f"Context:\n{context}"}) # Use system role for context too
        messages.append({"role": "user", "content": f"Request:\n{query}"})

        chat_completion = model_client.chat.completions.create(
            messages=messages,
            model=model_name,
            temperature=0.7,
            max_tokens=1024, # Consistent token limit
            # top_p=1, # Defaults often work well
            # stop=None, # Defaults often work well
            # stream=False # Keep it simple, get full response
        )

        response_content = chat_completion.choices[0].message.content
        return response_content

    except Exception as e:
        print(f"Error in groq_qna for model {model_name}: {e}")
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

    if model_key.startswith('gemini'):
        return gemini_qna(query, context=context, model_client=client)
    elif model_key.startswith('groq'):
        return groq_qna(query, context=context, model_client=client, model_name=name)
    else:
        st.error(f"Handler not implemented for model key type: {model_key}")
        return f"Error: No handler for {model_key}"


# --- Valuation and LLM Functions (Keep calculate_valuation, get_similar_businesses) ---
# ... (Keep original functions) ...

# --- Business Assessment Simulation and Evaluation ---

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
    # Add 5-15 more diverse scenarios here...
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

def simulate_assessment_conversation(model_key, scenario, num_turns=5):
    """
    Simulates a business assessment conversation using a single model
    for both interviewer questions and entrepreneur answers.
    """
    conversation = []
    history = f"Scenario: {scenario['prompt']}\n\n"
    last_response_type = "answer" # Start by needing a question

    for turn in range(num_turns):
        # Generate Question
        if last_response_type == "answer":
            q_prompt = f"Based on the scenario and conversation history, act as the expert investor interviewer and ask the *next most important question* to evaluate this business."
            question = ask_model(model_key, q_prompt, context=history)
            if question.startswith("Error:") or not question.strip():
                 question = "[Model failed to generate question]" # Fallback
                 print(f"Warning: Model {model_key} failed to generate question for turn {turn+1}, scenario '{scenario['name']}'")
            history += f"Interviewer Q{turn+1}: {question}\n"
            last_response_type = "question"
            # Ensure we have a question before attempting an answer
            if "[Model failed" in question:
                answer = "[Skipped due to question generation failure]"
            else:
                 # Generate Answer
                a_prompt = f"Based on the scenario and conversation history (especially the last question: '{question}'), act as the entrepreneur and provide a plausible, concise answer consistent with the scenario."
                answer = ask_model(model_key, a_prompt, context=history)
                if answer.startswith("Error:") or not answer.strip():
                    answer = "[Model failed to generate answer]" # Fallback
                    print(f"Warning: Model {model_key} failed to generate answer for turn {turn+1}, scenario '{scenario['name']}'")
                history += f"Entrepreneur A{turn+1}: {answer}\n\n"
                last_response_type = "answer"

            conversation.append({"question": question, "answer": answer})

        # Add a small delay to avoid rate limits, especially with free tiers or rapid calls
        time.sleep(1.5) # Sleep 1.5 seconds between turns (adjust as needed)

    return conversation


def evaluate_conversations(evaluator_model_key, scenario, conversation_a, model_a_key, conversation_b, model_b_key):
    """
    Uses an evaluator model to compare two conversations based on a scenario.
    """
    eval_results = {
        "evaluator_model": evaluator_model_key,
        "scenario": scenario['name'],
        "model_a": model_a_key,
        "model_b": model_b_key,
        "preference": "N/A",
        "score_a_relevance": 0,
        "score_a_depth": 0,
        "score_a_consistency": 0,
        "score_b_relevance": 0,
        "score_b_depth": 0,
        "score_b_consistency": 0,
        "rationale": "Evaluation failed.",
        "raw_eval_output": ""
    }

    try:
        # Format conversations for the prompt
        conv_a_text = "\n".join([f" Q: {qa['question']}\n A: {qa['answer']}" for qa in conversation_a])
        conv_b_text = "\n".join([f" Q: {qa['question']}\n A: {qa['answer']}" for qa in conversation_b])

        eval_prompt = f"""
        **Evaluation Task:** You are an expert AI analyst. Evaluate two simulated business assessment conversations based on the provided scenario.

        **Scenario:**
        {scenario['prompt']}

        **Conversation A (Generated by Model: {model_a_key}):**
        {conv_a_text}

        **Conversation B (Generated by Model: {model_b_key}):**
        {conv_b_text}

        **Evaluation Criteria:**
        For each conversation, evaluate the following on a scale of 1 (Poor) to 5 (Excellent):
        1.  **Question Relevance:** How relevant and targeted were the interviewer's questions to the specific business scenario?
        2.  **Inquiry Depth:** Did the questions progressively delve deeper into key business aspects ( financials, market, team, etc.)?
        3.  **Answer Consistency:** How plausible and consistent were the simulated entrepreneur's answers with the initial scenario?

        **Output Format:**
        Provide your evaluation ONLY in JSON format. Do not include any text before or after the JSON block. The JSON object should have the following structure:
        {{
          "preference": "Model A" | "Model B" | "Neither",
          "score_a_relevance": <score_1_to_5>,
          "score_a_depth": <score_1_to_5>,
          "score_a_consistency": <score_1_to_5>,
          "score_b_relevance": <score_1_to_5>,
          "score_b_depth": <score_1_to_5>,
          "score_b_consistency": <score_1_to_5>,
          "rationale": "Brief explanation for your preference and scores, highlighting key strengths/weaknesses of each conversation."
        }}
        """

        evaluation_response = ask_model(evaluator_model_key, eval_prompt)
        eval_results["raw_eval_output"] = evaluation_response # Store raw output for debugging

        if evaluation_response.startswith("Error:"):
             eval_results["rationale"] = f"Evaluator model failed: {evaluation_response}"
             return eval_results

        # Attempt to parse the JSON response
        try:
            # Find the JSON block (sometimes models add extra text)
            json_match = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', evaluation_response, re.DOTALL | re.IGNORECASE)
            if json_match:
                json_str = json_match.group(1) or json_match.group(2)
                parsed_json = json.loads(json_str)

                # Update results dictionary safely
                eval_results["preference"] = parsed_json.get("preference", "Parse Error")
                eval_results["score_a_relevance"] = safe_int(parsed_json.get("score_a_relevance"), default=0)
                eval_results["score_a_depth"] = safe_int(parsed_json.get("score_a_depth"), default=0)
                eval_results["score_a_consistency"] = safe_int(parsed_json.get("score_a_consistency"), default=0)
                eval_results["score_b_relevance"] = safe_int(parsed_json.get("score_b_relevance"), default=0)
                eval_results["score_b_depth"] = safe_int(parsed_json.get("score_b_depth"), default=0)
                eval_results["score_b_consistency"] = safe_int(parsed_json.get("score_b_consistency"), default=0)
                eval_results["rationale"] = parsed_json.get("rationale", "Rationale missing in JSON.")

            else:
                eval_results["rationale"] = "Failed to find valid JSON in the evaluator's response."
                print(f"Could not parse JSON from: {evaluation_response}")

        except json.JSONDecodeError as json_e:
            eval_results["rationale"] = f"JSON Parsing Error: {json_e}. Response was: {evaluation_response[:200]}..." # Show beginning of response
            print(f"JSON Decode Error: {json_e}\nResponse: {evaluation_response}")
        except Exception as parse_e:
            eval_results["rationale"] = f"Error processing evaluation response: {parse_e}"
            print(f"Evaluation Processing Error: {parse_e}\nResponse: {evaluation_response}")


    except Exception as e:
        eval_results["rationale"] = f"An unexpected error occurred during evaluation: {str(e)}"
        print(f"Unexpected Evaluation Error: {e}")

    # Add a delay after evaluation call
    time.sleep(1.5) # Adjust as needed

    return eval_results


def run_evaluation_framework(scenarios, models_to_test_keys, num_turns=5):
    """
    Runs the full evaluation loop: simulate, evaluate, rotate.
    """
    results = []
    total_evaluations = len(scenarios) * len(list(itertools.permutations(models_to_test_keys, 3)))
    progress_bar = st.progress(0)
    status_text = st.empty()
    eval_count = 0

    if len(models_to_test_keys) < 3:
        st.error("Need at least 3 models selected to run the A/B/C evaluation.")
        return []

    for i, scenario in enumerate(scenarios):
        # Get all unique permutations of (Model A, Model B, Evaluator C)
        model_permutations = list(itertools.permutations(models_to_test_keys, 3))

        for model_a_key, model_b_key, evaluator_key in model_permutations:
            eval_count += 1
            status_text.text(f"Running Evaluation {eval_count}/{total_evaluations}: Scenario '{scenario['name']}', A='{model_a_key}', B='{model_b_key}', C='{evaluator_key}'")

            # --- Generate Conversations ---
            # Generate Conversation A
            st.write(f"  Generating Conversation A ({model_a_key})...") # Temporary feedback
            conversation_a = simulate_assessment_conversation(model_a_key, scenario, num_turns)
            # Add a check if conversation generation failed severely
            if not conversation_a or all("[Model failed" in qa['question'] for qa in conversation_a):
                 st.warning(f"Skipping evaluation for {model_a_key} on {scenario['name']} due to generation failure.")
                 # Optionally log this failure more formally
                 progress_bar.progress(eval_count / total_evaluations)
                 continue # Skip to next permutation

            # Generate Conversation B
            st.write(f"  Generating Conversation B ({model_b_key})...") # Temporary feedback
            conversation_b = simulate_assessment_conversation(model_b_key, scenario, num_turns)
            if not conversation_b or all("[Model failed" in qa['question'] for qa in conversation_b):
                 st.warning(f"Skipping evaluation involving {model_b_key} on {scenario['name']} due to generation failure.")
                 progress_bar.progress(eval_count / total_evaluations)
                 continue # Skip to next permutation


            # --- Evaluate Conversations ---
            st.write(f"  Evaluating A vs B using {evaluator_key}...") # Temporary feedback
            evaluation_result = evaluate_conversations(
                evaluator_model_key=evaluator_key,
                scenario=scenario,
                conversation_a=conversation_a,
                model_a_key=model_a_key,
                conversation_b=conversation_b,
                model_b_key=model_b_key
            )

            results.append(evaluation_result)

            # Update progress
            progress_bar.progress(eval_count / total_evaluations)

    status_text.text(f"Evaluation complete! Ran {eval_count} evaluations.")
    return results


# --- Main Application Logic ---
def main():
    with st.sidebar:
        st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
        st.title("💼 Business Insights Hub")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### Navigation")
        page_options = [
            "💰 Company Valuation",
            "📊 Business Assessment",
            "🔬 Model Evaluation Framework" # New Page
        ]
        page = st.radio("", page_options, key="main_nav")

        st.markdown("---")
        # Add API Key status
        st.markdown("#### API Status")
        st.success("Gemini API Initialized" if gemini_client else "Gemini API Failed/Missing Key")
        st.success("Groq API Initialized" if groq_client else "Groq API Failed/Missing Key")
        st.info(f"Available Models: {', '.join(available_models.keys())}")

        st.markdown("---")
        st.markdown(f"<div style='text-align: center; padding: 1rem; font-size: 0.8rem; color: #64748B;'>{datetime.now().strftime('%B %d, %Y')}</div>", unsafe_allow_html=True)

    # Initialize session state variables (Keep existing ones)
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
    # Add state for evaluation results
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None


    # --- Page Routing ---
    if "Company Valuation" in page:
        render_valuation_page()
    elif "Business Assessment" in page:
        render_assessment_page()
    elif "Model Evaluation Framework" in page:
        render_evaluation_page() # Call the new page function


# --- Page Rendering Functions ---

def render_valuation_page():
    # ... (Keep existing valuation page code) ...
    st.markdown("# 💰 Company Valuation Estimator")
    st.markdown("Estimate your company's value using multiple valuation methods and industry comparisons.")
    # ... rest of the function
    pass # Placeholder, keep original implementation


def display_valuation_results(valuation_data, similar_businesses):
     # ... (Keep existing valuation results display code) ...
    pass # Placeholder, keep original implementation


def render_assessment_page():
    # ... (Keep existing assessment page code, but ensure it uses the updated `ask_model`) ...
    st.markdown("# 📊 Business Assessment")
    st.markdown("Get personalized insights through an adaptive business evaluation.")

    max_questions = 15 # Maximum number of questions
    assessment_model_key = 'gemini' # Or let user choose? For now, hardcode default

    # Display progress
    progress = min(1.0, st.session_state.current_question_idx / max_questions)
    st.progress(progress)

    if not st.session_state.assessment_completed and st.session_state.current_question_idx < max_questions:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        # Generate or get current question
        if not st.session_state.conversation_history:
             current_question = "Tell me about your business and what problem you're solving."
             # Optionally, use LLM for the first question too based on a minimal context
             # context = "Starting a new business assessment."
             # first_q_prompt = "Ask the very first opening question for a business assessment interview."
             # current_question = ask_model(assessment_model_key, first_q_prompt, context)

        else:
            # Generate next question using the chosen model
            history_context = "\n\n".join([
                f"Q: {exchange['question']}\nA: {exchange['answer']}"
                for exchange in st.session_state.conversation_history
            ])
            next_q_prompt = f"Based on this business assessment conversation history, ask the single most relevant next question to deepen the evaluation. Focus on uncovering key missing information or probing deeper into a previous point."
            current_question = ask_model(assessment_model_key, next_q_prompt, history_context)
            if current_question.startswith("Error:"):
                 st.error(f"Failed to generate next question: {current_question}")
                 # Fallback question
                 current_question = "Can you elaborate on your financial projections?"
            else:
                 # Basic cleanup
                 current_question = current_question.strip().strip('"')

        st.markdown(f"### Question {st.session_state.current_question_idx + 1} of {max_questions}")
        st.markdown(f"**{current_question}**")

        answer = st.text_area("Your Answer", height=100, key=f"answer_{st.session_state.current_question_idx}")

        if st.button("Submit Answer", use_container_width=True):
            if answer.strip():
                st.session_state.conversation_history.append({
                    "question": current_question,
                    "answer": answer
                })
                # No need for assessment_responses separately if history holds it
                st.session_state.current_question_idx += 1

                if st.session_state.current_question_idx >= max_questions:
                    st.session_state.assessment_completed = True

                st.rerun()
            else:
                st.warning("Please provide an answer before proceeding.")

        st.markdown("</div>", unsafe_allow_html=True)

    elif st.session_state.assessment_completed:
        display_assessment_results(assessment_model_key) # Pass model used

def display_assessment_results(model_key_used):
    # ... (Keep structure but use `ask_model` for analysis) ...
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Business Assessment Results")

    conversation_text = "\n\n".join([
        f"Q: {qa['question']}\nA: {qa['answer']}" for qa in st.session_state.conversation_history
    ])

    analysis_prompt = f"""
    Analyze this business assessment interview transcript and provide a comprehensive evaluation. The interview was conducted by an AI ({model_key_used}).

    Interview Transcript:
    {conversation_text}

    Provide a detailed analysis covering:
    1. Business Profile Summary (Based on answers)
    2. SWOT Analysis (Strengths, Weaknesses, Opportunities, Threats derived from the conversation)
    3. Financial Health Assessment (Comment on data provided, identify gaps)
    4. Market Position (Based on answers about competitors, market size etc.)
    5. Key Strengths & Concerns (Highlight most critical points from the interview)
    6. Strategic Recommendations (Actionable advice based on the assessment)
    7. Investment Potential Rating (High/Medium/Low with justification based *only* on the transcript info)

    Format the response with clear headings and bullet points.
    Focus on actionable insights and specific recommendations derived *solely* from the provided transcript.
    """

    # Use a potentially better model for the final analysis, e.g., Gemini or larger Groq
    analysis_model_key = 'gemini' if 'gemini' in available_models else list(available_models.keys())[0] # Default to Gemini if available
    st.info(f"Generating analysis using: {analysis_model_key}")

    with st.spinner("Generating comprehensive business assessment..."):
        analysis = ask_model(analysis_model_key, analysis_prompt)
        if analysis.startswith("Error:"):
             st.error(f"Failed to generate analysis: {analysis}")
        else:
             st.markdown(analysis)


    if st.button("Start New Assessment", use_container_width=True):
        # Reset assessment state
        st.session_state.conversation_history = []
        st.session_state.current_question_idx = 0
        st.session_state.assessment_completed = False
        # st.session_state.assessment_responses = {} # Removed if only using history
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# --- NEW: Model Evaluation Page ---
def render_evaluation_page():
    st.markdown("# 🔬 Conversational Model Evaluation Framework")
    st.markdown("Compare the performance of different AI models in conducting simulated business assessment interviews.")
    st.markdown("---")

    st.markdown('<div class="card-evaluation">', unsafe_allow_html=True) # Use a specific card style
    st.markdown("### Configuration")

    # Model Selection
    available_model_keys = list(available_models.keys())
    selected_models = st.multiselect(
        "Select Models to Evaluate (Need at least 3)",
        options=available_model_keys,
        default=available_model_keys[:3] if len(available_model_keys) >= 3 else available_model_keys # Default to first 3 if possible
    )

    # Scenario Selection
    num_scenarios = st.slider("Number of Scenarios to Run (max 10)", 1, min(10, len(BUSINESS_SCENARIOS)), 5)
    selected_scenarios = BUSINESS_SCENARIOS[:num_scenarios]

    # Conversation Turns
    num_turns = st.slider("Number of Turns per Simulated Conversation", 3, 8, 5)

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("▶️ Run Evaluation", disabled=len(selected_models) < 3):
        if len(selected_models) < 3:
            st.error("Please select at least 3 models for the A/B/C evaluation.")
        else:
            with st.spinner(f"Running evaluations across {len(selected_scenarios)} scenarios and {len(list(itertools.permutations(selected_models, 3)))} permutations... This may take a while."):
                st.session_state.evaluation_results = run_evaluation_framework(
                    selected_scenarios,
                    selected_models,
                    num_turns
                )
            st.success("Evaluation complete!")
            st.rerun() # Rerun to display results now stored in session state

    # Display Results
    if st.session_state.evaluation_results:
        st.markdown("---")
        st.markdown("### Evaluation Results")
        results_df = pd.DataFrame(st.session_state.evaluation_results)

        # Calculate average scores for display
        results_df['avg_score_a'] = results_df[['score_a_relevance', 'score_a_depth', 'score_a_consistency']].mean(axis=1)
        results_df['avg_score_b'] = results_df[['score_b_relevance', 'score_b_depth', 'score_b_consistency']].mean(axis=1)

        # Select and reorder columns for better readability
        display_cols = [
            'scenario', 'model_a', 'model_b', 'evaluator_model', 'preference',
            'avg_score_a', 'avg_score_b', 'rationale',
            'score_a_relevance', 'score_a_depth', 'score_a_consistency',
            'score_b_relevance', 'score_b_depth', 'score_b_consistency',
            'raw_eval_output' # Keep raw output for inspection if needed
        ]
        st.dataframe(results_df[display_cols], use_container_width=True)

        # --- Aggregated Analysis ---
        st.markdown("### Aggregated Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Average Scores Received (When Evaluated)")
            # Calculate average score received by each model when it was Model A or Model B
            scores_a = results_df[['model_a', 'avg_score_a']].rename(columns={'model_a': 'model', 'avg_score_a': 'avg_score'})
            scores_b = results_df[['model_b', 'avg_score_b']].rename(columns={'model_b': 'model', 'avg_score_b': 'avg_score'})
            all_scores = pd.concat([scores_a, scores_b], ignore_index=True)
            avg_scores_received = all_scores.groupby('model')['avg_score'].mean().reset_index().sort_values('avg_score', ascending=False)
            fig_scores = px.bar(avg_scores_received, x='model', y='avg_score', title="Average Score Received by Model", labels={'avg_score': 'Average Score (1-5)'})
            st.plotly_chart(fig_scores, use_container_width=True)

        with col2:
            st.markdown("#### Preference Count (When Evaluated)")
            # Count how many times each model was preferred
            preference_counts = results_df['preference'].value_counts().reset_index()
            preference_counts.columns = ['model', 'preference_count']
            # Filter out 'Neither' or 'Parse Error' for this specific chart if needed
            preference_counts = preference_counts[preference_counts['model'].isin(selected_models)]
            fig_pref = px.pie(preference_counts, names='model', values='preference_count', title="Times Model Was Preferred")
            st.plotly_chart(fig_pref, use_container_width=True)

        # Add download button
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
           label="Download Evaluation Results as CSV",
           data=csv,
           file_name='model_evaluation_results.csv',
           mime='text/csv',
        )

    elif st.session_state.evaluation_results == []: # Handle case where evaluation ran but yielded no results
         st.warning("Evaluation ran but produced no results. Check logs or model statuses.")


# --- Footer (Keep as is) ---
def render_footer():
    # ... (Keep existing footer code) ...
    pass # Placeholder, keep original implementation


# --- Error Handling & Validation Placeholders (Keep if you have them) ---
def display_error_message(error_text):
    st.error(f"Error: {error_text}")

def validate_input(value, value_type="string", min_value=None, max_value=None):
    # ... (Keep existing validation code) ...
    pass # Placeholder, keep original implementation

def format_currency(amount):
     # ... (Keep existing format code) ...
    pass # Placeholder, keep original implementation


# --- Main Execution ---
if __name__ == "__main__":
    try:
        main()
        # render_footer() # Call footer if you have one
    except Exception as e:
        st.error(f"An critical error occurred in the main application: {str(e)}")
        import traceback
        st.exception(e) # Displays the full traceback in Streamlit app
        # traceback.print_exc() # Prints to console
