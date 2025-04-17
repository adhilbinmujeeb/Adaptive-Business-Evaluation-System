import json
from core.llm_service import LLMService  # Assuming LLMService is defined here
# import random # random was imported but not used

# Placeholder for MongoDB-like collection interaction if needed outside the function
# from pymongo import MongoClient
# client = MongoClient('mongodb://localhost:27017/')
# db = client['your_database']
# question_paths_collection = db['question_paths']

def get_next_question(conversation_history, industry, stage, question_paths_collection, llm_service: LLMService):
    """
    Determines the next question to ask based on conversation history, industry, and stage.

    Tries to follow predefined question paths first. If no path is found or an error occurs,
    it falls back to generating a question using an LLM.

    Args:
        conversation_history (list): A list of dictionaries, where each dict has "question" and "answer" keys.
        industry (str): The industry of the business being discussed.
        stage (str): The current stage of the business (e.g., Seed, Series A).
        question_paths_collection: A collection object (e.g., MongoDB collection) with a `find_one` method
                                   to retrieve predefined question paths.
        llm_service (LLMService): An instance of the LLM service client.

    Returns:
        str: The next question to ask.
    """
    if not conversation_history:
        # Handle the very first question
        return "Tell me about your business and what problem you're solving."

    try:
        # Attempt to find a predefined question path
        paths_doc = question_paths_collection.find_one({"industry": industry, "stage": stage}) or \
                   question_paths_collection.find_one({"industry": "Generic"}) # Fallback to Generic

        if paths_doc and "question_paths" in paths_doc:
            last_exchange = conversation_history[-1]
            last_question = last_exchange.get("question") # Use .get for safety

            if last_question:
                # Find the path corresponding to the last question asked
                current_path = next(
                    (p for p in paths_doc["question_paths"] if p.get("initial_question") == last_question),
                    None
                )

                # If a path and follow-ups exist, return the next predefined question
                if current_path and "follow_ups" in current_path and current_path["follow_ups"]:
                    # Assuming we always take the first follow-up for simplicity
                    return current_path["follow_ups"][0]["question"]

    except Exception as e:
        # Log the error for debugging (optional but recommended)
        print(f"Error finding predefined question path: {e}")
        # Proceed to LLM fallback
        pass

    # Fallback: Use LLM to generate the next question
    conversation_context = "\n\n".join(
        [f"Q: {e.get('question', 'N/A')}\nA: {e.get('answer', 'N/A')}" for e in conversation_history]
    )

    fallback_prompt = f"""
    Based on the conversation history, ask the next insightful question relevant to assessing a business for investment.
    Focus on uncovering key aspects like business model, market, financials, and team. Avoid asking questions already answered clearly.

    Conversation History:
    {conversation_context}

    Business Context:
    Industry: {industry}
    Stage: {stage}

    Instructions: Return only the question text.
    """
    try:
        next_question = llm_service.enhanced_groq_qna(fallback_prompt, task_type="question_generation").strip()
        # Basic validation to ensure it's a question
        if next_question and '?' not in next_question:
             next_question += '?'
        if not next_question:
            # Ultimate fallback if LLM fails
             return "Can you tell me more about your team?"
        return next_question
    except Exception as llm_error:
        print(f"Error generating question with LLM: {llm_error}")
        # Provide a generic fallback question if LLM fails
        return "What are your key priorities for the next 6 months?"


def extract_business_profile(conversation_history, company_name, industry, llm_service: LLMService):
    """
    Extracts structured business profile information from the conversation history using an LLM.

    Args:
        conversation_history (list): A list of dictionaries with "question" and "answer" keys.
        company_name (str): The name of the company.
        industry (str): The industry of the company.
        llm_service (LLMService): An instance of the LLM service client.

    Returns:
        dict: A dictionary containing the extracted business profile information,
              conforming to a predefined schema. Returns the empty schema if extraction fails.
    """
    profile_schema = {
        "basic_info": {
            "name": company_name,
            "industry": industry,
            "stage": None,
            "description": None
        },
        "financials": {
            "revenue": None, # Specify timeframe if possible (e.g., ARR, LTM)
            "profit": None,
            "ebitda": None
        },
        "market": {
            "target_customers": None,
            "market_size": None, # e.g., TAM, SAM, SOM
            "competitors": []
        },
        "team": {
            "founders": None, # Could be list of names/roles
            "team_size": None
        },
        # Add other relevant sections as needed, e.g., product, traction, funding
    }

    if not conversation_history:
        print("Warning: Conversation history is empty. Cannot extract profile.")
        return profile_schema # Return empty schema if no conversation

    full_conversation = "\n\n".join(
        [f"Q: {item.get('question', 'N/A')}\nA: {item.get('answer', 'N/A')}" for item in conversation_history]
    )

    extraction_prompt = f"""
    Analyze the following interview transcript and extract key business information.
    Fill in the values for the provided JSON schema based *only* on the information present in the transcript.
    If information for a field is not found, leave its value as `null`.
    Do not invent information.

    Schema:
    ```json
    {json.dumps(profile_schema, indent=2)}
    ```

    Interview Transcript:
    {full_conversation}

    Instructions: Return only the completed JSON object. Ensure the output is valid JSON.
    """

    try:
        llm_response = llm_service.enhanced_groq_qna(extraction_prompt, task_type="profile_extraction")
        # Assuming extract_structured_data handles parsing and validation
        extracted_data = llm_service.extract_structured_data(llm_response, profile_schema)
        # Fallback to empty schema if extraction doesn't return a valid dict
        return extracted_data if isinstance(extracted_data, dict) else profile_schema
    except Exception as e:
        print(f"Error during business profile extraction: {e}")
        return profile_schema # Return empty schema on error


def calculate_investment_readiness(business_profile: dict, assessment_responses: list = None):
    """
    Calculates an investment readiness score based on the completeness of the business profile.

    Note: The `assessment_responses` parameter is currently unused in the calculation logic
          provided in the original code. It's kept here for potential future use.

    Args:
        business_profile (dict): The extracted business profile data.
        assessment_responses (list, optional): Additional assessment responses (currently unused). Defaults to None.

    Returns:
        dict: A dictionary containing the overall readiness score, category scores,
              and recommendations.
    """
    readiness_score = {
        "overall_score": 0,
        "category_scores": {
            "business_model": 0,
            "financials": 0,
            "market_understanding": 0,
            "team": 0
        },
        "recommendations": []
    }

    # Helper function to check if a field in the profile is filled
    def check_completeness(field_path: str, weight: int) -> int:
        """Checks if a nested dictionary field exists and is non-empty."""
        value = business_profile
        try:
            for key in field_path.split('.'):
                if isinstance(value, dict):
                    value = value.get(key)
                else:
                    # Path is invalid if we encounter a non-dict intermediate value
                    return 0
            # Check if the final value is considered "empty"
            if value not in [None, "", [], {}]:
                return weight
            else:
                return 0
        except (AttributeError, KeyError):
            # Handle cases where path doesn't exist
            return 0

    # --- Calculate Category Scores based on data completeness ---
    # These weights and fields are examples and should be tuned based on investment criteria

    # Business Model Score (Max 100)
    readiness_score["category_scores"]["business_model"] = min(100, sum([
        check_completeness("basic_info.description", 50), # Is there a description?
        check_completeness("basic_info.stage", 50)       # Is the stage identified?
    ]))

    # Financials Score (Max 100) - Example: heavily weighted on revenue info
    readiness_score["category_scores"]["financials"] = min(100, sum([
        check_completeness("financials.revenue", 70),    # Is revenue mentioned?
        check_completeness("financials.profit", 15),     # Is profit mentioned?
        check_completeness("financials.ebitda", 15)      # Is EBITDA mentioned?
        # Add checks for other financial metrics if needed
    ]))

    # Market Understanding Score (Max 100)
    readiness_score["category_scores"]["market_understanding"] = min(100, sum([
        check_completeness("market.target_customers", 50), # Target customers identified?
        check_completeness("market.market_size", 30),      # Market size mentioned?
        check_completeness("market.competitors", 20)       # Competitors listed (even if list is empty, means it was addressed)?
                                                            # check_completeness returns weight if value is not [], {}, None, ""
                                                            # So, if competitors is [], weight is given. Adjust if needed.
    ]))

    # Team Score (Max 100)
    readiness_score["category_scores"]["team"] = min(100, sum([
        check_completeness("team.founders", 60),         # Founders info present?
        check_completeness("team.team_size", 40)         # Team size mentioned?
    ]))

    # --- Calculate Overall Score ---
    # Weights determining the importance of each category for overall readiness
    category_weights = {
        "business_model": 0.4,
        "financials": 0.3,
        "market_understanding": 0.2,
        "team": 0.1
    }

    overall_score_raw = sum(
        readiness_score["category_scores"][cat] * weight
        for cat, weight in category_weights.items()
    )
    readiness_score["overall_score"] = round(overall_score_raw)

    # --- Generate Basic Recommendations ---
    if readiness_score["overall_score"] < 50:
        readiness_score["recommendations"].append(
            "Significant gaps in core business information. Focus on documenting business model, financials, market, and team details."
        )
    elif readiness_score["overall_score"] < 75:
        readiness_score["recommendations"].append(
            "Moderately ready. Enhance detail in areas with lower scores, particularly financials and market specifics."
        )
    else:
         readiness_score["recommendations"].append(
            "Good foundation. Consider refining details and ensuring all key investor metrics are clearly presented."
         )

    # Add more specific recommendations based on category scores
    if readiness_score["category_scores"]["financials"] < 60:
         readiness_score["recommendations"].append("Recommendation: Provide more detailed financial data (Revenue, Profit/Loss, projections).")
    if readiness_score["category_scores"]["market_understanding"] < 60:
         readiness_score["recommendations"].append("Recommendation: Elaborate on target customers, market size analysis, and competitive landscape.")

    # Note: Consider incorporating `assessment_responses` here if it contains qualitative judgements
    # or answers to specific readiness questions not covered by the profile extraction.

    return readiness_score

# Example Usage (requires mock objects/data)
if __name__ == '__main__':

    # --- Mock LLMService ---
    class MockLLMService(LLMService):
        def enhanced_groq_qna(self, prompt, task_type):
            print(f"\n--- Mock LLM Call ({task_type}) ---")
            # print(f"Prompt: {prompt[:200]}...") # Uncomment to see prompt details
            if task_type == "question_generation":
                return "What is your current annual recurring revenue (ARR)?"
            elif task_type == "profile_extraction":
                # Simulate LLM extracting data into JSON
                mock_extracted_json = {
                    "basic_info": {"name": "TestCorp", "industry": "SaaS", "stage": "Seed", "description": "AI for cats."},
                    "financials": {"revenue": "$100k ARR", "profit": None, "ebitda": None},
                    "market": {"target_customers": "Cat owners", "market_size": "$1B TAM", "competitors": ["DogAI", "ManualPets"]},
                    "team": {"founders": "Alice (CEO), Bob (CTO)", "team_size": 5},
                }
                return json.dumps(mock_extracted_json)
            return ""

        def extract_structured_data(self, llm_response, schema):
             print("--- Mock Data Extraction ---")
             try:
                 # Basic simulation: assume LLM response is valid JSON
                 data = json.loads(llm_response)
                 # Simple validation against schema keys (top level)
                 if all(key in data for key in schema.keys()):
                    return data
                 else:
                    print("Warning: Extracted data keys mismatch schema.")
                    return schema # Return empty schema on mismatch
             except json.JSONDecodeError:
                 print("Error: LLM response was not valid JSON.")
                 return schema # Return empty schema on error

    # --- Mock Question Paths Collection ---
    class MockCollection:
        def find_one(self, query):
            print(f"--- Mock DB find_one: {query} ---")
            if query.get("industry") == "SaaS" and query.get("stage") == "Seed":
                return {
                    "industry": "SaaS",
                    "stage": "Seed",
                    "question_paths": [
                        {
                            "initial_question": "Tell me about your business and what problem you're solving.",
                            "follow_ups": [{"question": "Who are your target customers?"}]
                        },
                        {
                            "initial_question": "Who are your target customers?",
                            "follow_ups": [{"question": "What is your current traction or revenue?"}]
                        }
                    ]
                }
            elif query.get("industry") == "Generic":
                 return {
                    "industry": "Generic",
                    "question_paths": [
                         {
                            "initial_question": "Tell me about your business and what problem you're solving.",
                            "follow_ups": [{"question": "What makes your solution unique?"}]
                        }
                    ]
                 }
            return None

    # --- Simulation ---
    mock_llm = MockLLMService()
    mock_db_collection = MockCollection()
    conv_history = []
    current_industry = "SaaS"
    current_stage = "Seed"
    company = "TestCorp"

    # 1. Get first question
    q1 = get_next_question(conv_history, current_industry, current_stage, mock_db_collection, mock_llm)
    print(f"Q1: {q1}")
    a1 = "We are TestCorp, developing AI for cats to understand their needs. We target cat owners globally."
    conv_history.append({"question": q1, "answer": a1})

    # 2. Get next question (should use predefined path)
    q2 = get_next_question(conv_history, current_industry, current_stage, mock_db_collection, mock_llm)
    print(f"Q2: {q2}")
    a2 = "Our target customers are millennial cat owners with disposable income."
    conv_history.append({"question": q2, "answer": a2})

    # 3. Get next question (should use predefined path again)
    q3 = get_next_question(conv_history, current_industry, current_stage, mock_db_collection, mock_llm)
    print(f"Q3: {q3}")
    a3 = "We have $100k ARR and 5 employees, including the two founders Alice and Bob."
    conv_history.append({"question": q3, "answer": a3})

     # 4. Get next question (predefined path ends, should use LLM fallback)
    q4 = get_next_question(conv_history, current_industry, current_stage, mock_db_collection, mock_llm)
    print(f"Q4 (LLM Fallback): {q4}")
    # Assume an answer for Q4 if needed for further steps
    # a4 = "Our ARR comes from subscriptions..."
    # conv_history.append({"question": q4, "answer": a4})


    # 5. Extract Profile
    print("\n--- Extracting Profile ---")
    profile = extract_business_profile(conv_history, company, current_industry, mock_llm)
    print("Extracted Profile:")
    print(json.dumps(profile, indent=2))

    # 6. Calculate Readiness
    print("\n--- Calculating Readiness ---")
    # Assuming assessment_responses is not used yet
    readiness = calculate_investment_readiness(profile, assessment_responses=None)
    print("Investment Readiness Score:")
    print(json.dumps(readiness, indent=2))
