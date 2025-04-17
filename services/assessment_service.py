from core.llm_service import LLMService
import random
import json

def get_next_question(conversation_history, industry, stage, question_paths_collection, llm_service):
    if not conversation_history:
        return "Tell me about your business and what problem you're solving."
    
    try:
        paths_doc = question_paths_collection.find_one({"industry": industry, "stage": stage}) or \
                   question_paths_collection.find_one({"industry": "Generic"})
        if paths_doc and "question_paths" in paths_doc:
            last_exchange = conversation_history[-1]
            last_question = last_exchange["question"]
            current_path = next((p for p in paths_doc["question_paths"] if p.get("initial_question") == last_question), None)
            if current_path and "follow_ups" in current_path:
                return current_path["follow_ups"][0]["question"]
    except Exception:
        pass

    conversation_context = "\n\n".join([f"Q: {e['question']}\nA: {e['answer']}" for e in conversation_history])
    fallback_prompt = f"""
    Based on the conversation history, ask the next insightful question.
    Conversation History:
    {conversation_context}
    Industry: {industry}
    Stage: {stage}
    Return only the question text.
    """
    return llm_service.enhanced_groq_qna(fallback_prompt, task_type="question_generation").strip()

def extract_business_profile(conversation_history, company_name, industry, llm_service):
    profile_schema = {
        "basic_info": {"name": company_name, "industry": industry, "stage": None, "description": None},
        "financials": {"revenue": None, "profit": None, "ebitda": None},
        "market": {"target_customers": None, "market_size": None, "competitors": []},
        "team": {"founders": None, "team_size": None},
    }
    full_conversation = "\n\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in conversation_history])
    extraction_prompt = f"""
    Extract information into a JSON object matching this schema:
    ```json
    {json.dumps(profile_schema, indent=2)}
