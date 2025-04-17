import streamlit as st
from core.llm_service import LLMService, extract_structured_data
import json

def get_next_question(conversation_history, industry, stage):
    """Get next question from questions collection based on conversation history and business context."""
    db = get_database()
    
    # Map industry and stage to question categories
    industry_categories = {
        "Software/SaaS": ["Software/SaaS", "IT & Technology", "Innovation-focused"],
        "E-commerce": ["E-commerce", "Direct-to-Consumer", "Marketplace/Platform"],
        "Manufacturing": ["Manufacturing", "Operations & Logistics"],
        "Services": ["Services", "Service-based", "Customer Service"],
        "Retail": ["Retail", "Direct-to-Consumer"],
        "Healthcare": ["Healthcare", "Biotech/Pharma"],
        "Food & Beverage": ["Food & Beverage"],
        "Real Estate": ["Real Estate"],
        "Financial Services": ["Financial Services", "Finance & Accounting"],
        "Media & Entertainment": ["Media & Entertainment"],
        "Education": ["Education"],
        "Non-profit": ["Non-profit", "Environmental, Social & Governance"],
        "Biotech/Pharma": ["Biotech/Pharma", "Research & Development"],
        "Other": ["Innovation-focused", "Differentiation"]
    }
    stage_categories = {
        "Idea/Concept Stage": ["Idea/Concept Stage", "Innovation-focused", "Research & Development"],
        "Startup/Early Stage": ["Startup/Early Stage", "Market Expansion", "Digital Transformation"],
        "Growth Stage": ["Growth Stage", "Market Expansion", "Acquisition/Partnership"],
        "Mature Stage": ["Mature Stage", "Cost Leadership", "Differentiation"],
        "Turnaround/Restructuring Stage": ["Turnaround/Restructuring Stage", "Crisis Management", "Exit Strategy"],
        "Other": ["Innovation-focused", "Market Expansion"]
    }
    
    primary_categories = industry_categories.get(industry, ["Innovation-focused"]) + stage_categories.get(stage, ["Innovation-focused"])
    primary_categories = list(set(primary_categories))  # Remove duplicates
    
    if not conversation_history:
        # Start with a question relevant to industry or stage
        question_doc = db['questions'].find_one({"category": {"$in": primary_categories}})
        return question_doc['question'] if question_doc else "What problem does your business solve?"
    
    # Check last response for keywords to select follow-up
    last_response = conversation_history[-1]["answer"].lower()
    last_question = conversation_history[-1]["question"]
    
    # Keyword-to-category mapping for follow-ups
    keyword_category_map = {
        "market gap": ["Market Expansion", "Differentiation"],
        "consumer need": ["Market Expansion", "Direct-to-Consumer"],
        "revenue": ["Finance & Accounting", "Financial Metrics"],
        "sales": ["Marketing & Sales", "Financial Metrics"],
        "product": ["Research & Development", "Innovation-focused"],
        "innovation": ["Innovation-focused", "Research & Development"],
        "team": ["Human Resources", "Team Composition"],
        "operations": ["Operations & Logistics"],
        "customer": ["Customer Service", "Direct-to-Consumer"],
        "growth": ["Growth Stage", "Market Expansion"],
        "strategy": ["Differentiation", "Cost Leadership", "Exit Strategy"]
    }
    
    # Find a category based on keywords in the response
    selected_category = None
    for keyword, categories in keyword_category_map.items():
        if keyword in last_response:
            selected_category = categories[0]
            break
    
    # Query for a follow-up question
    if selected_category:
        question_doc = db['questions'].find_one({
            "category": selected_category,
            "question": {"$ne": last_question}  # Avoid repeating the last question
        })
        if question_doc:
            return question_doc['question']
    
    # Fallback to a question in primary categories
    question_doc = db['questions'].find_one({
        "category": {"$in": primary_categories},
        "question": {"$ne": last_question}
    })
    return question_doc['question'] if question_doc else "Tell me about your business model."

def extract_business_profile(conversation_history, llm_service: LLMService):
    """Extract structured business profile from conversation history."""
    profile = {
        "business_id": None,
        "basic_info": {
            "name": None,
            "industry": None,
            "stage": None,
            "description": None
        },
        "financials": {
            "revenue": None,
            "profit": None,
            "growth_rate": None,
            "burn_rate": None
        },
        "market": {
            "target_customers": None,
            "market_size": None,
            "competitors": []
        },
        "team": {
            "founders": None,
            "key_expertise": []
        }
    }

    extraction_prompt = f"""
Based on the following conversation, extract key business details into a structured profile.
Focus on concrete facts mentioned, not inferences. If information is not provided, indicate with null.
Conversation:
{json.dumps(conversation_history, indent=2)}
Extract the information as structured JSON matching this format:
{json.dumps(profile, indent=2)}
"""
    response = llm_service.generate_response(extraction_prompt, system_message="You are a precise data extractor.")
    extracted_data = extract_structured_data(response, profile)

    if extracted_data:
        # Validate and map to business_listings and business_attributes if business_id is present
        if extracted_data.get("business_id"):
            db = get_database()
            listing = db['business_listings'].find_one({"business_id": extracted_data["business_id"]})
            attr = db['business_attributes'].find_one({"business_id": extracted_data["business_id"]})
            
            if listing:
                extracted_data["basic_info"]["name"] = listing.get("business_basics", {}).get("business_name")
                extracted_data["basic_info"]["industry"] = listing.get("business_basics", {}).get("industry_category", [])[0] if listing.get("business_basics", {}).get("industry_category") else None
                extracted_data["market"]["competitors"] = listing.get("market_information", {}).get("competitors", [])
                extracted_data["team"]["founders"] = ", ".join([f["name"] for f in listing.get("business_basics", {}).get("founder_names", [])])
            
            if attr:
                extracted_data["basic_info"]["stage"] = attr.get("Business Attributes.Business Fundamentals.Development Stage")
                extracted_data["financials"]["revenue"] = parse_revenue_bracket(
                    attr.get("Business Attributes.Financial Metrics.Revenue Brackets (Annual)")
                )
                extracted_data["financials"]["growth_rate"] = attr.get("Business Attributes.Growth & Scalability.Growth Rate")
                extracted_data["market"]["target_customers"] = attr.get("Business Attributes.Business Fundamentals.Business Model.Target Market Segment")
                extracted_data["market"]["market_size"] = attr.get("Business Attributes.Market Position.Market Size Categories")
                extracted_data["team"]["key_expertise"] = attr.get("Business Attributes.Team Composition.Expertise Coverage", "").split(", ")

        return extracted_data
    return profile

def parse_revenue_bracket(revenue_str):
    """Parse revenue bracket string to a float (midpoint)."""
    if not revenue_str or revenue_str == "not_provided":
        return None
    import re
    match = re.match(r"\$(\d+,\d{3})\s*-\s*\$(\d+,\d{3})", revenue_str)
    if match:
        low = float(match.group(1).replace(",", ""))
        high = float(match.group(2).replace(",", ""))
        return (low + high) / 2
    return None

def calculate_investment_readiness(business_profile):
    """Calculate investment readiness score based on profile completeness."""
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

    def check_completeness(data, field, weight):
        return weight if data.get(field) and data[field] != "not_provided" else 0

    # Business Model
    readiness_score["category_scores"]["business_model"] = check_completeness(
        business_profile["basic_info"], "description", 50
    )

    # Financials
    financial_score = 0
    if business_profile["financials"]["revenue"]:
        financial_score += 50
    if business_profile["financials"]["growth_rate"]:
        financial_score += 30
    readiness_score["category_scores"]["financials"] = financial_score

    # Market Understanding
    market_score = check_completeness(business_profile["market"], "target_customers", 50)
    if business_profile["market"]["competitors"]:
        market_score += 30
    readiness_score["category_scores"]["market_understanding"] = market_score

    # Team
    team_score = check_completeness(business_profile["team"], "founders", 50)
    if business_profile["team"]["key_expertise"]:
        team_score += 30
    readiness_score["category_scores"]["team"] = team_score

    # Overall Score
    weights = {"business_model": 0.3, "financials": 0.3, "market_understanding": 0.2, "team": 0.2}
    overall_score = sum(
        score * weights[category] for category, score in readiness_score["category_scores"].items()
    )
    readiness_score["overall_score"] = round(overall_score)

    # Recommendations
    if readiness_score["category_scores"]["financials"] < 50:
        readiness_score["recommendations"].append("Provide detailed financial metrics, including revenue and growth rate.")
    if readiness_score["category_scores"]["market_understanding"] < 50:
        readiness_score["recommendations"].append("Clarify your target market and competitive landscape.")

    return readiness_score
