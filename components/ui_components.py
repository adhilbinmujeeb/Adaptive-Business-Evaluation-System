import streamlit as st
from services.valuation_service import calculate_valuation, calculate_valuation_confidence
from services.assessment_service import get_next_question, extract_business_profile, calculate_investment_readiness
from core.database import get_database

def render_valuation_page(llm_service):
    st.markdown("# 💰 Company Valuation Estimator")
    db = get_database()
    valuation_questions = [
        {"id": "company_name", "label": "Company Name", "type": "text", "required": True},
        {"id": "business_id", "label": "Business ID (if known)", "type": "text", "required": False},
        {"id": "industry", "label": "Industry", "type": "industry_select", "required": True},
        {"id": "revenue", "label": "Annual Revenue (USD)", "type": "number", "required": False},
        {"id": "earnings", "label": "Annual Earnings (USD)", "type": "number", "required": False},
        {"id": "ebitda", "label": "Annual EBITDA (USD)", "type": "number", "required": False},
        {"id": "assets", "label": "Total Assets (USD)", "type": "number", "required": False},
        {"id": "liabilities", "label": "Total Liabilities (USD)", "type": "number", "required": False},
        {"id": "growth", "label": "Growth Rate", "type": "select", "options": ["High", "Moderate", "Low"], "required": False}
    ]

    if 'valuation_step' not in st.session_state:
        st.session_state.valuation_step = 0
        st.session_state.valuation_data = {}

    current_step = st.session_state.valuation_step

    if current_step < len(valuation_questions):
        st.progress(current_step / len(valuation_questions))
        question = valuation_questions[current_step]
        st.markdown(f"<div class='card'><h3>{question['label']}{' *' if question['required'] else ''}</h3>", unsafe_allow_html=True)
        
        input_key = f"val_input_{question['id']}"
        current_value = st.session_state.valuation_data.get(question['id'])
        
        if question['type'] == 'text':
            answer = st.text_input("Value", key=input_key, value=current_value or "")
        elif question['type'] == 'number':
            raw_answer = st.text_input("USD", key=input_key, value=str(current_value) if current_value is not None else "")
            answer = float(raw_answer) if raw_answer.replace('.', '', 1).isdigit() else None
        elif question['type'] == 'industry_select':
            try:
                industry_list = db['business_listings'].distinct("business_basics.industry_category")
                industries = sorted(set(item for sublist in industry_list if isinstance(sublist, list) for item in sublist))
                if not industries:
                    industries = ["Software/SaaS", "E-commerce", "Manufacturing", "Services", "Retail", 
                                 "Healthcare", "Food & Beverage", "Real Estate", "Financial Services", 
                                 "Media & Entertainment", "Education", "Non-profit", "Biotech/Pharma", "Other"]
            except Exception as e:
                st.error(f"Failed to fetch industries: {e}")
                industries = ["Software/SaaS", "E-commerce", "Manufacturing", "Services", "Retail", 
                             "Healthcare", "Food & Beverage", "Real Estate", "Financial Services", 
                             "Media & Entertainment", "Education", "Non-profit", "Biotech/Pharma", "Other"]
            answer = st.selectbox("Select Industry", industries, key=input_key, 
                                 index=industries.index(current_value) if current_value in industries else 0)
        elif question['type'] == 'select':
            answer = st.selectbox("Select Option", question['options'], key=input_key,
                                  index=question['options'].index(current_value) if current_value in question['options'] else 0)

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Previous", disabled=current_step == 0):
                st.session_state.valuation_step -= 1
                st.rerun()
        with col2:
            if st.button("Next", disabled=question['required'] and not answer):
                if answer:
                    st.session_state.valuation_data[question['id']] = answer
                st.session_state.valuation_step += 1
                st.rerun()
    else:
        company_data = st.session_state.valuation_data
        valuation_results = calculate_valuation(company_data)
        confidence_scores = calculate_valuation_confidence(company_data, valuation_results)

        st.markdown("## Valuation Results")
        if valuation_results:
            for method, value in valuation_results.items():
                confidence = confidence_scores.get(method, 0.0)
                st.markdown(f"**{method.replace('_', ' ').title()}:** ${value:,.2f} (Confidence: {confidence:.2%})")
            overall_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.0
            st.progress(overall_confidence)
            st.markdown(f"**Overall Confidence:** {overall_confidence:.2%}")
        else:
            st.warning("Insufficient data to calculate valuation.")

        if st.button("Start Over"):
            st.session_state.valuation_step = 0
            st.session_state.valuation_data = {}
            st.rerun()

def render_assessment_page(llm_service):
    st.markdown("# 📊 Business Assessment")
    db = get_database()

    if 'assessment_history' not in st.session_state:
        st.session_state.assessment_history = []

    # Get industry and stage from user input or valuation data
    industry_options = ["Software/SaaS", "E-commerce", "Manufacturing", "Services", "Retail", 
                       "Healthcare", "Food & Beverage", "Real Estate", "Financial Services", 
                       "Media & Entertainment", "Education", "Non-profit", "Biotech/Pharma", "Other"]
    stage_options = ["Idea/Concept Stage", "Startup/Early Stage", "Growth Stage", 
                    "Mature Stage", "Turnaround/Restructuring Stage", "Other"]
    industry = st.selectbox("Select Industry", industry_options, 
                            index=industry_options.index(st.session_state.valuation_data.get('industry', 'Other')) 
                            if 'industry' in st.session_state.valuation_data else 0, 
                            key="assessment_industry")
    stage = st.selectbox("Select Stage", stage_options, 
                         index=stage_options.index(st.session_state.valuation_data.get('stage', 'Other')) 
                         if 'stage' in st.session_state.valuation_data else 0, 
                         key="assessment_stage")

    # Get next question
    next_question = get_next_question(st.session_state.assessment_history, industry, stage)
    st.markdown(f"<div class='card'><h3>{next_question}</h3>", unsafe_allow_html=True)
    
    answer = st.text_area("Your Answer", key="assessment_answer")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Clear"):
            st.session_state.assessment_history = []
            st.rerun()
    with col2:
        if st.button("Submit Answer", disabled=not answer):
            st.session_state.assessment_history.append({
                "question": next_question,
                "answer": answer
            })
            if len(st.session_state.assessment_history) >= 3:  # Limit to 3 questions for demo
                profile = extract_business_profile(st.session_state.assessment_history, llm_service)
                readiness = calculate_investment_readiness(profile)
                
                st.markdown("## Assessment Results")
                st.markdown(f"**Overall Investment Readiness Score:** {readiness['overall_score']}/100")
                for category, score in readiness['category_scores'].items():
                    st.markdown(f"**{category.replace('_', ' ').title()}:** {score}/100")
                if readiness['recommendations']:
                    st.markdown("**Recommendations:**")
                    for rec in readiness['recommendations']:
                        st.markdown(f"- {rec}")
                
                # Update profile in business_listings
                if profile.get("business_id"):
                    db['business_listings'].update_one(
                        {"business_id": profile["business_id"]},
                        {"$set": {"extracted_profile": profile}},
                        upsert=False
                    )
                
                if st.button("Start Over"):
                    st.session_state.assessment_history = []
                    st.session_state.valuation_step = 0
                    st.session_state.valuation_data = {}
                    st.rerun()
            else:
                st.rerun()
