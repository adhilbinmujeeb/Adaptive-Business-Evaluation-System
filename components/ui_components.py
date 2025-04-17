import streamlit as st
from services.valuation_service import calculate_valuation, calculate_valuation_confidence
from services.assessment_service import get_next_question, extract_business_profile, calculate_investment_readiness
from services.trend_service import analyze_industry_trends
from utils.data_processing import safe_float
from core.database import get_database
import json

def render_sidebar(llm_service):
    with st.sidebar:
        st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
        st.title("💼 Business Insights Hub")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("### Navigation")
        page = st.radio("", ["💰 Company Valuation", "📊 Business Assessment", "📈 Industry Trends"], key="main_nav")
        st.markdown("---")
        st.info(f"LLM Provider: {llm_service.provider.upper()} (Model: Dynamic)")
    return page

def render_valuation_page(llm_service):
    st.markdown("# 💰 Company Valuation Estimator")
    db = get_database()
    valuation_questions = [
        {"id": "company_name", "label": "Company Name", "type": "text", "required": True},
        {"id": "industry", "label": "Industry", "type": "industry_select", "required": True},
        {"id": "revenue", "label": "Annual Revenue (USD)", "type": "number", "required": False},
        {"id": "earnings", "label": "Annual Net Income (USD)", "type": "number", "required": False},
        {"id": "ebitda", "label": "Annual EBITDA (USD)", "type": "number", "required": False},
        {"id": "assets", "label": "Total Assets (USD)", "type": "number", "required": False},
        {"id": "liabilities", "label": "Total Liabilities (USD)", "type": "number", "required": False},
        {"id": "cash_flows_str", "label": "Projected Cash Flows (5 years, comma-separated USD)", "type": "cash_flow", "required": False},
        {"id": "growth", "label": "Growth Rate", "type": "slider", "options": ["Low", "Moderate", "High"], "required": True}
    ]
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
        elif question['type'] == 'industry_select':
            industries = sorted(db['listings_collection'].distinct("business_basics.industry_category") or ["Software/SaaS", "E-commerce", "Other"])
            answer = st.selectbox("Select Industry", industries, key=input_key, index=industries.index(current_value) if current_value in industries else 0)
        elif question['type'] == 'cash_flow':
            year_cols = st.columns(5)
            cash_flows_input = []
            default_flows = current_value.split(',') if current_value else ['0'] * 5
            for i, col in enumerate(year_cols):
                cf_str = col.text_input(f"Year {i+1}", key=f"cf_{i}_{question['id']}", value=default_flows[i])
                cash_flows_input.append(cf_str)
        elif question['type'] == 'slider':
            answer = st.select_slider("Select", options=question['options'], key=input_key, value=current_value or question['options'][1])

        col_back, col_next = st.columns([1, 5])
        with col_back:
            if current_step > 0 and st.button("⬅️ Back"):
                st.session_state.valuation_step -= 1
                st.rerun()
        with col_next:
            if st.button("Next ➡️", use_container_width=True):
                final_answer = answer if question['type'] in ['text', 'industry_select', 'slider'] else (
                    ','.join(cash_flows_input) if question['type'] == 'cash_flow' else safe_float(raw_answer, None)
                )
                if question['required'] and not final_answer:
                    st.warning("This field is required.")
                else:
                    st.session_state.valuation_data[question['id']] = final_answer
                    st.session_state.valuation_step += 1
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.success("All information gathered!")
        st.markdown("<div class='card'><h3>Company Information Summary</h3>", unsafe_allow_html=True)
        for q in valuation_questions:
            st.write(f"**{q['label']}:** {st.session_state.valuation_data.get(q['id'], 'N/A')}")
        st.markdown("</div>", unsafe_allow_html=True)

        valuation_results = calculate_valuation(st.session_state.valuation_data)
        confidence_scores = calculate_valuation_confidence(st.session_state.valuation_data, valuation_results)
        
        st.markdown("<div class='card'><h2>Valuation Estimates</h2>", unsafe_allow_html=True)
        if valuation_results['average_valuation'] > 0:
            st.metric("Estimated Average Valuation", f"${valuation_results['average_valuation']:,.0f}")
            st.write(f"Range: **${valuation_results['valuation_range'][0]:,.0f} - ${valuation_results['valuation_range'][1]:,.0f}**")
            st.progress(confidence_scores['_overall_confidence'])
        else:
            st.warning("Could not calculate valuation.")
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Start New Valuation"):
            st.session_state.valuation_step = 0
            st.session_state.valuation_data = {}
            st.rerun()

def render_assessment_page(llm_service):
    st.markdown("# 📊 Interactive Business Assessment")
    db = get_database()
    max_questions = 10

    if st.session_state.question_count == 0 and not st.session_state.assessment_completed:
        st.markdown("<div class='card'><h4>Business Context</h4>", unsafe_allow_html=True)
        industries = sorted(db['listings_collection'].distinct("business_basics.industry_category") or ["Software/SaaS", "E-commerce", "Other"])
        st.session_state.assessment_industry = st.selectbox("Primary Industry", industries, key="assess_industry_select")
        stages = ["Concept/Idea", "Pre-Revenue", "Early Revenue", "Growth Stage", "Mature"]
        st.session_state.assessment_stage = st.selectbox("Business Stage", stages, key="assess_stage_select")
        if st.button("Start Assessment"):
            st.session_state.current_assessment_question = get_next_question([], st.session_state.assessment_industry, st.session_state.assessment_stage, db['question_paths_collection'], llm_service)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    elif not st.session_state.assessment_completed and st.session_state.question_count < max_questions:
        st.progress(st.session_state.question_count / max_questions)
        st.markdown(f"<div class='card'><h3>Question {st.session_state.question_count + 1}</h3>", unsafe_allow_html=True)
        st.write(st.session_state.current_assessment_question)
        response = st.text_area("Your Answer", key=f"assess_q_{st.session_state.question_count}")
        if st.button("Submit Answer"):
            if response:
                st.session_state.conversation_history.append({"question": st.session_state.current_assessment_question, "answer": response})
                st.session_state.question_count += 1
                if st.session_state.question_count >= max_questions:
                    st.session_state.assessment_completed = True
                else:
                    st.session_state.current_assessment_question = get_next_question(
                        st.session_state.conversation_history, st.session_state.assessment_industry, 
                        st.session_state.assessment_stage, db['question_paths_collection'], llm_service
                    )
                st.rerun()
            else:
                st.warning("Please provide an answer.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.success("Assessment Complete!")
        business_profile = extract_business_profile(
            st.session_state.conversation_history, "Unknown", st.session_state.assessment_industry, llm_service
        )
        readiness_score = calculate_investment_readiness(business_profile, st.session_state.conversation_history)
        
        st.markdown("<div class='card'><h2>Investment Readiness Score</h2>", unsafe_allow_html=True)
        st.metric("Overall Readiness", f"{readiness_score['overall_score']}/100")
        st.progress(readiness_score['overall_score'] / 100.0)
        for category, score in readiness_score['category_scores'].items():
            st.write(f"**{category.replace('_', ' ').title()}:** {score}/100")
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Start New Assessment"):
            st.session_state.conversation_history = []
            st.session_state.question_count = 0
            st.session_state.assessment_completed = False
            st.session_state.current_assessment_question = "Tell me about your business and what problem you're solving."
            st.session_state.assessment_industry = "Other"
            st.session_state.assessment_stage = "Early Revenue"
            st.rerun()

def render_trends_page():
    st.markdown("# 📈 Industry Trends Analysis")
    db = get_database()
    industries = sorted(db['business_attributes'].distinct("Business Attributes.Business Fundamentals.Regulatory Requirements.Compliance Status") or ["Software/SaaS", "E-commerce", "Other"])
    
    col1, col2 = st.columns(2)
    with col1:
        selected_industry = st.selectbox("Select Industry", industries, key="trend_industry")
    with col2:
        selected_timespan = st.selectbox("Select Timespan", ["1y", "3y", "5y"], key="trend_timespan")
    
    if st.button("Analyze Trends"):
        trend_results = analyze_industry_trends(selected_industry, selected_timespan, db['listings_collection'])
        if "error" in trend_results:
            st.error(trend_results["error"])
        elif not trend_results.get("trend_data"):
            st.info("No data found.")
        else:
            st.markdown(f"### Trend Results for {selected_industry} ({selected_timespan})")
            trend_df = pd.DataFrame(trend_results["trend_data"])
            st.dataframe(trend_df)
