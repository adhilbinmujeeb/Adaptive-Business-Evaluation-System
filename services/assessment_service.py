from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from models.assessment import Question, Answer, AssessmentSession, AssessmentResult
from models.business_profile import BusinessProfile
from core.database import DatabaseConnection
from core.llm_service import LLMService

class AssessmentService:
    def __init__(self):
        self.db = DatabaseConnection()
        self.llm = LLMService()
        self.interview_phases = {
            "initial_discovery": {
                "name": "Initial Discovery",
                "questions": [
                    {
                        "_id": "id_1",
                        "text": "Tell me about your business and what problem you're solving.",
                        "category": "Overview"
                    },
                    {
                        "_id": "id_2",
                        "text": "How long have you been operating and what's your current stage?",
                        "category": "Business Stage"
                    },
                    {
                        "_id": "id_3",
                        "text": "What industry are you in and who are your target customers?",
                        "category": "Market"
                    },
                    {
                        "_id": "id_4",
                        "text": "What's your revenue model and current traction?",
                        "category": "Financial"
                    }
                ]
            },
            "business_model": {
                "name": "Business Model Deep Dive",
                "questions": {
                    "digital_saas": [
                        {
                            "_id": "saas_1",
                            "text": "What's your monthly recurring revenue and growth rate?",
                            "category": "Financial"
                        },
                        {
                            "_id": "saas_2",
                            "text": "What's your customer acquisition cost compared to lifetime value?",
                            "category": "Metrics"
                        },
                        {
                            "_id": "saas_3",
                            "text": "What's your churn rate and retention strategy?",
                            "category": "Operations"
                        }
                    ],
                    "physical_product": [
                        {
                            "_id": "product_1",
                            "text": "What are your production costs and gross margins?",
                            "category": "Financial"
                        },
                        {
                            "_id": "product_2",
                            "text": "How do you manage your supply chain and inventory?",
                            "category": "Operations"
                        },
                        {
                            "_id": "product_3",
                            "text": "What are your distribution channels and retail strategy?",
                            "category": "Strategy"
                        }
                    ],
                    "service": [
                        {
                            "_id": "service_1",
                            "text": "How do you scale your service delivery beyond your personal time?",
                            "category": "Operations"
                        },
                        {
                            "_id": "service_2",
                            "text": "What's your hourly/project rate structure and utilization rate?",
                            "category": "Financial"
                        },
                        {
                            "_id": "service_3",
                            "text": "How do you maintain quality as you expand your team?",
                            "category": "Operations"
                        }
                    ]
                }
            },
            "market_analysis": {
                "name": "Market & Competition Analysis",
                "questions": [
                    {
                        "_id": "market_1",
                        "text": "What's your total addressable market size and how did you calculate it?",
                        "category": "Market"
                    },
                    {
                        "_id": "market_2",
                        "text": "Who are your top 3 competitors and how do you differentiate?",
                        "category": "Competition"
                    },
                    {
                        "_id": "market_3",
                        "text": "What barriers to entry exist in your market?",
                        "category": "Strategy"
                    },
                    {
                        "_id": "market_4",
                        "text": "What market trends are impacting your growth potential?",
                        "category": "Market"
                    }
                ]
            },
            "financial_performance": {
                "name": "Financial Performance",
                "questions": {
                    "pre_revenue": [
                        {
                            "_id": "pre_fin_1",
                            "text": "What's your burn rate and runway?",
                            "category": "Financial"
                        },
                        {
                            "_id": "pre_fin_2",
                            "text": "What are your financial projections for the next 24 months?",
                            "category": "Financial"
                        },
                        {
                            "_id": "pre_fin_3",
                            "text": "What assumptions underlie your revenue forecasts?",
                            "category": "Financial"
                        }
                    ],
                    "revenue_generating": [
                        {
                            "_id": "rev_fin_1",
                            "text": "What has your year-over-year revenue growth been?",
                            "category": "Financial"
                        },
                        {
                            "_id": "rev_fin_2",
                            "text": "Break down your cost structure between fixed and variable costs.",
                            "category": "Financial"
                        },
                        {
                            "_id": "rev_fin_3",
                            "text": "What's your path to profitability and timeline?",
                            "category": "Financial"
                        }
                    ],
                    "profitable": [
                        {
                            "_id": "prof_fin_1",
                            "text": "What's your EBITDA and how has it evolved over time?",
                            "category": "Financial"
                        },
                        {
                            "_id": "prof_fin_2",
                            "text": "What's your cash conversion cycle?",
                            "category": "Financial"
                        },
                        {
                            "_id": "prof_fin_3",
                            "text": "How do you reinvest profits back into the business?",
                            "category": "Strategy"
                        }
                    ]
                }
            }
        }

    def start_assessment(
        self,
        business_name: str,
        business_stage: str,
        industry: str
    ) -> AssessmentSession:
        """Start a new assessment session."""
        session = AssessmentSession(
            business_name=business_name,
            business_stage=business_stage,
            industry=industry,
            start_time=datetime.now(),
            questions_answers=[],
            completion_status=0.0
        )
        return session

    def get_next_question(
        self,
        session: AssessmentSession,
        previous_answers: List[Dict[str, Any]] = None
    ) -> Optional[Question]:
        """Get the next relevant question based on previous answers and business context."""
        if previous_answers is None:
            previous_answers = []

        # Determine current phase based on answers
        current_phase = self._determine_current_phase(session, previous_answers)
        
        # Get questions for current phase
        phase_questions = self._get_phase_questions(current_phase, session, previous_answers)
        
        # Filter out already answered questions
        answered_ids = {qa["question_id"] for qa in previous_answers}
        available_questions = [q for q in phase_questions if q["_id"] not in answered_ids]
        
        if not available_questions:
            return None

        # Use LLM to select most relevant question
        selected_question = self._select_next_question(available_questions, session, previous_answers)
        
        return Question(
            id=selected_question["_id"],
            text=selected_question["text"],
            category=selected_question["category"],
            follow_up_questions=selected_question.get("follow_up_questions", [])
        )

    def process_answer(
        self,
        session: AssessmentSession,
        question: Question,
        answer_text: str
    ) -> Answer:
        """Process and analyze a user's answer."""
        # Extract structured data using LLM
        structured_data = self._extract_answer_data(question, answer_text)
        
        # Check for red flags
        red_flags = self._check_red_flags(question, answer_text, structured_data)
        if red_flags:
            structured_data["red_flags"] = red_flags
        
        # Check for opportunity signals
        opportunities = self._check_opportunities(question, answer_text, structured_data)
        if opportunities:
            structured_data["opportunities"] = opportunities
        
        # Create answer object
        answer = Answer(
            question_id=question.id,
            text=answer_text,
            timestamp=datetime.now(),
            structured_data=structured_data,
            confidence_score=self._calculate_answer_confidence(answer_text)
        )
        
        # Update session progress
        total_questions = self._get_total_questions_estimate(session)
        session.questions_answers.append({"question_id": question.id, "answer": answer})
        session.completion_status = len(session.questions_answers) / total_questions
        
        return answer

    def _determine_current_phase(
        self,
        session: AssessmentSession,
        previous_answers: List[Dict[str, Any]]
    ) -> str:
        """Determine which interview phase to proceed with."""
        if not previous_answers:
            return "initial_discovery"
            
        num_answers = len(previous_answers)
        
        if num_answers < 4:
            return "initial_discovery"
        elif num_answers < 8:
            return "business_model"
        elif num_answers < 12:
            return "market_analysis"
        else:
            return "financial_performance"

    def _get_phase_questions(
        self,
        phase: str,
        session: AssessmentSession,
        previous_answers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get questions for the current phase."""
        phase_data = self.interview_phases.get(phase, {})
        
        if phase == "business_model":
            # Determine business type from previous answers
            business_type = self._determine_business_type(previous_answers)
            return phase_data.get("questions", {}).get(business_type, [])
            
        elif phase == "financial_performance":
            # Determine financial stage
            financial_stage = self._determine_financial_stage(previous_answers)
            return phase_data.get("questions", {}).get(financial_stage, [])
            
        return phase_data.get("questions", [])

    def _determine_business_type(
        self,
        previous_answers: List[Dict[str, Any]]
    ) -> str:
        """Determine the type of business from previous answers."""
        # Use LLM to analyze previous answers and categorize the business
        context = self._summarize_previous_answers(previous_answers)
        
        prompt = f"""
        Based on the following business context, determine if this is primarily a:
        1. Digital/SaaS business
        2. Physical product business
        3. Service business

        Context:
        {context}

        Return only one of: "digital_saas", "physical_product", or "service"
        """
        
        response = self.llm.generate_response(prompt)
        return response.strip().lower()

    def _determine_financial_stage(
        self,
        previous_answers: List[Dict[str, Any]]
    ) -> str:
        """Determine the financial stage of the business."""
        # Use LLM to analyze previous answers and determine financial stage
        context = self._summarize_previous_answers(previous_answers)
        
        prompt = f"""
        Based on the following business context, determine if this is a:
        1. Pre-revenue company
        2. Revenue-generating but not profitable
        3. Profitable company

        Context:
        {context}

        Return only one of: "pre_revenue", "revenue_generating", or "profitable"
        """
        
        response = self.llm.generate_response(prompt)
        return response.strip().lower()

    def _check_red_flags(
        self,
        question: Question,
        answer_text: str,
        structured_data: Dict[str, Any]
    ) -> List[str]:
        """Check for red flags in the answer."""
        prompt = f"""
        Analyze this answer for potential red flags from an investor's perspective.
        Look for issues like:
        - Inconsistent financial numbers
        - Unrealistic market size claims
        - Vague answers about competition
        - Excessive founder salaries
        - Unreasonable valuation expectations

        Question: {question.text}
        Answer: {answer_text}
        Structured Data: {json.dumps(structured_data)}

        Return a list of specific red flags, or an empty list if none found.
        """
        
        response = self.llm.generate_response(prompt)
        try:
            return json.loads(response)
        except:
            return []

    def _check_opportunities(
        self,
        question: Question,
        answer_text: str,
        structured_data: Dict[str, Any]
    ) -> List[str]:
        """Check for opportunity signals in the answer."""
        prompt = f"""
        Analyze this answer for positive opportunity signals from an investor's perspective.
        Look for indicators like:
        - Unusually high margins for the industry
        - Proprietary technology or IP
        - Evidence of product-market fit
        - Strong team with relevant experience
        - Clear customer acquisition strategy with proven ROI

        Question: {question.text}
        Answer: {answer_text}
        Structured Data: {json.dumps(structured_data)}

        Return a list of specific opportunities, or an empty list if none found.
        """
        
        response = self.llm.generate_response(prompt)
        try:
            return json.loads(response)
        except:
            return []

    def _get_total_questions_estimate(self, session: AssessmentSession) -> int:
        """Estimate total number of questions for the assessment."""
        # Base questions from each phase
        total = (
            len(self.interview_phases["initial_discovery"]["questions"]) +
            len(self.interview_phases["market_analysis"]["questions"])
        )
        
        # Add business model questions (assume average)
        business_model_questions = self.interview_phases["business_model"]["questions"]
        avg_business_model = sum(len(q) for q in business_model_questions.values()) // len(business_model_questions)
        total += avg_business_model
        
        # Add financial questions (assume average)
        financial_questions = self.interview_phases["financial_performance"]["questions"]
        avg_financial = sum(len(q) for q in financial_questions.values()) // len(financial_questions)
        total += avg_financial
        
        return total

    def _summarize_previous_answers(
        self,
        previous_answers: List[Dict[str, Any]]
    ) -> str:
        """Create a summary of previous answers for context."""
        summary = []
        for qa in previous_answers:
            question = self.db.get_question_by_id(qa["question_id"])
            answer = qa["answer"]
            summary.append(f"Q: {question['text']}\nA: {answer.text}")
        return "\n\n".join(summary)
