from typing import List, Dict, Any, Optional
from datetime import datetime
from ..models.assessment import Question, Answer, AssessmentSession, AssessmentResult
from ..models.business_profile import BusinessProfile
from ..core.database import DatabaseConnection
from ..core.llm_service import LLMService
from ..core.config import QUESTION_WEIGHTS, BusinessStage

class AssessmentService:
    def __init__(self):
        self.db = DatabaseConnection()
        self.llm = LLMService()

    def start_assessment(
        self,
        business_id: str,
        business_stage: str,
        industry: str
    ) -> AssessmentSession:
        """Initialize a new assessment session."""
        return AssessmentSession(
            business_id=business_id,
            business_stage=business_stage,
            industry=industry,
            start_time=datetime.now()
        )

    def get_next_question(
        self,
        session: AssessmentSession,
        previous_answers: List[Dict[str, Any]] = None
    ) -> Optional[Question]:
        """Get the next most relevant question based on previous answers."""
        if session.completion_status >= 1.0:
            return None

        # Get weights for current business stage
        stage_weights = QUESTION_WEIGHTS.get(session.business_stage, {})
        
        # Get questions already asked
        asked_questions = set(qa["question"]["id"] for qa in session.questions_answers)
        
        # Get potential next questions
        potential_questions = self._get_potential_questions(
            session.business_stage,
            session.industry,
            asked_questions
        )
        
        if not potential_questions:
            return None
            
        # Score questions based on relevance
        scored_questions = []
        for question in potential_questions:
            score = self._calculate_question_relevance(
                question,
                previous_answers,
                stage_weights
            )
            scored_questions.append((question, score))
            
        # Sort by score and return highest scoring question
        scored_questions.sort(key=lambda x: x[1], reverse=True)
        return scored_questions[0][0] if scored_questions else None

    def process_answer(
        self,
        session: AssessmentSession,
        question: Question,
        answer_text: str
    ) -> Answer:
        """Process and analyze an answer."""
        # Extract relevant information using LLM
        extracted_data = self._extract_answer_data(question, answer_text)
        
        # Calculate confidence score
        confidence_score = self._calculate_answer_confidence(
            question,
            answer_text,
            extracted_data
        )
        
        answer = Answer(
            question_id=question.id,
            response=answer_text,
            timestamp=datetime.now(),
            confidence_score=confidence_score,
            extracted_data=extracted_data
        )
        
        # Update session with new answer
        session.add_answer(question, answer)
        
        return answer

    def generate_assessment_result(
        self,
        session: AssessmentSession
    ) -> AssessmentResult:
        """Generate final assessment result and recommendations."""
        # Extract business profile from answers
        business_profile = self._build_business_profile(session)
        
        # Calculate category scores
        scores = self._calculate_category_scores(session)
        
        # Generate recommendations using LLM
        recommendations = self._generate_recommendations(
            session,
            business_profile,
            scores
        )
        
        # Identify risks and opportunities
        risks = self._identify_risk_factors(session)
        opportunities = self._identify_opportunities(session)
        
        return AssessmentResult(
            session_id=str(session.business_id),
            business_profile=business_profile,
            scores=scores,
            recommendations=recommendations,
            risk_factors=risks,
            opportunities=opportunities,
            completion_time=datetime.now()
        )

    def _get_potential_questions(
        self,
        business_stage: str,
        industry: str,
        asked_questions: set
    ) -> List[Question]:
        """Get potential next questions from the database."""
        questions = self.db.get_collection("questions").find({
            "business_stage": business_stage,
            "id": {"$nin": list(asked_questions)}
        })
        
        return [Question(**q) for q in questions]

    def _calculate_question_relevance(
        self,
        question: Question,
        previous_answers: List[Dict[str, Any]],
        stage_weights: Dict[str, float]
    ) -> float:
        """Calculate relevance score for a question."""
        base_score = stage_weights.get(question.category, 0.5)
        
        # Adjust score based on dependencies
        if question.dependencies:
            dependency_score = self._check_dependencies(
                question.dependencies,
                previous_answers
            )
            base_score *= dependency_score
            
        return base_score

    def _extract_answer_data(
        self,
        question: Question,
        answer_text: str
    ) -> Dict[str, Any]:
        """Extract structured data from answer using LLM."""
        prompt = f"""
        Extract relevant business information from this answer.
        Question Category: {question.category}
        Question: {question.text}
        Answer: {answer_text}
        
        Extract and structure key information relevant to business assessment.
        """
        
        return self.llm.validate_response(
            answer_text,
            [question.category, question.subcategory]
        )

    def _calculate_answer_confidence(
        self,
        question: Question,
        answer_text: str,
        extracted_data: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for an answer."""
        # Base confidence on answer length and completeness
        base_confidence = min(len(answer_text.split()) / 50, 1.0)
        
        # Adjust based on extracted data completeness
        data_completeness = len(extracted_data) / 3  # Assuming we expect ~3 key pieces of info
        
        return (base_confidence + data_completeness) / 2

    def _build_business_profile(
        self,
        session: AssessmentSession
    ) -> Dict[str, Any]:
        """Build comprehensive business profile from assessment answers."""
        profile = {
            "business_stage": session.business_stage,
            "industry": session.industry,
            "metrics": {},
            "analysis": {}
        }
        
        for qa in session.questions_answers:
            category = qa["question"]["category"]
            extracted_data = qa["answer"]["extracted_data"]
            
            if category not in profile["metrics"]:
                profile["metrics"][category] = {}
                
            profile["metrics"][category].update(extracted_data)
            
        return profile

    def _calculate_category_scores(
        self,
        session: AssessmentSession
    ) -> Dict[str, float]:
        """Calculate scores for each assessment category."""
        scores = {}
        
        for category, weight in QUESTION_WEIGHTS[session.business_stage].items():
            category_answers = [
                qa for qa in session.questions_answers
                if qa["question"]["category"] == category
            ]
            
            if category_answers:
                avg_confidence = sum(
                    a["answer"]["confidence_score"] for a in category_answers
                ) / len(category_answers)
                
                scores[category] = avg_confidence * weight
                
        return scores

    def _generate_recommendations(
        self,
        session: AssessmentSession,
        business_profile: Dict[str, Any],
        scores: Dict[str, float]
    ) -> List[str]:
        """Generate strategic recommendations using LLM."""
        prompt = f"""
        Based on the following business assessment, provide strategic recommendations:
        
        Business Profile: {business_profile}
        Category Scores: {scores}
        Business Stage: {session.business_stage}
        Industry: {session.industry}
        
        Provide 3-5 specific, actionable recommendations for business improvement.
        """
        
        response = self.llm.generate_response(prompt)
        
        # Parse recommendations from response
        # This could be enhanced with more structured parsing
        recommendations = [
            r.strip() for r in response.split('\n')
            if r.strip() and not r.startswith(('Based on', 'Here are', 'Recommendations:'))
        ]
        
        return recommendations[:5]  # Limit to top 5 recommendations

    def _identify_risk_factors(
        self,
        session: AssessmentSession
    ) -> List[str]:
        """Identify key risk factors from assessment answers."""
        risks = []
        
        for qa in session.questions_answers:
            answer_data = qa["answer"]["extracted_data"]
            
            # Look for specific risk indicators
            if "challenges" in answer_data:
                risks.extend(answer_data["challenges"])
                
            if "competition" in answer_data and answer_data["competition"].get("threat_level", "").lower() == "high":
                risks.append(f"High competition in {answer_data['competition'].get('area', 'market')}")
                
            # Add financial risks
            if "financials" in answer_data:
                fin = answer_data["financials"]
                if fin.get("burn_rate", 0) > fin.get("revenue", 0):
                    risks.append("High burn rate relative to revenue")
                    
        return list(set(risks))  # Remove duplicates

    def _identify_opportunities(
        self,
        session: AssessmentSession
    ) -> List[str]:
        """Identify growth opportunities from assessment answers."""
        opportunities = []
        
        for qa in session.questions_answers:
            answer_data = qa["answer"]["extracted_data"]
            
            # Look for growth indicators
            if "market" in answer_data:
                market = answer_data["market"]
                if market.get("growth_rate", 0) > 0.2:
                    opportunities.append(f"High growth potential in {market.get('segment', 'market')}")
                    
            # Add expansion opportunities
            if "expansion" in answer_data:
                opportunities.extend(answer_data["expansion"].get("opportunities", []))
                
        return list(set(opportunities))  # Remove duplicates
