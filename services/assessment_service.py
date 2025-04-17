from typing import List, Dict, Optional, Any
from datetime import datetime
from models.assessment import Question, Answer, AssessmentSession, AssessmentResult
from models.business_profile import BusinessProfile
from core.database import DatabaseConnection
from core.llm_service import LLMService
from core.config import BusinessStage, FUNCTIONAL_AREAS

class AssessmentService:
    def __init__(self):
        self.db = DatabaseConnection()
        self.llm = LLMService()

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
        """Get the next relevant question based on previous answers."""
        if previous_answers is None:
            previous_answers = []

        # Get base questions for business stage
        stage_questions = self.db.get_questions_for_stage(session.business_stage)
        
        # Filter out already answered questions
        answered_ids = {qa["question_id"] for qa in previous_answers}
        available_questions = [q for q in stage_questions if q["_id"] not in answered_ids]
        
        if not available_questions:
            return None
            
        # Use LLM to select most relevant question
        context = self._build_question_context(session, previous_answers)
        selected_question = self._select_next_question(available_questions, context)
        
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
        # Extract structured data from answer using LLM
        structured_data = self._extract_answer_data(question, answer_text)
        
        # Create answer object
        answer = Answer(
            question_id=question.id,
            text=answer_text,
            timestamp=datetime.now(),
            structured_data=structured_data,
            confidence_score=self._calculate_answer_confidence(answer_text)
        )
        
        # Update session progress
        total_questions = len(self.db.get_questions_for_stage(session.business_stage))
        session.questions_answers.append({"question_id": question.id, "answer": answer})
        session.completion_status = len(session.questions_answers) / total_questions
        
        return answer

    def generate_assessment_result(
        self,
        session: AssessmentSession
    ) -> AssessmentResult:
        """Generate final assessment results and recommendations."""
        # Analyze answers and calculate scores
        scores = self._calculate_category_scores(session.questions_answers)
        
        # Generate recommendations using LLM
        recommendations = self._generate_recommendations(
            session.business_stage,
            session.industry,
            scores,
            session.questions_answers
        )
        
        # Identify opportunities and risks
        opportunities = self._identify_opportunities(session.questions_answers)
        risk_factors = self._identify_risks(session.questions_answers)
        
        return AssessmentResult(
            business_name=session.business_name,
            completion_date=datetime.now(),
            scores=scores,
            recommendations=recommendations,
            opportunities=opportunities,
            risk_factors=risk_factors
        )

    def _build_question_context(
        self,
        session: AssessmentSession,
        previous_answers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build context for question selection."""
        return {
            "business_stage": session.business_stage,
            "industry": session.industry,
            "previous_answers": previous_answers,
            "completion_status": session.completion_status
        }

    def _select_next_question(
        self,
        available_questions: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Select the most relevant next question using LLM."""
        prompt = f"""
        Given the following context and available questions, select the most relevant question to ask next:
        
        Context:
        - Business Stage: {context['business_stage']}
        - Industry: {context['industry']}
        - Progress: {context['completion_status']:.0%} complete
        
        Previous answers summary:
        {self._summarize_previous_answers(context['previous_answers'])}
        
        Available questions:
        {self._format_questions_for_prompt(available_questions)}
        
        Return the ID of the most relevant question to ask next.
        """
        
        response = self.llm.generate_response(prompt)
        question_id = response.strip()
        
        return next(q for q in available_questions if str(q["_id"]) == question_id)

    def _extract_answer_data(
        self,
        question: Question,
        answer_text: str
    ) -> Dict[str, Any]:
        """Extract structured data from answer text using LLM."""
        prompt = f"""
        Extract key information from the following answer to: "{question.text}"
        
        Answer: {answer_text}
        
        Return a JSON object with the following structure:
        {{
            "key_points": ["point1", "point2", ...],
            "sentiment": "positive/negative/neutral",
            "confidence": 0.0-1.0,
            "category_specific_data": {{}}
        }}
        """
        
        response = self.llm.generate_response(prompt)
        return eval(response)  # Note: In production, use proper JSON parsing

    def _calculate_answer_confidence(self, answer_text: str) -> float:
        """Calculate confidence score for an answer."""
        # Simple heuristic based on answer length and detail
        words = answer_text.split()
        base_score = min(len(words) / 100.0, 0.8)  # Cap at 0.8
        
        # Add bonus for specific details
        detail_indicators = ['because', 'specifically', 'for example', 'such as']
        detail_bonus = sum(0.05 for indicator in detail_indicators if indicator in answer_text.lower())
        
        return min(base_score + detail_bonus, 1.0)

    def _calculate_category_scores(
        self,
        questions_answers: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate scores for each assessment category."""
        category_scores = {area: 0.0 for area in FUNCTIONAL_AREAS}
        category_counts = {area: 0 for area in FUNCTIONAL_AREAS}
        
        for qa in questions_answers:
            answer = qa["answer"]
            question = self.db.get_question_by_id(qa["question_id"])
            
            if question["category"] in category_scores:
                score = answer.confidence_score * self._calculate_answer_quality(answer)
                category_scores[question["category"]] += score
                category_counts[question["category"]] += 1
        
        # Calculate averages
        for category in category_scores:
            if category_counts[category] > 0:
                category_scores[category] /= category_counts[category]
        
        return category_scores

    def _calculate_answer_quality(self, answer: Answer) -> float:
        """Calculate the quality score of an answer."""
        # Base score from structured data
        if not answer.structured_data:
            return 0.5
            
        base_score = 0.6
        
        # Add points for comprehensive responses
        if len(answer.structured_data.get("key_points", [])) >= 3:
            base_score += 0.2
            
        # Add points for positive sentiment
        if answer.structured_data.get("sentiment") == "positive":
            base_score += 0.1
            
        # Add points for high confidence
        if answer.structured_data.get("confidence", 0) > 0.8:
            base_score += 0.1
            
        return min(base_score, 1.0)

    def _generate_recommendations(
        self,
        business_stage: str,
        industry: str,
        scores: Dict[str, float],
        questions_answers: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on assessment results."""
        context = f"""
        Business Stage: {business_stage}
        Industry: {industry}
        
        Category Scores:
        {self._format_scores_for_prompt(scores)}
        
        Key Findings:
        {self._summarize_answers(questions_answers)}
        
        Generate 3-5 specific, actionable recommendations based on the assessment results.
        """
        
        response = self.llm.generate_response(context)
        return [rec.strip() for rec in response.split('\n') if rec.strip()]

    def _identify_opportunities(
        self,
        questions_answers: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify business opportunities from assessment answers."""
        context = self._summarize_answers(questions_answers)
        
        prompt = f"""
        Based on the following assessment summary, identify 3-4 key business opportunities:
        
        {context}
        
        Return each opportunity on a new line.
        """
        
        response = self.llm.generate_response(prompt)
        return [opp.strip() for opp in response.split('\n') if opp.strip()]

    def _identify_risks(
        self,
        questions_answers: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify potential risks from assessment answers."""
        context = self._summarize_answers(questions_answers)
        
        prompt = f"""
        Based on the following assessment summary, identify 3-4 key business risks:
        
        {context}
        
        Return each risk on a new line.
        """
        
        response = self.llm.generate_response(prompt)
        return [risk.strip() for risk in response.split('\n') if risk.strip()]

    def _summarize_previous_answers(
        self,
        previous_answers: List[Dict[str, Any]]
    ) -> str:
        """Create a summary of previous answers for context."""
        if not previous_answers:
            return "No previous answers."
            
        summary = []
        for qa in previous_answers[-3:]:  # Only use last 3 answers for context
            question = self.db.get_question_by_id(qa["question_id"])
            answer = qa["answer"]
            summary.append(f"Q: {question['text']}\nA: {answer.text[:100]}...")
            
        return "\n\n".join(summary)

    def _format_questions_for_prompt(
        self,
        questions: List[Dict[str, Any]]
    ) -> str:
        """Format questions for LLM prompt."""
        return "\n".join(
            f"ID: {q['_id']}\nCategory: {q['category']}\nQuestion: {q['text']}"
            for q in questions
        )

    def _format_scores_for_prompt(
        self,
        scores: Dict[str, float]
    ) -> str:
        """Format category scores for LLM prompt."""
        return "\n".join(
            f"{category}: {score:.1%}"
            for category, score in scores.items()
        )

    def _summarize_answers(
        self,
        questions_answers: List[Dict[str, Any]]
    ) -> str:
        """Create a summary of all answers for analysis."""
        summary = []
        for qa in questions_answers:
            question = self.db.get_question_by_id(qa["question_id"])
            answer = qa["answer"]
            key_points = answer.structured_data.get("key_points", [])
            
            summary.append(
                f"Topic: {question['category']}\n"
                f"Key Points: {', '.join(key_points)}\n"
                f"Sentiment: {answer.structured_data.get('sentiment', 'neutral')}"
            )
            
        return "\n\n".join(summary)
