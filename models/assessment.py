from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Question:
    """Represents a business assessment question."""
    id: str
    text: str
    category: str
    follow_up_questions: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "category": self.category,
            "follow_up_questions": self.follow_up_questions or []
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Question':
        return cls(
            id=data["id"],
            text=data["text"],
            category=data["category"],
            follow_up_questions=data.get("follow_up_questions", [])
        )

@dataclass
class Answer:
    """Represents an answer to an assessment question."""
    question_id: str
    text: str
    timestamp: datetime
    structured_data: Dict[str, Any]
    confidence_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "structured_data": self.structured_data,
            "confidence_score": self.confidence_score
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Answer':
        return cls(
            question_id=data["question_id"],
            text=data["text"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            structured_data=data["structured_data"],
            confidence_score=data["confidence_score"]
        )

@dataclass
class AssessmentSession:
    """Represents an ongoing assessment session."""
    business_name: str
    business_stage: str
    industry: str
    start_time: datetime
    questions_answers: List[Dict[str, Any]]
    completion_status: float
    current_question: Optional[Question] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "business_name": self.business_name,
            "business_stage": self.business_stage,
            "industry": self.industry,
            "start_time": self.start_time.isoformat(),
            "questions_answers": [
                {
                    "question_id": qa["question_id"],
                    "answer": qa["answer"].to_dict() if isinstance(qa["answer"], Answer) else qa["answer"]
                }
                for qa in self.questions_answers
            ],
            "completion_status": self.completion_status,
            "current_question": self.current_question.to_dict() if self.current_question else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AssessmentSession':
        return cls(
            business_name=data["business_name"],
            business_stage=data["business_stage"],
            industry=data["industry"],
            start_time=datetime.fromisoformat(data["start_time"]),
            questions_answers=[
                {
                    "question_id": qa["question_id"],
                    "answer": Answer.from_dict(qa["answer"]) if isinstance(qa["answer"], dict) else qa["answer"]
                }
                for qa in data["questions_answers"]
            ],
            completion_status=data["completion_status"],
            current_question=Question.from_dict(data["current_question"]) if data.get("current_question") else None
        )

@dataclass
class AssessmentResult:
    """Represents the final result of a business assessment."""
    business_name: str
    completion_date: datetime
    scores: Dict[str, float]
    recommendations: List[str]
    opportunities: List[str]
    risk_factors: List[str]
    detailed_analysis: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "business_name": self.business_name,
            "completion_date": self.completion_date.isoformat(),
            "scores": self.scores,
            "recommendations": self.recommendations,
            "opportunities": self.opportunities,
            "risk_factors": self.risk_factors,
            "detailed_analysis": self.detailed_analysis
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AssessmentResult':
        return cls(
            business_name=data["business_name"],
            completion_date=datetime.fromisoformat(data["completion_date"]),
            scores=data["scores"],
            recommendations=data["recommendations"],
            opportunities=data["opportunities"],
            risk_factors=data["risk_factors"],
            detailed_analysis=data.get("detailed_analysis")
        )

    def get_overall_score(self) -> float:
        """Calculate the overall assessment score."""
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

    def get_primary_recommendation(self) -> str:
        """Get the most important recommendation."""
        return self.recommendations[0] if self.recommendations else "No recommendations available."

    def get_critical_risks(self) -> List[str]:
        """Get the most critical risk factors."""
        return self.risk_factors[:3] if self.risk_factors else []

    def get_top_opportunities(self) -> List[str]:
        """Get the top opportunities."""
        return self.opportunities[:3] if self.opportunities else []

    def get_category_score(self, category: str) -> float:
        """Get the score for a specific category."""
        return self.scores.get(category, 0.0)

    def is_high_potential(self) -> bool:
        """Determine if the business has high potential."""
        return self.get_overall_score() >= 0.75

    def needs_immediate_attention(self) -> bool:
        """Determine if the business needs immediate attention."""
        return self.get_overall_score() < 0.4

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the assessment results."""
        return {
            "overall_score": self.get_overall_score(),
            "primary_recommendation": self.get_primary_recommendation(),
            "critical_risks": self.get_critical_risks(),
            "top_opportunities": self.get_top_opportunities(),
            "high_potential": self.is_high_potential(),
            "needs_attention": self.needs_immediate_attention()
        }
