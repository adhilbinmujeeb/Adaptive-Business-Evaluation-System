from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class Question:
    id: str
    text: str
    category: str
    subcategory: str
    business_stage: str
    weight: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

@dataclass
class Answer:
    question_id: str
    response: str
    timestamp: datetime
    confidence_score: float = 1.0
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "response": self.response,
            "timestamp": self.timestamp.isoformat(),
            "confidence_score": self.confidence_score,
            "extracted_data": self.extracted_data
        }

@dataclass
class AssessmentSession:
    business_id: str
    business_stage: str
    industry: str
    start_time: datetime
    questions_answers: List[Dict[str, Any]] = field(default_factory=list)
    completion_status: float = 0.0
    current_question_index: int = 0
    end_time: Optional[datetime] = None
    
    def add_answer(self, question: Question, answer: Answer) -> None:
        self.questions_answers.append({
            "question": question.to_dict(),
            "answer": answer.to_dict()
        })
        self.completion_status = len(self.questions_answers) / 25  # Assuming 25 total questions
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "business_id": self.business_id,
            "business_stage": self.business_stage,
            "industry": self.industry,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "questions_answers": self.questions_answers,
            "completion_status": self.completion_status,
            "current_question_index": self.current_question_index
        }

@dataclass
class AssessmentResult:
    session_id: str
    business_profile: Dict[str, Any]
    scores: Dict[str, float]
    recommendations: List[str]
    risk_factors: List[str]
    opportunities: List[str]
    completion_time: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "business_profile": self.business_profile,
            "scores": self.scores,
            "recommendations": self.recommendations,
            "risk_factors": self.risk_factors,
            "opportunities": self.opportunities,
            "completion_time": self.completion_time.isoformat()
        }
