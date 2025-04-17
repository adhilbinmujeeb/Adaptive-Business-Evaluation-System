from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from models.business_profile import BusinessProfile

@dataclass
class Question:
    id: str
    text: str
    phase: str
    business_type: Optional[str] = None
    business_stage: Optional[str] = None
    category: Optional[str] = None
    follow_up_questions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "phase": self.phase,
            "business_type": self.business_type,
            "business_stage": self.business_stage,
            "category": self.category,
            "follow_up_questions": self.follow_up_questions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Question':
        return cls(
            id=data.get("id", ""),
            text=data.get("text", ""),
            phase=data.get("phase", ""),
            business_type=data.get("business_type"),
            business_stage=data.get("business_stage"),
            category=data.get("category"),
            follow_up_questions=data.get("follow_up_questions", [])
        )

@dataclass
class Answer:
    question: Question
    text: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question.to_dict(),
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Answer':
        return cls(
            question=Question.from_dict(data.get("question", {})),
            text=data.get("text", ""),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            metadata=data.get("metadata", {})
        )

@dataclass
class AssessmentSession:
    business_profile: BusinessProfile
    current_phase: str
    questions_asked: List[Question] = field(default_factory=list)
    answers_received: List[Answer] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    completed: bool = False
    start_time: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "business_profile": self.business_profile.to_dict(),
            "current_phase": self.current_phase,
            "questions_asked": [q.to_dict() for q in self.questions_asked],
            "answers_received": [a.to_dict() for a in self.answers_received],
            "red_flags": self.red_flags,
            "opportunities": self.opportunities,
            "completed": self.completed,
            "start_time": self.start_time.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AssessmentSession':
        return cls(
            business_profile=BusinessProfile.from_dict(data.get("business_profile", {})),
            current_phase=data.get("current_phase", ""),
            questions_asked=[Question.from_dict(q) for q in data.get("questions_asked", [])],
            answers_received=[Answer.from_dict(a) for a in data.get("answers_received", [])],
            red_flags=data.get("red_flags", []),
            opportunities=data.get("opportunities", []),
            completed=data.get("completed", False),
            start_time=datetime.fromisoformat(data.get("start_time", datetime.now().isoformat()))
        )

@dataclass
class AssessmentResult:
    business_profile: BusinessProfile
    scores: Dict[str, float]
    recommendations: List[str]
    risks: List[str]
    opportunities: List[str]
    key_findings: List[str]
    completion_time: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "business_profile": self.business_profile.to_dict(),
            "scores": self.scores,
            "recommendations": self.recommendations,
            "risks": self.risks,
            "opportunities": self.opportunities,
            "key_findings": self.key_findings,
            "completion_time": self.completion_time.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AssessmentResult':
        return cls(
            business_profile=BusinessProfile.from_dict(data.get("business_profile", {})),
            scores=data.get("scores", {}),
            recommendations=data.get("recommendations", []),
            risks=data.get("risks", []),
            opportunities=data.get("opportunities", []),
            key_findings=data.get("key_findings", []),
            completion_time=datetime.fromisoformat(data.get("completion_time", datetime.now().isoformat()))
        )
