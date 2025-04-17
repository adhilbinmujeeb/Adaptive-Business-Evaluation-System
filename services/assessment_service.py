from typing import Dict, Any, Optional, List
from core.database import DatabaseConnection
from core.llm_service import LLMService
from models.assessment import Question, Answer, AssessmentSession, AssessmentResult
from models.business_profile import BusinessProfile

class AssessmentService:
    def __init__(self, db: DatabaseConnection, llm_service: LLMService):
        self.db = db
        self.llm_service = llm_service
        self.current_sessions: Dict[str, AssessmentSession] = {}

    def start_assessment(self, business_profile: BusinessProfile) -> AssessmentSession:
        """Start a new assessment session."""
        try:
            session = AssessmentSession(
                business_profile=business_profile,
                current_phase="Initial Discovery",
                questions_asked=[],
                answers_received=[],
                completed=False
            )
            
            # Store session using business profile ID as key
            session_id = str(business_profile.id)
            self.current_sessions[session_id] = session
            
            return session
        except Exception as e:
            print(f"Error starting assessment: {str(e)}")
            raise

    def get_next_question(self, session_id: str) -> Optional[Question]:
        """Get the next question based on the current phase and previous answers."""
        try:
            session = self.current_sessions.get(session_id)
            if not session:
                raise ValueError("Invalid session ID")

            # Get questions for current phase from database
            questions = self.db.get_questions_for_stage(
                session.business_profile.business_stage
            )
            
            if not questions:
                return None

            # Use LLM to select most relevant question based on context
            context = {
                "business_profile": session.business_profile.to_dict(),
                "previous_answers": [a.to_dict() for a in session.answers_received],
                "current_phase": session.current_phase
            }
            
            next_question = self.llm_service.select_next_question(questions, context)
            
            if next_question:
                session.questions_asked.append(next_question)
                
            return next_question
        except Exception as e:
            print(f"Error getting next question: {str(e)}")
            return None

    def process_answer(self, session_id: str, answer: Answer) -> Optional[AssessmentResult]:
        """Process an answer and update the assessment session."""
        try:
            session = self.current_sessions.get(session_id)
            if not session:
                raise ValueError("Invalid session ID")

            # Add answer to session
            session.answers_received.append(answer)
            
            # Use LLM to analyze answer
            analysis = self.llm_service.analyze_answer(
                answer,
                session.business_profile,
                session.current_phase
            )
            
            # Check for red flags
            if analysis.get("red_flags"):
                session.red_flags.extend(analysis["red_flags"])
            
            # Check for opportunities
            if analysis.get("opportunities"):
                session.opportunities.extend(analysis["opportunities"])
            
            # Update phase if needed
            if self._should_advance_phase(session):
                self._advance_phase(session)
            
            # Check if assessment is complete
            if self._is_assessment_complete(session):
                return self._generate_assessment_result(session)
                
            return None
        except Exception as e:
            print(f"Error processing answer: {str(e)}")
            return None

    def _should_advance_phase(self, session: AssessmentSession) -> bool:
        """Determine if the assessment should advance to the next phase."""
        phases = [
            "Initial Discovery",
            "Business Model Deep Dive",
            "Market & Competition Analysis",
            "Financial Performance",
            "Team & Operations",
            "Investment & Growth Strategy"
        ]
        
        try:
            # Get current phase index
            current_index = phases.index(session.current_phase)
            
            # Check if we have enough information for current phase
            phase_questions = len([q for q in session.questions_asked 
                                 if q.phase == session.current_phase])
            phase_answers = len([a for a in session.answers_received 
                               if a.question.phase == session.current_phase])
            
            # Advance if we have at least 3 questions answered in current phase
            return phase_questions >= 3 and phase_answers >= 3
        except Exception as e:
            print(f"Error checking phase advancement: {str(e)}")
            return False

    def _advance_phase(self, session: AssessmentSession) -> None:
        """Advance the assessment to the next phase."""
        phases = [
            "Initial Discovery",
            "Business Model Deep Dive",
            "Market & Competition Analysis",
            "Financial Performance",
            "Team & Operations",
            "Investment & Growth Strategy"
        ]
        
        try:
            current_index = phases.index(session.current_phase)
            if current_index < len(phases) - 1:
                session.current_phase = phases[current_index + 1]
        except Exception as e:
            print(f"Error advancing phase: {str(e)}")

    def _is_assessment_complete(self, session: AssessmentSession) -> bool:
        """Check if the assessment is complete."""
        try:
            # Check if we're in the last phase
            if session.current_phase == "Investment & Growth Strategy":
                # Check if we have enough answers in the last phase
                last_phase_answers = len([a for a in session.answers_received 
                                        if a.question.phase == session.current_phase])
                return last_phase_answers >= 3
            return False
        except Exception as e:
            print(f"Error checking assessment completion: {str(e)}")
            return False

    def _generate_assessment_result(self, session: AssessmentSession) -> AssessmentResult:
        """Generate the final assessment result."""
        try:
            # Use LLM to analyze all answers and generate comprehensive assessment
            analysis = self.llm_service.generate_assessment_summary({
                "business_profile": session.business_profile.to_dict(),
                "answers": [a.to_dict() for a in session.answers_received],
                "red_flags": session.red_flags,
                "opportunities": session.opportunities
            })
            
            # Mark session as completed
            session.completed = True
            
            return AssessmentResult(
                business_profile=session.business_profile,
                scores=analysis.get("scores", {}),
                recommendations=analysis.get("recommendations", []),
                risks=analysis.get("risks", []),
                opportunities=analysis.get("opportunities", []),
                key_findings=analysis.get("key_findings", [])
            )
        except Exception as e:
            print(f"Error generating assessment result: {str(e)}")
            raise

    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get the current status of an assessment session."""
        try:
            session = self.current_sessions.get(session_id)
            if not session:
                raise ValueError("Invalid session ID")
                
            return {
                "current_phase": session.current_phase,
                "questions_asked": len(session.questions_asked),
                "answers_received": len(session.answers_received),
                "completed": session.completed,
                "red_flags": len(session.red_flags),
                "opportunities": len(session.opportunities)
            }
        except Exception as e:
            print(f"Error getting session status: {str(e)}")
            return {}
