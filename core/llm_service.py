from typing import Dict, Any, List, Optional
import os
from groq import Groq
from models.assessment import Question, Answer
from models.business_profile import BusinessProfile

class LLMService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.client = Groq(
                api_key=os.getenv('GROQ_API_KEY', 'default_key')
            )
            self._initialized = True

    def select_next_question(
        self,
        available_questions: List[Question],
        context: Dict[str, Any]
    ) -> Optional[Question]:
        """Select the most relevant next question based on context."""
        try:
            if not available_questions:
                return None
                
            # Prepare prompt for question selection
            prompt = self._create_question_selection_prompt(available_questions, context)
            
            # Get response from Groq
            response = self.client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You are an expert business analyst helping to select the most relevant question for a business assessment."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            # Parse response to get question ID
            selected_id = response.choices[0].message.content.strip()
            
            # Find and return the selected question
            for question in available_questions:
                if question.id == selected_id:
                    return question
                    
            return available_questions[0]  # Fallback to first question if parsing fails
            
        except Exception as e:
            print(f"Error selecting next question: {str(e)}")
            return available_questions[0] if available_questions else None

    def analyze_answer(
        self,
        answer: Answer,
        business_profile: BusinessProfile,
        current_phase: str
    ) -> Dict[str, Any]:
        """Analyze an answer for insights, red flags, and opportunities."""
        try:
            # Prepare prompt for answer analysis
            prompt = self._create_answer_analysis_prompt(answer, business_profile, current_phase)
            
            # Get response from Groq
            response = self.client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You are an expert business analyst analyzing responses from a business assessment."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=500
            )
            
            # Parse and structure the response
            analysis = self._parse_analysis_response(response.choices[0].message.content)
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing answer: {str(e)}")
            return {
                "red_flags": [],
                "opportunities": [],
                "insights": [],
                "confidence_score": 0.5
            }

    def generate_assessment_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive assessment summary."""
        try:
            # Prepare prompt for summary generation
            prompt = self._create_summary_prompt(context)
            
            # Get response from Groq
            response = self.client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You are an expert business analyst generating a comprehensive business assessment summary."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=1000
            )
            
            # Parse and structure the response
            summary = self._parse_summary_response(response.choices[0].message.content)
            
            return summary
            
        except Exception as e:
            print(f"Error generating assessment summary: {str(e)}")
            return {
                "scores": {},
                "recommendations": [],
                "risks": [],
                "opportunities": [],
                "key_findings": []
            }

    def _create_question_selection_prompt(
        self,
        available_questions: List[Question],
        context: Dict[str, Any]
    ) -> str:
        """Create prompt for question selection."""
        prompt_parts = [
            "Based on the following context and available questions, select the most relevant question ID.",
            "\nContext:",
            f"Business Stage: {context['business_profile'].get('business_stage', 'Unknown')}",
            f"Current Phase: {context['current_phase']}",
            f"Previous Answers: {len(context['previous_answers'])}",
            "\nAvailable Questions:"
        ]
        
        for q in available_questions:
            prompt_parts.append(f"ID: {q.id} - {q.text}")
            
        prompt_parts.append("\nReturn only the question ID of the most relevant question.")
        
        return "\n".join(prompt_parts)

    def _create_answer_analysis_prompt(
        self,
        answer: Answer,
        business_profile: BusinessProfile,
        current_phase: str
    ) -> str:
        """Create prompt for answer analysis."""
        return f"""
        Analyze the following business assessment response:
        
        Question: {answer.question.text}
        Answer: {answer.text}
        
        Business Context:
        - Stage: {business_profile.business_stage}
        - Industry: {business_profile.industry}
        - Current Phase: {current_phase}
        
        Provide analysis in the following format:
        - Red Flags: (list any concerning aspects)
        - Opportunities: (list potential opportunities)
        - Insights: (list key insights)
        - Confidence Score: (0.0-1.0)
        """

    def _create_summary_prompt(self, context: Dict[str, Any]) -> str:
        """Create prompt for assessment summary."""
        return f"""
        Generate a comprehensive business assessment summary based on the following:
        
        Business Profile:
        {context['business_profile']}
        
        Assessment Responses:
        {context['answers']}
        
        Identified Red Flags:
        {context['red_flags']}
        
        Identified Opportunities:
        {context['opportunities']}
        
        Provide summary in the following format:
        - Scores: (category scores from 0-100)
        - Recommendations: (prioritized list)
        - Risks: (key risks identified)
        - Opportunities: (key opportunities identified)
        - Key Findings: (main insights)
        """

    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse and structure the analysis response."""
        try:
            lines = response.strip().split('\n')
            result = {
                "red_flags": [],
                "opportunities": [],
                "insights": [],
                "confidence_score": 0.5
            }
            
            current_section = None
            for line in lines:
                line = line.strip()
                if line.startswith('- Red Flags:'):
                    current_section = "red_flags"
                elif line.startswith('- Opportunities:'):
                    current_section = "opportunities"
                elif line.startswith('- Insights:'):
                    current_section = "insights"
                elif line.startswith('- Confidence Score:'):
                    try:
                        score = float(line.split(':')[1].strip())
                        result["confidence_score"] = min(1.0, max(0.0, score))
                    except:
                        pass
                elif line.startswith('- ') and current_section:
                    result[current_section].append(line[2:])
                    
            return result
            
        except Exception as e:
            print(f"Error parsing analysis response: {str(e)}")
            return {
                "red_flags": [],
                "opportunities": [],
                "insights": [],
                "confidence_score": 0.5
            }

    def _parse_summary_response(self, response: str) -> Dict[str, Any]:
        """Parse and structure the summary response."""
        try:
            lines = response.strip().split('\n')
            result = {
                "scores": {},
                "recommendations": [],
                "risks": [],
                "opportunities": [],
                "key_findings": []
            }
            
            current_section = None
            for line in lines:
                line = line.strip()
                if line.startswith('- Scores:'):
                    current_section = "scores"
                elif line.startswith('- Recommendations:'):
                    current_section = "recommendations"
                elif line.startswith('- Risks:'):
                    current_section = "risks"
                elif line.startswith('- Opportunities:'):
                    current_section = "opportunities"
                elif line.startswith('- Key Findings:'):
                    current_section = "key_findings"
                elif line.startswith('- ') and current_section:
                    if current_section == "scores":
                        try:
                            category, score = line[2:].split(':')
                            result["scores"][category.strip()] = float(score.strip())
                        except:
                            pass
                    else:
                        result[current_section].append(line[2:])
                    
            return result
            
        except Exception as e:
            print(f"Error parsing summary response: {str(e)}")
            return {
                "scores": {},
                "recommendations": [],
                "risks": [],
                "opportunities": [],
                "key_findings": []
            }
