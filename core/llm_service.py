from typing import Optional, List, Dict, Any
import os
from groq import Groq
from groq.types import ChatCompletion
import backoff
from datetime import datetime

class LLMService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Get API key from environment variable
        self.api_key = os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
            
        # Initialize Groq client
        try:
            self.client = Groq(
                api_key=self.api_key,
                timeout=60.0  # Set timeout to 60 seconds
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Groq client: {str(e)}")
            
        self.model = "llama2-70b-4096"  # Default model
        self._initialized = True
        
    @backoff.on_exception(
        backoff.expo,
        (Exception),
        max_tries=3,
        max_time=30
    )
    def generate_response(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Generate a response using the Groq API with retry logic."""
        try:
            chat_completion: ChatCompletion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful business analysis assistant. Provide clear, concise, and accurate responses."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return chat_completion.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            raise
            
    def analyze_business_profile(
        self,
        profile_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze a business profile and extract key insights."""
        prompt = f"""
        Analyze the following business profile and provide key insights:
        
        Business Name: {profile_data.get('business_name')}
        Industry: {profile_data.get('industry')}
        Stage: {profile_data.get('business_stage')}
        Revenue: ${profile_data.get('revenue', 0):,.2f}
        Growth Rate: {profile_data.get('growth_rate', 'N/A')}
        
        Provide analysis in the following JSON format:
        {{
            "strengths": ["strength1", "strength2", ...],
            "weaknesses": ["weakness1", "weakness2", ...],
            "opportunities": ["opportunity1", "opportunity2", ...],
            "risks": ["risk1", "risk2", ...],
            "recommendations": ["recommendation1", "recommendation2", ...]
        }}
        """
        
        response = self.generate_response(prompt)
        try:
            return eval(response)  # In production, use proper JSON parsing
        except Exception as e:
            print(f"Error parsing analysis response: {str(e)}")
            return {
                "strengths": [],
                "weaknesses": [],
                "opportunities": [],
                "risks": [],
                "recommendations": []
            }
            
    def generate_follow_up_questions(
        self,
        context: Dict[str, Any],
        previous_answers: List[Dict[str, Any]],
        max_questions: int = 3
    ) -> List[str]:
        """Generate relevant follow-up questions based on context and previous answers."""
        prompt = f"""
        Given the following context and previous answers, generate {max_questions} relevant follow-up questions:
        
        Context:
        - Industry: {context.get('industry')}
        - Business Stage: {context.get('business_stage')}
        - Current Focus: {context.get('current_focus')}
        
        Previous Answers:
        {self._format_previous_answers(previous_answers)}
        
        Generate {max_questions} specific, probing questions that will help gather more insights.
        Return each question on a new line.
        """
        
        response = self.generate_response(prompt)
        return [q.strip() for q in response.split('\n') if q.strip()]
        
    def validate_response(
        self,
        response: str,
        criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate a response against given criteria."""
        prompt = f"""
        Validate the following response against the given criteria:
        
        Response:
        {response}
        
        Criteria:
        {self._format_criteria(criteria)}
        
        Return validation results in the following JSON format:
        {{
            "is_valid": true/false,
            "missing_elements": ["element1", "element2", ...],
            "suggestions": ["suggestion1", "suggestion2", ...],
            "confidence_score": 0.0-1.0
        }}
        """
        
        validation_response = self.generate_response(prompt)
        try:
            return eval(validation_response)  # In production, use proper JSON parsing
        except Exception as e:
            print(f"Error parsing validation response: {str(e)}")
            return {
                "is_valid": False,
                "missing_elements": [],
                "suggestions": ["Error processing validation"],
                "confidence_score": 0.0
            }
            
    def _format_previous_answers(
        self,
        previous_answers: List[Dict[str, Any]]
    ) -> str:
        """Format previous answers for prompt context."""
        formatted = []
        for i, answer in enumerate(previous_answers[-3:], 1):  # Only use last 3 answers
            formatted.append(f"Q{i}: {answer.get('question', 'N/A')}")
            formatted.append(f"A{i}: {answer.get('answer', 'N/A')}\n")
        return "\n".join(formatted)
        
    def _format_criteria(self, criteria: Dict[str, Any]) -> str:
        """Format validation criteria for prompt."""
        formatted = []
        for key, value in criteria.items():
            formatted.append(f"- {key}: {value}")
        return "\n".join(formatted)
