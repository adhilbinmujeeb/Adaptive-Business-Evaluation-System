from typing import Optional, List, Dict, Any
import os
import json
import time
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
            
        self.api_key = os.getenv('GROQ_API_KEY')
        self.use_mock = not self.api_key  # Use mock mode if no API key is available
        
        if not self.use_mock:
            try:
                from groq import Groq
                self.client = Groq(
                    api_key=self.api_key,
                    timeout=60.0
                )
                self.model = "llama2-70b-4096"
            except Exception as e:
                print(f"Warning: Failed to initialize Groq client ({str(e)}). Falling back to mock mode.")
                self.use_mock = True
        
        self._initialized = True



    def _retry_with_exponential_backoff(self, func, max_retries=3, initial_delay=1):
        """Helper function to implement retry logic with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise e
                
                delay = initial_delay * (2 ** attempt)
                print(f"Attempt {attempt + 1} failed, retrying in {delay} seconds...")
                time.sleep(delay)
        
    def generate_response(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Generate a response using either Groq API or mock responses."""
        if self.use_mock:
            return self._generate_mock_response(prompt)
            
        def _generate():
            try:
                completion = self.client.chat.completions.create(
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
                
                return completion.choices[0].message.content.strip()
                
            except Exception as e:
                print(f"Error generating response: {str(e)}")
                return self._generate_mock_response(prompt)
        
        return self._retry_with_exponential_backoff(_generate)
    
    def _generate_mock_response(self, prompt: str) -> str:
        """Generate mock responses for development/testing."""
        if "business profile" in prompt.lower():
            return json.dumps({
                "strengths": [
                    "Strong market position",
                    "Healthy revenue growth",
                    "Experienced management team"
                ],
                "weaknesses": [
                    "Limited international presence",
                    "High customer acquisition costs"
                ],
                "opportunities": [
                    "Market expansion potential",
                    "New product development",
                    "Strategic partnerships"
                ],
                "risks": [
                    "Increasing competition",
                    "Regulatory changes",
                    "Market volatility"
                ],
                "recommendations": [
                    "Focus on core market expansion",
                    "Invest in technology infrastructure",
                    "Develop strategic partnerships"
                ]
            })
        elif "follow-up questions" in prompt.lower():
            return "\n".join([
                "What are your main competitive advantages?",
                "How do you plan to scale operations?",
                "What are your key growth metrics?"
            ])
        elif "validate" in prompt.lower():
            return json.dumps({
                "is_valid": True,
                "missing_elements": [],
                "suggestions": ["Consider adding more quantitative metrics"],
                "confidence_score": 0.85
            })
        else:
            return "This is a mock response for development purposes."
            
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
            if isinstance(response, str):
                return json.loads(response)
            return response
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
            if isinstance(validation_response, str):
                return json.loads(validation_response)
            return validation_response
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
