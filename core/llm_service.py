from typing import List, Dict, Any, Optional
from groq import Groq
from .config import GROQ_API_KEY, DEFAULT_MODEL

class LLMService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'client'):
            self.client = Groq(api_key=GROQ_API_KEY)
            self.default_model = DEFAULT_MODEL

    def generate_response(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate a response using the Groq API."""
        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            completion = self.client.chat.completions.create(
                messages=messages,
                model=model or self.default_model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            return f"Error: {str(e)}"

    def analyze_business_profile(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a business profile and provide insights."""
        prompt = f"""
        Analyze the following business profile and provide structured insights:

        Business Profile:
        {profile_data}

        Provide analysis in the following areas:
        1. Key Strengths
        2. Risk Factors
        3. Growth Opportunities
        4. Strategic Recommendations
        5. Valuation Considerations

        Format the response as a JSON object with these categories as keys.
        """

        try:
            response = self.generate_response(
                prompt,
                system_message="You are an expert business analyst providing detailed insights.",
                temperature=0.5
            )
            
            # Basic cleaning of response to ensure it's JSON-compatible
            response = response.replace("```json", "").replace("```", "").strip()
            
            # You might want to add more robust JSON parsing here
            import json
            return json.loads(response)
        except Exception as e:
            print(f"Error analyzing business profile: {e}")
            return {}

    def generate_follow_up_questions(
        self,
        conversation_history: List[Dict[str, str]],
        business_stage: str,
        industry: str
    ) -> str:
        """Generate relevant follow-up questions based on conversation history."""
        prompt = f"""
        Based on the following conversation history for a {business_stage} business in the {industry} industry,
        generate the most relevant follow-up question.

        Conversation History:
        {conversation_history}

        Generate a single, specific question that will help gather critical information
        about the business, considering its stage and industry context.
        """

        return self.generate_response(
            prompt,
            system_message="You are an expert business interviewer conducting due diligence.",
            temperature=0.7
        )

    def validate_response(self, response: str, expected_info: List[str]) -> Dict[str, Any]:
        """Validate and extract specific information from a response."""
        prompt = f"""
        Analyze the following response and extract/validate information about: {', '.join(expected_info)}

        Response: {response}

        For each item, provide:
        1. Whether the information is present (true/false)
        2. The extracted information if present
        3. Confidence score (0-1)
        4. Any inconsistencies or red flags

        Format the response as a JSON object.
        """

        try:
            return json.loads(self.generate_response(
                prompt,
                temperature=0.3
            ))
        except Exception as e:
            print(f"Error validating response: {e}")
            return {}
