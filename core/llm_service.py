from groq import Groq, APIError, AuthenticationError
import streamlit as st
import json
import re

class LLMService:
    def __init__(self, api_key, default_provider="groq", default_model=None):
        self.api_key = api_key
        self.provider = default_provider
        self.client = self._initialize_client()
        self.default_model = default_model or self.get_default_model()

    def _initialize_client(self):
        try:
            if self.provider == "groq":
                client = Groq(api_key=self.api_key)
                return client
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        except AuthenticationError:
            st.error("Groq API authentication failed. Check your API key.")
            st.stop()
        except Exception as e:
            st.error(f"Failed to initialize Groq client: {e}")
            st.stop()

    def get_default_model(self):
        if self.provider == "groq":
            return "llama3-8b-8192"
        return None

    def select_optimal_model(self, task_type="generic"):
        model_capabilities = {
            "llama3-8b-8192": {"complexity": 0.6, "quality": 0.7, "speed": 0.95, "tokens": 8192, "cost": 0.2},
            "llama3-70b-8192": {"complexity": 0.9, "quality": 0.9, "speed": 0.6, "tokens": 8192, "cost": 0.8},
            "mixtral-8x7b-32768": {"complexity": 0.8, "quality": 0.85, "speed": 0.8, "tokens": 32768, "cost": 0.5},
            "gemma-7b-it": {"complexity": 0.5, "quality": 0.65, "speed": 0.9, "tokens": 8192, "cost": 0.15},
        }
        task_weights = {
            "valuation_explanation": {"complexity": 0.7, "quality": 0.8, "speed": 0.5},
            "question_generation": {"complexity": 0.6, "quality": 0.7, "speed": 0.7},
            "assessment_report": {"complexity": 0.9, "quality": 0.9, "speed": 0.2},
            "profile_extraction": {"complexity": 0.8, "quality": 0.8, "speed": 0.4, "tokens": 0.6},
            "generic": {"complexity": 0.7, "quality": 0.7, "speed": 0.6},
        }
        weights = task_weights.get(task_type, task_weights["generic"])
        provider_models = {m: c for m, c in model_capabilities.items() if self.is_model_available(m)}
        
        if not provider_models:
            return self.default_model
        
        model_scores = {}
        for model, capabilities in provider_models.items():
            score = sum(capabilities.get(attr, 0) * weight for attr, weight in weights.items())
            total_weight = sum(weights.values())
            model_scores[model] = score / total_weight if total_weight > 0 else 0
        
        return max(model_scores, key=model_scores.get, default=self.default_model)

    def is_model_available(self, model_name):
        if self.provider == "groq":
            return not model_name.startswith(("gpt-", "claude-"))
        return False

    def enhanced_groq_qna(self, query, business_context=None, system_prompt=None, task_type="generic"):
        selected_model = self.select_optimal_model(task_type)
        base_system_prompt = system_prompt or "You are an expert business analyst. Be concise, analytical, and professional."
        messages = [{"role": "system", "content": base_system_prompt}]
        
        user_content = f"Query: {query}"
        if business_context:
            context_str = json.dumps(business_context, indent=2, default=str) if isinstance(business_context, dict) else str(business_context)
            user_content = f"Business Context:\n```json\n{context_str}\n```n\nQuery: {query}"
        messages.append({"role": "user", "content": user_content})

        try:
            completion = self.client.chat.completions.create(
                messages=messages,
                model=selected_model,
                temperature=0.5,
            )
            return completion.choices[0].message.content
        except (AuthenticationError, APIError) as e:
            st.error(f"Groq API error: {e}")
            return f"Error: Groq API failed ({e})"
        except Exception as e:
            st.error(f"Unexpected error in Groq API: {e}")
            return "Error: Unexpected API failure"

    def extract_structured_data(self, llm_response, expected_schema, max_retries=2):
        json_pattern = r'```json\s*([\s\S]*?)\s*```'
        json_matches = re.findall(json_pattern, llm_response)
        
        for json_str in json_matches:
            try:
                data = json.loads(json_str)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                continue
        
        if max_retries > 0:
            retry_prompt = f"""
            Provide the information as a valid JSON object matching this schema. Return only the raw JSON.
            Expected Schema:
            {json.dumps(expected_schema, indent=2)}
            """
            retry_response = self.enhanced_groq_qna(retry_prompt, task_type="simple_qa")
            return self.extract_structured_data(retry_response, expected_schema, max_retries - 1)
        
        st.warning("Failed to extract structured data.")
        return None
