import streamlit as st
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import pymongo
from groq import Groq

# Data Models
@dataclass
class FinancialMetrics:
    revenue: float = 0.0
    profit: float = 0.0
    ebitda: float = 0.0
    growth_rate: float = 0.0
    burn_rate: Optional[float] = None
    cash_balance: Optional[float] = None
    revenue_growth: Optional[float] = None
    ebitda_margin: Optional[float] = None
    profit_margin: Optional[float] = None
    free_cash_flow: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FinancialMetrics':
        return cls(**{k: v for k, v in data.items() if v is not None})

@dataclass
class MarketMetrics:
    total_market_size: float = 0.0
    market_share: float = 0.0
    competitor_count: int = 0
    market_growth_rate: float = 0.0
    target_market_size: Optional[float] = None
    market_penetration: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketMetrics':
        return cls(**{k: v for k, v in data.items() if v is not None})

@dataclass
class BusinessProfile:
    id: str
    name: str
    industry: str
    business_stage: str
    business_type: str
    description: str
    founded_date: datetime
    financial_metrics: FinancialMetrics = field(default_factory=FinancialMetrics)
    market_metrics: MarketMetrics = field(default_factory=MarketMetrics)
    competitive_advantages: List[str] = field(default_factory=list)
    key_risks: List[str] = field(default_factory=list)
    growth_opportunities: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "industry": self.industry,
            "business_stage": self.business_stage,
            "business_type": self.business_type,
            "description": self.description,
            "founded_date": self.founded_date.isoformat(),
            "financial_metrics": self.financial_metrics.to_dict(),
            "market_metrics": self.market_metrics.to_dict(),
            "competitive_advantages": self.competitive_advantages,
            "key_risks": self.key_risks,
            "growth_opportunities": self.growth_opportunities,
            "last_updated": self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BusinessProfile':
        metrics_data = data.get("financial_metrics", {})
        market_data = data.get("market_metrics", {})
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            industry=data.get("industry", ""),
            business_stage=data.get("business_stage", ""),
            business_type=data.get("business_type", ""),
            description=data.get("description", ""),
            founded_date=datetime.fromisoformat(data.get("founded_date", datetime.now().isoformat())),
            financial_metrics=FinancialMetrics.from_dict(metrics_data),
            market_metrics=MarketMetrics.from_dict(market_data),
            competitive_advantages=data.get("competitive_advantages", []),
            key_risks=data.get("key_risks", []),
            growth_opportunities=data.get("growth_opportunities", []),
            last_updated=datetime.fromisoformat(data.get("last_updated", datetime.now().isoformat()))
        )

class ValuationMethod(Enum):
    REVENUE_MULTIPLE = "revenue_multiple"
    EBITDA_MULTIPLE = "ebitda_multiple"
    DCF = "dcf"

@dataclass
class ValuationResult:
    method: ValuationMethod
    value: float
    confidence_score: float
    multiplier_used: Optional[float]
    assumptions: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "value": self.value,
            "confidence_score": self.confidence_score,
            "multiplier_used": self.multiplier_used,
            "assumptions": self.assumptions
        }

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
        return {k: v for k, v in self.__dict__.items() if v is not None}

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

@dataclass
class AssessmentSession:
    business_profile: BusinessProfile
    current_phase: str
    questions_asked: List[Question] = field(default_factory=list)
    answers_received: List[Answer] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    completed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "business_profile": self.business_profile.to_dict(),
            "current_phase": self.current_phase,
            "questions_asked": [q.to_dict() for q in self.questions_asked],
            "answers_received": [a.to_dict() for a in self.answers_received],
            "red_flags": self.red_flags,
            "opportunities": self.opportunities,
            "completed": self.completed
        }

# Services
class DatabaseConnection:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            try:
                uri = os.getenv('MONGODB_URI')
                if uri:
                    self.client = pymongo.MongoClient(uri)
                    self.db = self.client.get_database()
                    self.businesses = self.db.businesses
                    self.questions = self.db.questions
                else:
                    print("Warning: MONGODB_URI not set, using mock data")
                    self._setup_mock_collections()
            except Exception as e:
                print(f"Error connecting to MongoDB: {str(e)}")
                self._setup_mock_collections()
            self._initialized = True
    
    def _setup_mock_collections(self):
        """Setup mock collections for testing."""
        class MockCollection:
            def __init__(self):
                self.data = []
            
            def find(self, query=None):
                return self.data
            
            def find_one(self, query=None):
                return self.data[0] if self.data else None
            
            def insert_one(self, document):
                self.data.append(document)
        
        self.businesses = MockCollection()
        self.questions = MockCollection()
    
    def get_questions_for_stage(self, business_stage: str) -> List[Question]:
        """Get assessment questions for a specific business stage."""
        try:
            if hasattr(self, 'questions'):
                questions_data = self.questions.find({"business_stage": business_stage})
                return [Question(**q) for q in questions_data]
            return self._get_default_questions(business_stage)
        except Exception as e:
            print(f"Error getting questions: {str(e)}")
            return self._get_default_questions(business_stage)
    
    def _get_default_questions(self, business_stage: str) -> List[Question]:
        """Get default questions if database is not available."""
        questions = []
        if business_stage == "Startup":
            questions = [
                Question(
                    id="1",
                    text="What is your current monthly revenue?",
                    phase="Financial Performance",
                    business_stage="Startup"
                ),
                Question(
                    id="2",
                    text="What is your customer acquisition cost?",
                    phase="Market & Competition",
                    business_stage="Startup"
                )
            ]
        elif business_stage == "Growth":
            questions = [
                Question(
                    id="3",
                    text="What is your year-over-year growth rate?",
                    phase="Financial Performance",
                    business_stage="Growth"
                ),
                Question(
                    id="4",
                    text="What is your market share in your primary market?",
                    phase="Market & Competition",
                    business_stage="Growth"
                )
            ]
        else:  # Mature
            questions = [
                Question(
                    id="5",
                    text="What is your EBITDA margin?",
                    phase="Financial Performance",
                    business_stage="Mature"
                ),
                Question(
                    id="6",
                    text="What are your main competitive advantages?",
                    phase="Market & Competition",
                    business_stage="Mature"
                )
            ]
        return questions

class LLMService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.client = None
            try:
                api_key = os.getenv('GROQ_API_KEY')
                if api_key:
                    self.client = Groq(api_key=api_key)
                else:
                    print("Warning: GROQ_API_KEY not set, LLM features will be limited")
            except Exception as e:
                print(f"Warning: Failed to initialize Groq client: {str(e)}")
            self._initialized = True
    
    def analyze_business_profile(self, profile: BusinessProfile) -> Dict[str, Any]:
        """Analyze a business profile using LLM."""
        if not self.client:
            return self._fallback_analysis(profile)
        
        try:
            prompt = self._create_analysis_prompt(profile)
            response = self.client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You are an expert business analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=500
            )
            
            return self._parse_analysis_response(response.choices[0].message.content)
        except Exception as e:
            print(f"Error analyzing business profile: {str(e)}")
            return self._fallback_analysis(profile)
    
    def _fallback_analysis(self, profile: BusinessProfile) -> Dict[str, Any]:
        """Provide basic analysis when LLM is not available."""
        return {
            "strengths": profile.competitive_advantages[:3],
            "weaknesses": profile.key_risks[:3],
            "opportunities": profile.growth_opportunities[:3],
            "recommendations": [
                "Complete analysis with LLM service for detailed recommendations"
            ]
        }
    
    def _create_analysis_prompt(self, profile: BusinessProfile) -> str:
        """Create prompt for business profile analysis."""
        return f"""
        Analyze the following business:
        
        Name: {profile.name}
        Industry: {profile.industry}
        Stage: {profile.business_stage}
        
        Financial Metrics:
        - Revenue: ${profile.financial_metrics.revenue:,.2f}
        - Growth Rate: {profile.financial_metrics.growth_rate:.1%}
        
        Market Metrics:
        - Market Size: ${profile.market_metrics.total_market_size:,.2f}
        - Market Share: {profile.market_metrics.market_share:.1%}
        
        Provide analysis in the following format:
        - Strengths:
        - Weaknesses:
        - Opportunities:
        - Recommendations:
        """

class ValuationService:
    def __init__(self, db: DatabaseConnection):
        self.db = db
    
    def calculate_valuation(self, profile: BusinessProfile) -> List[ValuationResult]:
        """Calculate business valuation using multiple methods."""
        results = []
        
        # Revenue Multiple Valuation
        if profile.financial_metrics.revenue > 0:
            revenue_val = self._calculate_revenue_multiple(profile)
            if revenue_val:
                results.append(revenue_val)
        
        # EBITDA Multiple Valuation
        if profile.financial_metrics.ebitda and profile.financial_metrics.ebitda > 0:
            ebitda_val = self._calculate_ebitda_multiple(profile)
            if ebitda_val:
                results.append(ebitda_val)
        
        return results
    
    def _calculate_revenue_multiple(self, profile: BusinessProfile) -> Optional[ValuationResult]:
        """Calculate valuation using revenue multiple method."""
        try:
            # Get base multiple based on industry and stage
            base_multiple = self._get_industry_multiple(profile.industry, "revenue")
            
            # Adjust multiple based on growth and margins
            adjusted_multiple = self._adjust_multiple(
                base_multiple,
                profile.financial_metrics.growth_rate,
                profile.financial_metrics.profit_margin
            )
            
            value = profile.financial_metrics.revenue * adjusted_multiple
            
            return ValuationResult(
                method=ValuationMethod.REVENUE_MULTIPLE,
                value=value,
                confidence_score=0.7,
                multiplier_used=adjusted_multiple,
                assumptions={
                    "base_multiple": base_multiple,
                    "growth_rate": profile.financial_metrics.growth_rate,
                    "profit_margin": profile.financial_metrics.profit_margin
                }
            )
        except Exception as e:
            print(f"Error calculating revenue multiple valuation: {str(e)}")
            return None
    
    def _calculate_ebitda_multiple(self, profile: BusinessProfile) -> Optional[ValuationResult]:
        """Calculate valuation using EBITDA multiple method."""
        try:
            # Get base multiple based on industry and stage
            base_multiple = self._get_industry_multiple(profile.industry, "ebitda")
            
            # Adjust multiple based on growth and margins
            adjusted_multiple = self._adjust_multiple(
                base_multiple,
                profile.financial_metrics.growth_rate,
                profile.financial_metrics.ebitda_margin
            )
            
            value = profile.financial_metrics.ebitda * adjusted_multiple
            
            return ValuationResult(
                method=ValuationMethod.EBITDA_MULTIPLE,
                value=value,
                confidence_score=0.8,
                multiplier_used=adjusted_multiple,
                assumptions={
                    "base_multiple": base_multiple,
                    "growth_rate": profile.financial_metrics.growth_rate,
                    "ebitda_margin": profile.financial_metrics.ebitda_margin
                }
            )
        except Exception as e:
            print(f"Error calculating EBITDA multiple valuation: {str(e)}")
            return None
    
    def _get_industry_multiple(self, industry: str, method: str) -> float:
        """Get industry-specific multiple from database or defaults."""
        # TODO: Implement database lookup for industry multiples
        defaults = {
            "SaaS": {"revenue": 10.0, "ebitda": 15.0},
            "E-commerce": {"revenue": 3.0, "ebitda": 12.0},
            "Manufacturing": {"revenue": 2.0, "ebitda": 8.0}
        }
        
        industry_data = defaults.get(industry, {"revenue": 4.0, "ebitda": 10.0})
        return industry_data.get(method, 5.0)
    
    def _adjust_multiple(
        self,
        base_multiple: float,
        growth_rate: Optional[float],
        margin: Optional[float]
    ) -> float:
        """Adjust multiple based on growth and margins."""
        adjustment = 1.0
        
        if growth_rate and growth_rate > 0.2:  # High growth
            adjustment += 0.2
        elif growth_rate and growth_rate > 0.1:  # Moderate growth
            adjustment += 0.1
        
        if margin and margin > 0.2:  # High margin
            adjustment += 0.2
        elif margin and margin > 0.1:  # Moderate margin
            adjustment += 0.1
        
        return base_multiple * adjustment

class AssessmentService:
    def __init__(self, db: DatabaseConnection, llm_service: LLMService):
        self.db = db
        self.llm_service = llm_service
        self.current_sessions: Dict[str, AssessmentSession] = {}
    
    def start_assessment(self, business_profile: BusinessProfile) -> AssessmentSession:
        """Start a new assessment session."""
        session = AssessmentSession(
            business_profile=business_profile,
            current_phase="Initial Discovery"
        )
        self.current_sessions[business_profile.id] = session
        return session
    
    def get_next_question(self, session_id: str) -> Optional[Question]:
        """Get next question based on current phase and previous answers."""
        session = self.current_sessions.get(session_id)
        if not session:
            return None
        
        questions = self.db.get_questions_for_stage(session.business_profile.business_stage)
        if not questions:
            return None
        
        # Filter out already asked questions
        asked_ids = {q.id for q in session.questions_asked}
        available_questions = [q for q in questions if q.id not in asked_ids]
        
        if not available_questions:
            return None
        
        # Select next question
        next_question = available_questions[0]  # Simple selection for now
        session.questions_asked.append(next_question)
        
        return next_question
    
    def process_answer(self, session_id: str, answer: Answer) -> bool:
        """Process an answer and update the assessment session."""
        session = self.current_sessions.get(session_id)
        if not session:
            return False
        
        session.answers_received.append(answer)
        
        # Analyze answer using LLM
        analysis = self.llm_service.analyze_business_profile(session.business_profile)
        
        # Update session with findings
        if "risks" in analysis:
            session.red_flags.extend(analysis["risks"])
        if "opportunities" in analysis:
            session.opportunities.extend(analysis["opportunities"])
        
        # Check if we should advance to next phase
        if self._should_advance_phase(session):
            self._advance_phase(session)
        
        return True
    
    def _should_advance_phase(self, session: AssessmentSession) -> bool:
        """Determine if assessment should advance to next phase."""
        phase_questions = len([q for q in session.questions_asked 
                             if q.phase == session.current_phase])
        return phase_questions >= 3
    
    def _advance_phase(self, session: AssessmentSession) -> None:
        """Advance to next assessment phase."""
        phases = [
            "Initial Discovery",
            "Business Model Deep Dive",
            "Market & Competition Analysis",
            "Financial Performance",
            "Team & Operations",
            "Investment & Growth Strategy"
        ]
        
        current_index = phases.index(session.current_phase)
        if current_index < len(phases) - 1:
            session.current_phase = phases[current_index + 1]
        else:
            session.completed = True

# Streamlit UI
def main():
    st.title("Business Insights Hub")
    
    # Initialize services
    db = DatabaseConnection()
    llm_service = LLMService()
    valuation_service = ValuationService(db)
    assessment_service = AssessmentService(db, llm_service)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Business Assessment", "Valuation Analysis"]
    )
    
    if page == "Business Assessment":
        show_assessment_page(assessment_service)
    else:
        show_valuation_page(valuation_service)

def show_assessment_page(assessment_service: AssessmentService):
    st.header("Business Assessment")
    
    # Business Profile Form
    with st.form("business_profile"):
        st.subheader("Business Information")
        name = st.text_input("Business Name")
        industry = st.selectbox(
            "Industry",
            ["SaaS", "E-commerce", "Manufacturing", "Other"]
        )
        stage = st.selectbox(
            "Business Stage",
            ["Startup", "Growth", "Mature"]
        )
        
        # Financial Information
        st.subheader("Financial Information")
        revenue = st.number_input("Annual Revenue ($)", min_value=0.0)
        growth_rate = st.number_input("Growth Rate (%)", min_value=0.0, max_value=100.0) / 100
        
        submitted = st.form_submit_button("Start Assessment")
        
        if submitted:
            # Create business profile
            profile = BusinessProfile(
                id=str(hash(name + industry)),
                name=name,
                industry=industry,
                business_stage=stage,
                business_type="Unknown",
                description="",
                founded_date=datetime.now(),
                financial_metrics=FinancialMetrics(
                    revenue=revenue,
                    growth_rate=growth_rate
                ),
                market_metrics=MarketMetrics()
            )
            
            # Start assessment session
            session = assessment_service.start_assessment(profile)
            st.session_state.assessment_session = session
            st.session_state.current_question = assessment_service.get_next_question(profile.id)
    
    # Show current question if assessment is in progress
    if hasattr(st.session_state, 'assessment_session'):
        session = st.session_state.assessment_session
        question = st.session_state.current_question
        
        if question:
            st.subheader(f"Phase: {session.current_phase}")
            st.write(question.text)
            
            answer = st.text_area("Your Answer")
            if st.button("Submit Answer"):
                answer_obj = Answer(question=question, text=answer)
                if assessment_service.process_answer(session.business_profile.id, answer_obj):
                    # Get next question
                    st.session_state.current_question = assessment_service.get_next_question(
                        session.business_profile.id
                    )
                    st.experimental_rerun()
        elif session.completed:
            st.success("Assessment completed!")
            st.write("Red Flags:", session.red_flags)
            st.write("Opportunities:", session.opportunities)

def show_valuation_page(valuation_service: ValuationService):
    st.header("Valuation Analysis")
    
    # Business Profile Form
    with st.form("valuation_profile"):
        st.subheader("Business Information")
        name = st.text_input("Business Name")
        industry = st.selectbox(
            "Industry",
            ["SaaS", "E-commerce", "Manufacturing", "Other"]
        )
        stage = st.selectbox(
            "Business Stage",
            ["Startup", "Growth", "Mature"]
        )
        
        # Financial Information
        st.subheader("Financial Information")
        revenue = st.number_input("Annual Revenue ($)", min_value=0.0)
        ebitda = st.number_input("EBITDA ($)", min_value=-1e9)
        growth_rate = st.number_input("Growth Rate (%)", min_value=0.0, max_value=100.0) / 100
        
        submitted = st.form_submit_button("Calculate Valuation")
        
        if submitted:
            # Create business profile
            profile = BusinessProfile(
                id=str(hash(name + industry)),
                name=name,
                industry=industry,
                business_stage=stage,
                business_type="Unknown",
                description="",
                founded_date=datetime.now(),
                financial_metrics=FinancialMetrics(
                    revenue=revenue,
                    ebitda=ebitda,
                    growth_rate=growth_rate
                ),
                market_metrics=MarketMetrics()
            )
            
            # Calculate valuation
            results = valuation_service.calculate_valuation(profile)
            
            # Display results
            st.subheader("Valuation Results")
            for result in results:
                st.write(f"\n{result.method.value.title()} Valuation:")
                st.write(f"Value: ${result.value:,.2f}")
                st.write(f"Multiple Used: {result.multiplier_used:.2f}x")
                st.write(f"Confidence Score: {result.confidence_score:.1%}")
                
                with st.expander("Assumptions"):
                    for key, value in result.assumptions.items():
                        st.write(f"{key.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    main()
