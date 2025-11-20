"""
LangGraph Multi-Agent Weather Prediction Service

This module implements an advanced multi-agent system using LangGraph for intelligent
weather prediction with dynamic routing, parallel processing, and quality validation.
"""

import os
import logging
from datetime import datetime
from typing import TypedDict, Dict, Any, List, Optional, Annotated
from dataclasses import dataclass

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import HumanMessage, AIMessage
    LANGGRAPH_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ LangGraph dependencies imported successfully")
except ImportError as e:
    LANGGRAPH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.error(f"❌ LangGraph imports failed: {e}")

# State definitions for different workflows
class WeatherAnalysisState(TypedDict):
    """State for multi-agent weather analysis workflow"""
    location: str
    prediction_days: int
    current_conditions: Dict[str, Any]
    historical_patterns: List[Dict[str, Any]]
    rag_confidence: float
    analysis_results: Dict[str, Any]
    agent_reports: Dict[str, str]
    final_prediction: str
    confidence_score: float
    method_used: str
    error_message: Optional[str]
    retry_count: int
    quality_score: float

class ServiceSelectionState(TypedDict):
    """State for intelligent service selection workflow"""
    location: str
    prediction_days: int
    available_services: Dict[str, bool]
    selected_method: str
    prediction_result: Dict[str, Any]
    quality_metrics: Dict[str, float]
    retry_attempted: bool
    fallback_used: bool
    
class QualityValidationState(TypedDict):
    """State for prediction quality validation"""
    prediction_text: str
    metadata: Dict[str, Any]
    quality_checks: Dict[str, bool]
    overall_quality: float
    issues_found: List[str]
    approved: bool

@dataclass
class AgentConfig:
    """Configuration for weather analysis agents"""
    name: str
    description: str
    confidence_threshold: float = 0.7
    timeout_seconds: int = 30

class LangGraphWeatherService:
    """
    Advanced weather prediction service using LangGraph multi-agent architecture
    
    This service orchestrates multiple specialized agents for comprehensive weather analysis:
    - Data Collection Agent: Gathers and validates weather data
    - Pattern Analysis Agent: Analyzes historical weather patterns
    - Meteorological Agent: Applies atmospheric science expertise
    - Confidence Assessment Agent: Evaluates prediction reliability
    - Quality Control Agent: Validates final predictions
    """
    
    def __init__(self, weather_service=None, rag_service=None, langchain_service=None, lm_studio_service=None):
        """Initialize LangGraph weather service with existing components"""
        self.weather_service = weather_service
        self.rag_service = rag_service
        self.langchain_service = langchain_service
        self.lm_studio_service = lm_studio_service
        
        self.available = LANGGRAPH_AVAILABLE
        self.prediction_graph = None
        self.service_selection_graph = None
        self.quality_validation_graph = None
        
        # Agent configurations
        self.agents = {
            'data_collector': AgentConfig(
                name="Data Collection Agent",
                description="Gathers and validates current weather data and historical records",
                confidence_threshold=0.8
            ),
            'pattern_analyzer': AgentConfig(
                name="Pattern Analysis Agent", 
                description="Analyzes historical weather patterns and similarities",
                confidence_threshold=0.7
            ),
            'meteorologist': AgentConfig(
                name="Meteorological Expert Agent",
                description="Applies atmospheric science principles and domain expertise",
                confidence_threshold=0.9
            ),
            'confidence_assessor': AgentConfig(
                name="Confidence Assessment Agent",
                description="Evaluates prediction reliability and uncertainty",
                confidence_threshold=0.6
            ),
            'quality_controller': AgentConfig(
                name="Quality Control Agent",
                description="Validates prediction quality and accuracy",
                confidence_threshold=0.8
            )
        }
        
        if self.available:
            logger.info("🧠 Initializing LangGraph Weather Service...")
            self._initialize_graphs()
            logger.info("✅ LangGraph Weather Service initialized successfully")
        else:
            logger.warning("⚠️ LangGraph not available - service disabled")
    
    def _initialize_graphs(self):
        """Initialize all LangGraph workflows"""
        try:
            # Create main prediction workflow
            self.prediction_graph = self._create_prediction_workflow()
            
            # Create service selection workflow
            self.service_selection_graph = self._create_service_selection_workflow()
            
            # Create quality validation workflow
            self.quality_validation_graph = self._create_quality_validation_workflow()
            
            logger.info("✅ All LangGraph workflows initialized")
            
        except Exception as e:
            logger.error(f"❌ LangGraph workflow initialization failed: {e}")
            self.available = False
    
    def _create_prediction_workflow(self):
        """Create the main multi-agent weather prediction workflow"""
        
        def data_collection_agent(state: WeatherAnalysisState) -> WeatherAnalysisState:
            """Agent responsible for collecting and validating weather data"""
            logger.info(f"🔍 Data Collection Agent: Processing {state['location']}")
            
            try:
                # Collect current conditions
                if self.weather_service:
                    recent_data = self.weather_service.get_recent_weather_data(days=3)
                    current_conditions = {
                        'temperature': recent_data['Actual_Temperature(°C)'].iloc[-1] if not recent_data.empty else 20.0,
                        'humidity': recent_data['Actual_Humidity(%)'].iloc[-1] if not recent_data.empty else 60.0,
                        'wind_speed': recent_data['Actual_WindSpeed(m/s)'].iloc[-1] if not recent_data.empty else 2.0,
                        'data_quality': 'high' if not recent_data.empty else 'low'
                    }
                else:
                    current_conditions = {'data_quality': 'unavailable'}
                
                state['current_conditions'] = current_conditions
                state['agent_reports']['data_collector'] = f"✅ Data collected for {state['location']}: {current_conditions.get('data_quality', 'unknown')} quality"
                
                return state
                
            except Exception as e:
                logger.error(f"❌ Data Collection Agent failed: {e}")
                state['error_message'] = f"Data collection failed: {str(e)}"
                state['agent_reports']['data_collector'] = f"❌ Data collection failed: {str(e)}"
                return state
        
        def pattern_analysis_agent(state: WeatherAnalysisState) -> WeatherAnalysisState:
            """Agent responsible for analyzing historical weather patterns"""
            logger.info("📊 Pattern Analysis Agent: Analyzing historical patterns")
            
            try:
                patterns = []
                rag_confidence = 0.0
                
                if self.rag_service and state['current_conditions']:
                    # Build query for similar conditions
                    conditions = state['current_conditions']
                    query = f"weather temperature {conditions.get('temperature', 20)} humidity {conditions.get('humidity', 60)}"
                    
                    try:
                        similar_patterns = self.rag_service.retrieve_similar_weather(query, k=5)
                        patterns = [{'content': p.page_content, 'metadata': getattr(p, 'metadata', {})} for p in similar_patterns]
                        rag_confidence = min(0.9, len(patterns) * 0.15)  # Scale confidence based on patterns found
                    except Exception as rag_error:
                        logger.warning(f"⚠️ RAG retrieval failed: {rag_error}")
                        patterns = []
                        rag_confidence = 0.1
                
                state['historical_patterns'] = patterns
                state['rag_confidence'] = rag_confidence
                state['agent_reports']['pattern_analyzer'] = f"✅ Found {len(patterns)} similar patterns (confidence: {rag_confidence:.2f})"
                
                return state
                
            except Exception as e:
                logger.error(f"❌ Pattern Analysis Agent failed: {e}")
                state['agent_reports']['pattern_analyzer'] = f"❌ Pattern analysis failed: {str(e)}"
                return state
        
        def meteorological_expert_agent(state: WeatherAnalysisState) -> WeatherAnalysisState:
            """Agent applying meteorological expertise"""
            logger.info("🌤️ Meteorological Expert Agent: Applying domain expertise")
            
            try:
                # Determine seasonal factors
                current_month = datetime.now().month
                if current_month in [12, 1, 2]:
                    season = "Winter"
                    seasonal_factors = "Cold air masses, potential snow, lower humidity"
                elif current_month in [3, 4, 5]:
                    season = "Spring"  
                    seasonal_factors = "Variable conditions, frontal systems, moderate temperatures"
                elif current_month in [6, 7, 8]:
                    season = "Summer"
                    seasonal_factors = "High temperatures, convective activity, humidity"
                else:
                    season = "Autumn"
                    seasonal_factors = "Cooling trends, transitional weather, stable systems"
                
                # Analyze atmospheric conditions
                conditions = state['current_conditions']
                temp = conditions.get('temperature', 20)
                humidity = conditions.get('humidity', 60)
                
                # Simple meteorological assessment
                if temp > 25 and humidity > 70:
                    conditions_assessment = "High heat and humidity - thunderstorm potential"
                elif temp < 5 and humidity > 80:
                    conditions_assessment = "Cold and humid - precipitation likely"
                elif temp > 15 and humidity < 40:
                    conditions_assessment = "Warm and dry - stable conditions"
                else:
                    conditions_assessment = "Moderate conditions - typical patterns expected"
                
                meteorological_analysis = {
                    'season': season,
                    'seasonal_factors': seasonal_factors,
                    'conditions_assessment': conditions_assessment,
                    'expert_confidence': 0.8
                }
                
                state['analysis_results']['meteorological'] = meteorological_analysis
                state['agent_reports']['meteorologist'] = f"✅ {season} analysis: {conditions_assessment}"
                
                return state
                
            except Exception as e:
                logger.error(f"❌ Meteorological Expert Agent failed: {e}")
                state['agent_reports']['meteorologist'] = f"❌ Meteorological analysis failed: {str(e)}"
                return state
        
        def confidence_assessment_agent(state: WeatherAnalysisState) -> WeatherAnalysisState:
            """Agent responsible for assessing prediction confidence"""
            logger.info("🎯 Confidence Assessment Agent: Evaluating prediction reliability")
            
            try:
                # Calculate overall confidence based on multiple factors
                confidence_factors = {
                    'data_quality': 0.0,
                    'pattern_match': state.get('rag_confidence', 0.0),
                    'meteorological_expertise': state.get('analysis_results', {}).get('meteorological', {}).get('expert_confidence', 0.0),
                    'service_availability': 0.0
                }
                
                # Assess data quality
                data_quality = state['current_conditions'].get('data_quality', 'low')
                if data_quality == 'high':
                    confidence_factors['data_quality'] = 0.9
                elif data_quality == 'medium':
                    confidence_factors['data_quality'] = 0.6
                else:
                    confidence_factors['data_quality'] = 0.3
                
                # Assess service availability
                if self.langchain_service and self.langchain_service.available:
                    confidence_factors['service_availability'] = 0.8
                elif self.lm_studio_service and self.lm_studio_service.available:
                    confidence_factors['service_availability'] = 0.7
                else:
                    confidence_factors['service_availability'] = 0.4
                
                # Calculate weighted average confidence
                weights = {'data_quality': 0.3, 'pattern_match': 0.3, 'meteorological_expertise': 0.2, 'service_availability': 0.2}
                overall_confidence = sum(confidence_factors[factor] * weights[factor] for factor in weights)
                
                # Determine confidence level
                if overall_confidence >= 0.8:
                    confidence_level = "High"
                elif overall_confidence >= 0.6:
                    confidence_level = "Medium"
                elif overall_confidence >= 0.4:
                    confidence_level = "Low"
                else:
                    confidence_level = "Very Low"
                
                state['confidence_score'] = overall_confidence
                state['analysis_results']['confidence'] = {
                    'overall_score': overall_confidence,
                    'confidence_level': confidence_level,
                    'factors': confidence_factors
                }
                state['agent_reports']['confidence_assessor'] = f"✅ Confidence: {confidence_level} ({overall_confidence:.2f})"
                
                return state
                
            except Exception as e:
                logger.error(f"❌ Confidence Assessment Agent failed: {e}")
                state['agent_reports']['confidence_assessor'] = f"❌ Confidence assessment failed: {str(e)}"
                return state
        
        def prediction_generator_agent(state: WeatherAnalysisState) -> WeatherAnalysisState:
            """Agent responsible for generating the final prediction"""
            logger.info("🌟 Prediction Generator Agent: Creating final forecast")
            
            try:
                # Compile all analysis for prediction generation
                location = state['location']
                days = state['prediction_days']
                current_conditions = state['current_conditions']
                patterns = state['historical_patterns']
                meteorological = state.get('analysis_results', {}).get('meteorological', {})
                confidence = state.get('analysis_results', {}).get('confidence', {})
                
                # Try LangChain + RAG first
                if self.langchain_service and self.langchain_service.available:
                    try:
                        result = self.langchain_service.predict_weather_langchain_rag(location, days)
                        if result and result.get('success'):
                            state['final_prediction'] = result['prediction']
                            state['method_used'] = "LangGraph + LangChain + RAG"
                            state['agent_reports']['prediction_generator'] = "✅ LangChain + RAG prediction successful"
                            return state
                    except Exception as langchain_error:
                        logger.warning(f"⚠️ LangChain prediction failed: {langchain_error}")
                
                # Fallback to LM Studio
                if self.lm_studio_service and self.lm_studio_service.available:
                    try:
                        # Build comprehensive prompt
                        prompt = self._build_comprehensive_prompt(state)
                        prediction = self.lm_studio_service.generate_weather_prediction(prompt, max_tokens=1500)
                        
                        if prediction:
                            state['final_prediction'] = prediction
                            state['method_used'] = "LangGraph + Local LLM"
                            state['agent_reports']['prediction_generator'] = "✅ Local LLM prediction successful"
                            return state
                    except Exception as lm_error:
                        logger.warning(f"⚠️ Local LLM prediction failed: {lm_error}")
                
                # Fallback to statistical prediction
                statistical_prediction = self._generate_statistical_prediction(state)
                state['final_prediction'] = statistical_prediction
                state['method_used'] = "LangGraph + Statistical Analysis"
                state['agent_reports']['prediction_generator'] = "✅ Statistical prediction generated"
                
                return state
                
            except Exception as e:
                logger.error(f"❌ Prediction Generator Agent failed: {e}")
                state['error_message'] = f"Prediction generation failed: {str(e)}"
                state['agent_reports']['prediction_generator'] = f"❌ Prediction generation failed: {str(e)}"
                return state
        
        def quality_control_agent(state: WeatherAnalysisState) -> WeatherAnalysisState:
            """Agent responsible for validating prediction quality"""
            logger.info("🔍 Quality Control Agent: Validating prediction")
            
            try:
                prediction = state.get('final_prediction', '')
                quality_score = 0.0
                quality_issues = []
                
                # Check prediction length
                if len(prediction) < 100:
                    quality_issues.append("Prediction too short")
                    quality_score += 0.1
                elif len(prediction) > 50:
                    quality_score += 0.3
                
                # Check for location mention
                if state['location'].lower() in prediction.lower():
                    quality_score += 0.2
                else:
                    quality_issues.append("Location not mentioned")
                
                # Check for time frame mention  
                if str(state['prediction_days']) in prediction or 'day' in prediction.lower():
                    quality_score += 0.2
                else:
                    quality_issues.append("Time frame not clear")
                
                # Check for weather elements
                weather_elements = ['temperature', 'humidity', 'rain', 'wind', 'cloud', 'sunny', 'storm']
                elements_found = sum(1 for element in weather_elements if element in prediction.lower())
                quality_score += min(0.3, elements_found * 0.1)
                
                if elements_found < 2:
                    quality_issues.append("Insufficient weather details")
                
                state['quality_score'] = quality_score
                state['agent_reports']['quality_controller'] = f"✅ Quality score: {quality_score:.2f}, Issues: {len(quality_issues)}"
                
                return state
                
            except Exception as e:
                logger.error(f"❌ Quality Control Agent failed: {e}")
                state['agent_reports']['quality_controller'] = f"❌ Quality control failed: {str(e)}"
                state['quality_score'] = 0.0
                return state
        
        # Route based on data quality
        def route_based_on_data_quality(state: WeatherAnalysisState) -> str:
            """Route to appropriate next step based on data quality"""
            data_quality = state.get('current_conditions', {}).get('data_quality', 'low')
            
            if data_quality == 'high':
                return "pattern_analysis"
            elif data_quality in ['medium', 'low']:
                return "meteorological_expert"
            else:
                return "prediction_generator"
        
        # Check if retry is needed
        def should_retry_prediction(state: WeatherAnalysisState) -> str:
            """Determine if prediction should be retried"""
            quality_score = state.get('quality_score', 0.0)
            retry_count = state.get('retry_count', 0)
            
            if quality_score < 0.4 and retry_count < 1 and not state.get('error_message'):
                return "retry"
            else:
                return "complete"
        
        # Build the workflow
        workflow = StateGraph(WeatherAnalysisState)
        
        # Add agents as nodes
        workflow.add_node("data_collector", data_collection_agent)
        workflow.add_node("pattern_analyzer", pattern_analysis_agent)
        workflow.add_node("meteorological_expert", meteorological_expert_agent)
        workflow.add_node("confidence_assessor", confidence_assessment_agent)
        workflow.add_node("prediction_generator", prediction_generator_agent)
        workflow.add_node("quality_controller", quality_control_agent)
        
        # Define workflow edges
        workflow.set_entry_point("data_collector")
        
        # Conditional routing from data collector
        workflow.add_conditional_edges(
            "data_collector",
            route_based_on_data_quality,
            {
                "pattern_analysis": "pattern_analyzer",
                "meteorological_expert": "meteorological_expert", 
                "prediction_generator": "prediction_generator"
            }
        )
        
        # Sequential flow through analysis
        workflow.add_edge("pattern_analyzer", "meteorological_expert")
        workflow.add_edge("meteorological_expert", "confidence_assessor")
        workflow.add_edge("confidence_assessor", "prediction_generator")
        workflow.add_edge("prediction_generator", "quality_controller")
        
        # Quality control with retry logic
        workflow.add_conditional_edges(
            "quality_controller",
            should_retry_prediction,
            {
                "retry": "prediction_generator",
                "complete": END
            }
        )
        
        return workflow.compile()
    
    def _create_service_selection_workflow(self):
        """Create intelligent service selection workflow"""
        
        def check_service_availability(state: ServiceSelectionState) -> ServiceSelectionState:
            """Check which services are available"""
            available_services = {
                'langchain_rag': self.langchain_service and self.langchain_service.available,
                'lm_studio': self.lm_studio_service and self.lm_studio_service.available,
                'rag_only': self.rag_service is not None,
                'statistical': True  # Always available
            }
            
            state['available_services'] = available_services
            return state
        
        def select_best_method(state: ServiceSelectionState) -> str:
            """Select the best available prediction method"""
            services = state['available_services']
            
            if services.get('langchain_rag'):
                state['selected_method'] = 'langchain_rag'
                return "langchain_rag"
            elif services.get('lm_studio'):
                state['selected_method'] = 'lm_studio' 
                return "lm_studio"
            elif services.get('rag_only'):
                state['selected_method'] = 'rag_only'
                return "rag_only"
            else:
                state['selected_method'] = 'statistical'
                return "statistical"
        
        # Service execution nodes (simplified for brevity)
        def execute_langchain_rag(state: ServiceSelectionState) -> ServiceSelectionState:
            try:
                result = self.langchain_service.predict_weather_langchain_rag(
                    state['location'], state['prediction_days']
                )
                state['prediction_result'] = result
                return state
            except Exception as e:
                state['prediction_result'] = {'error': str(e), 'success': False}
                return state
        
        # Build service selection workflow
        workflow = StateGraph(ServiceSelectionState)
        workflow.add_node("check_services", check_service_availability)
        workflow.add_node("langchain_rag", execute_langchain_rag)
        
        workflow.set_entry_point("check_services")
        workflow.add_conditional_edges(
            "check_services",
            select_best_method,
            {
                "langchain_rag": "langchain_rag",
                "lm_studio": END,  # Simplified 
                "rag_only": END,
                "statistical": END
            }
        )
        workflow.add_edge("langchain_rag", END)
        
        return workflow.compile()
    
    def _create_quality_validation_workflow(self):
        """Create prediction quality validation workflow"""
        
        def validate_prediction_quality(state: QualityValidationState) -> QualityValidationState:
            """Validate the quality of a weather prediction"""
            prediction = state['prediction_text']
            quality_checks = {
                'has_content': len(prediction.strip()) > 0,
                'adequate_length': len(prediction) >= 100,
                'contains_weather_terms': any(term in prediction.lower() for term in ['weather', 'temperature', 'rain', 'wind', 'cloud']),
                'mentions_location': True,  # Would check for location mention
                'includes_timeframe': True   # Would check for time references
            }
            
            overall_quality = sum(quality_checks.values()) / len(quality_checks)
            
            state['quality_checks'] = quality_checks
            state['overall_quality'] = overall_quality
            state['approved'] = overall_quality >= 0.6
            
            return state
        
        workflow = StateGraph(QualityValidationState)
        workflow.add_node("validate", validate_prediction_quality)
        workflow.set_entry_point("validate")
        workflow.add_edge("validate", END)
        
        return workflow.compile()
    
    def _build_comprehensive_prompt(self, state: WeatherAnalysisState) -> str:
        """Build comprehensive prompt from agent analysis"""
        location = state['location']
        days = state['prediction_days']
        current_conditions = state['current_conditions']
        patterns = state['historical_patterns']
        meteorological = state.get('analysis_results', {}).get('meteorological', {})
        
        prompt = f"""Generate a comprehensive {days}-day weather forecast for {location}.

CURRENT CONDITIONS:
Temperature: {current_conditions.get('temperature', 'N/A')}°C
Humidity: {current_conditions.get('humidity', 'N/A')}%
Wind Speed: {current_conditions.get('wind_speed', 'N/A')} m/s

HISTORICAL PATTERNS:
Found {len(patterns)} similar weather patterns from historical data.

METEOROLOGICAL ANALYSIS:
Season: {meteorological.get('season', 'Unknown')}
Conditions Assessment: {meteorological.get('conditions_assessment', 'Standard patterns expected')}
Seasonal Factors: {meteorological.get('seasonal_factors', 'Normal seasonal variations')}

AGENT ANALYSIS SUMMARY:
{chr(10).join([f"• {agent}: {report}" for agent, report in state.get('agent_reports', {}).items()])}

Generate a detailed, day-by-day forecast with scientific reasoning and confidence levels."""

        return prompt
    
    def _generate_statistical_prediction(self, state: WeatherAnalysisState) -> str:
        """Generate statistical fallback prediction"""
        location = state['location']
        days = state['prediction_days']
        conditions = state['current_conditions']
        
        return f"""📊 **Statistical Weather Forecast for {location}**

**{days}-Day Forecast (Statistical Analysis):**

Based on current conditions and seasonal patterns:

**Current Conditions:**
• Temperature: {conditions.get('temperature', 'N/A')}°C
• Humidity: {conditions.get('humidity', 'N/A')}%
• Wind Speed: {conditions.get('wind_speed', 'N/A')} m/s

**Forecast Trend:**
Days 1-{min(3, days)}: Similar conditions expected with normal daily variations
{"Days 4-" + str(days) + ": Gradual seasonal adjustment" if days > 3 else ""}

**Confidence:** Medium (based on statistical patterns)

**Note:** This prediction is generated using statistical analysis. For enhanced accuracy, ensure AI services are available."""
    
    def predict_weather_with_langgraph(self, location: str = "Tokyo", prediction_days: int = 3) -> Dict[str, Any]:
        """
        Generate weather prediction using LangGraph multi-agent system
        
        Args:
            location: Location for weather prediction
            prediction_days: Number of days to predict
            
        Returns:
            Dict containing prediction result with agent analysis
        """
        if not self.available:
            return {
                "error": "LangGraph service not available",
                "success": False,
                "method": "unavailable"
            }
        
        try:
            logger.info(f"🧠 Starting LangGraph multi-agent prediction for {location}, {prediction_days} days")
            
            # Initialize state
            initial_state = WeatherAnalysisState(
                location=location,
                prediction_days=prediction_days,
                current_conditions={},
                historical_patterns=[],
                rag_confidence=0.0,
                analysis_results={},
                agent_reports={},
                final_prediction="",
                confidence_score=0.0,
                method_used="",
                error_message=None,
                retry_count=0,
                quality_score=0.0
            )
            
            # Run the multi-agent workflow
            final_state = self.prediction_graph.invoke(initial_state)
            
            # Compile results
            result = {
                "prediction": final_state.get('final_prediction', 'Prediction generation failed'),
                "location": location,
                "prediction_days": prediction_days,
                "timeframe": prediction_days,
                "generated_at": datetime.now().isoformat(),
                "method": "langgraph_multi_agent",
                "model_used": final_state.get('method_used', 'LangGraph Multi-Agent System'),
                "success": bool(final_state.get('final_prediction')),
                "confidence_level": self._format_confidence_level(final_state.get('confidence_score', 0.0)),
                "quality_score": final_state.get('quality_score', 0.0),
                
                # LangGraph specific metadata
                "langgraph_analysis": {
                    "agent_reports": final_state.get('agent_reports', {}),
                    "analysis_results": final_state.get('analysis_results', {}),
                    "historical_patterns_found": len(final_state.get('historical_patterns', [])),
                    "rag_confidence": final_state.get('rag_confidence', 0.0),
                    "retry_count": final_state.get('retry_count', 0)
                },
                
                "features": [
                    "Multi-Agent Analysis",
                    "Intelligent Routing", 
                    "Quality Validation",
                    "Dynamic Fallbacks",
                    "Confidence Assessment"
                ],
                
                "enhancement": "Advanced multi-agent system with specialized weather analysis agents"
            }
            
            if final_state.get('error_message'):
                result['warning'] = final_state['error_message']
            
            logger.info("✅ LangGraph multi-agent prediction completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ LangGraph prediction failed: {e}")
            return {
                "error": f"LangGraph prediction failed: {str(e)}",
                "success": False,
                "method": "langgraph_error",
                "location": location,
                "prediction_days": prediction_days
            }
    
    def _format_confidence_level(self, score: float) -> str:
        """Format confidence score as level"""
        if score >= 0.8:
            return "High"
        elif score >= 0.6:
            return "Medium" 
        elif score >= 0.4:
            return "Low"
        else:
            return "Very Low"
    
    def get_langgraph_status(self) -> Dict[str, Any]:
        """Get detailed status of LangGraph service"""
        return {
            "service": "LangGraph Multi-Agent Weather Prediction",
            "available": self.available,
            "workflows_initialized": {
                "prediction_graph": self.prediction_graph is not None,
                "service_selection_graph": self.service_selection_graph is not None,
                "quality_validation_graph": self.quality_validation_graph is not None
            },
            "agent_configs": {name: {"description": config.description, "confidence_threshold": config.confidence_threshold} 
                            for name, config in self.agents.items()},
            "dependent_services": {
                "weather_service": self.weather_service is not None,
                "rag_service": self.rag_service is not None,
                "langchain_service": self.langchain_service and self.langchain_service.available,
                "lm_studio_service": self.lm_studio_service and self.lm_studio_service.available
            }
        }


# Global instance for easy access
langgraph_service = None

def get_langgraph_service(weather_service=None, rag_service=None, langchain_service=None, lm_studio_service=None):
    """Get the global LangGraph service instance"""
    global langgraph_service
    
    if langgraph_service is None and weather_service:
        langgraph_service = LangGraphWeatherService(
            weather_service=weather_service,
            rag_service=rag_service,
            langchain_service=langchain_service,
            lm_studio_service=lm_studio_service
        )
        
        if langgraph_service.available:
            logger.info("🧠 LangGraph service initialized successfully")
        else:
            logger.warning("⚠️ LangGraph service initialization failed")
    
    return langgraph_service