"""
Advanced LangChain + RAG Service for Weather Prediction
Combines LangChain's orchestration with RAG's historical pattern retrieval"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import json

# LangChain imports
try:
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.llms.base import LLM
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    LANGCHAIN_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ LangChain dependencies imported successfully")
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.error(f"❌ LangChain imports failed: {e}")

class LMStudioLangChainWrapper(LLM):
    """LangChain-compatible wrapper for LM Studio service"""
    
    lm_studio_service: Any = None
    
    def __init__(self, lm_studio_service):
        super().__init__()
        self.lm_studio_service = lm_studio_service
    
    def _call(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        """Call LM Studio service with LangChain interface"""
        try:
            response = self.lm_studio_service.generate_weather_prediction(prompt)
            
            # Handle timeout responses
            if response and "⚠️ Prediction generation timed out" in response:
                logger.warning("⚠️ LM Studio prediction timed out but continuing...")
                return response  # Return the timeout message as a valid response
            elif response and response.strip():
                return response
            else:
                logger.error("❌ LM Studio returned empty response")
                return "Unable to generate prediction - LM Studio returned empty response"
                
        except Exception as e:
            logger.error(f"❌ LM Studio call failed: {e}")
            return f"Error generating prediction: {str(e)}"
    
    @property
    def _llm_type(self) -> str:
        return "lmstudio"
    
    @property 
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": "lm_studio_local"}

class LangChainRAGService:
    """Advanced weather prediction using LangChain + RAG orchestration"""
    
    def __init__(self, weather_service, lm_studio_service, rag_service):
        """Initialize LangChain + RAG service"""
        self.weather_service = weather_service
        self.lm_studio_service = lm_studio_service
        self.rag_service = rag_service
        self.available = False
        
        if not LANGCHAIN_AVAILABLE:
            logger.warning("⚠️ LangChain not available - service disabled")
            return
        
        logger.info("🧠 Initializing LangChain + RAG service...")
        self._setup_langchain_components()
    
    def _setup_langchain_components(self):
        """Setup LangChain chains and components"""
        try:
            # Initialize conversation memory for context retention
            self.memory = ConversationBufferWindowMemory(
                k=3,  # Remember last 3 interactions
                memory_key="conversation_history",
                return_messages=False
            )
            
            # Create LangChain-compatible LLM wrapper
            if self.lm_studio_service and self.lm_studio_service.available:
                self.llm = LMStudioLangChainWrapper(self.lm_studio_service)
                logger.info("✅ LM Studio wrapper created for LangChain")
            else:
                logger.error("❌ LM Studio not available for LangChain wrapper")
                return
            
            # Create enhanced prompt templates
            self.rag_prompt_template = self._create_rag_prompt_template()
            self.direct_prompt_template = self._create_direct_prompt_template()
            
            # Create LangChain chains
            self.rag_chain = LLMChain(
                llm=self.llm,
                prompt=self.rag_prompt_template,
                memory=self.memory,
                verbose=True
            )
            
            self.direct_chain = LLMChain(
                llm=self.llm,
                prompt=self.direct_prompt_template,
                verbose=True
            )
            
            self.available = True
            logger.info("✅ LangChain + RAG service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ LangChain + RAG initialization failed: {e}")
            self.available = False
    
    def _create_rag_prompt_template(self):
        """Create enhanced RAG prompt template with LangChain orchestration"""
        
        template = """You are an expert meteorological AI that combines advanced weather analysis with historical pattern recognition.

COMPREHENSIVE WEATHER ANALYSIS:
{weather_input}

ADVANCED ANALYSIS FRAMEWORK:

1. HISTORICAL PATTERN ANALYSIS:
   - Compare current conditions with retrieved historical patterns
   - Identify seasonal trends and atmospheric pressure patterns
   - Analyze temperature, humidity, and wind correlations

2. METEOROLOGICAL REASONING:
   - Apply atmospheric science principles
   - Consider geographical factors for the location
   - Evaluate pressure system movements and frontal analysis

3. TEMPORAL EVOLUTION MODELING:
   - Predict how weather will evolve over forecast period
   - Consider diurnal (daily) variations
   - Account for seasonal transition effects

4. CONFIDENCE ASSESSMENT:
   - High Confidence: Strong historical pattern matches
   - Medium Confidence: Moderate pattern similarity
   - Low Confidence: Limited or conflicting historical data

GENERATION INSTRUCTIONS:
- Use retrieved historical patterns as primary evidence
- Provide detailed daily forecasts with scientific reasoning
- Include confidence levels for each prediction day
- Highlight any unusual patterns or significant changes
- Format response with clear structure and daily breakdowns

Generate comprehensive weather forecast with historical pattern analysis:"""

        return PromptTemplate(
            input_variables=["weather_input"],
            template=template
        )
    
    def _create_direct_prompt_template(self):
        """Create direct prediction template (fallback when RAG unavailable)"""
        
        template = """You are an advanced weather prediction AI using LangChain orchestration.

WEATHER ANALYSIS INPUT:
{weather_input}

Generate detailed weather forecast with scientific analysis:"""

        return PromptTemplate(
            input_variables=["weather_input"],
            template=template
        )
    
    def predict_weather_langchain_rag(self, location="Tokyo", prediction_days=3):
        """Advanced weather prediction using LangChain + RAG orchestration"""
        try:
            if not self.available:
                logger.error("❌ LangChain + RAG service not available")
                return self._create_fallback_response(location, prediction_days, "Service not available")
            
            logger.info(f"🧠 Starting LangChain + RAG prediction for {location}, {prediction_days} days")
            
            # Gather context information
            current_conditions = self._get_current_conditions_summary()
            season = self._get_current_season()
            analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            # Attempt RAG-enhanced prediction first
            result = None
            source_type = "langchain_rag_enhanced"
            method_description = "LangChain + RAG + Historical Patterns"
            
            if self.rag_service and hasattr(self.rag_service, 'retrieve_similar_weather'):
                result = self._run_rag_enhanced_chain(
                    location, prediction_days, current_conditions, season, analysis_date
                )
                
                if result:
                    logger.info("✅ RAG-enhanced LangChain prediction successful")
                else:
                    logger.warning("⚠️ RAG-enhanced chain failed, falling back to direct LangChain")
            
            # Fallback to direct LangChain prediction if RAG failed or unavailable
            if not result:
                logger.info("🔄 Using direct LangChain prediction (no RAG)")
                result = self._run_direct_chain(
                    location, prediction_days, current_conditions, season, analysis_date
                )
                source_type = "langchain_direct"
                method_description = "LangChain + Local LLM (Direct)"
                
                if not result:
                    logger.error("❌ Both RAG-enhanced and direct LangChain predictions failed")
                    raise Exception("All LangChain prediction methods failed")
            
            if result:
                # Check if result indicates a timeout or error
                is_timeout = "⚠️ Prediction generation timed out" in result
                is_error = any(error_phrase in result.lower() for error_phrase in ["error", "unable to generate", "failed"])
                
                # Extract confidence from the result
                confidence = self._extract_confidence_level(result)
                
                # Determine success status
                success_status = not is_error
                
                # Store interaction in memory for future context (only if successful)
                if success_status and not is_timeout:
                    try:
                        self.memory.save_context(
                            {"input": f"Weather prediction for {location}, {prediction_days} days"},
                            {"output": result[:300] + "..." if len(result) > 300 else result}
                        )
                    except Exception as memory_error:
                        logger.warning(f"⚠️ Failed to save to memory: {memory_error}")
                
                # Prepare response with enhanced metadata
                response_data = {
                    "prediction": result,
                    "location": location,
                    "prediction_days": prediction_days,
                    "timeframe": prediction_days,
                    "generated_at": datetime.now().isoformat(),
                    "model_used": method_description,
                    "source": source_type,
                    "method": "langchain_rag",
                    "enhancement": "Advanced LangChain orchestration with historical pattern analysis",
                    "confidence_level": confidence,
                    "success": success_status
                }
                
                # Add timeout-specific metadata
                if is_timeout:
                    response_data.update({
                        "timeout_occurred": True,
                        "error_type": "timeout",
                        "note": "Prediction generation took longer than expected",
                        "suggestion": "Try with a shorter timeframe or simpler request"
                    })
                
                # Add feature metadata
                response_data["features"] = [
                    "LangChain Chain Orchestration",
                    "RAG Historical Pattern Retrieval", 
                    "Conversation Memory",
                    "Confidence Assessment",
                    "Multi-step Reasoning"
                ]
                
                if is_timeout:
                    response_data["features"].append("Timeout Handling")
                
                response_data["fallback_used"] = False
                
                return response_data
            else:
                logger.error("❌ LangChain prediction generation failed")
                return self._create_fallback_response(location, prediction_days, "Prediction generation failed")
                
        except Exception as e:
            logger.error(f"❌ LangChain + RAG prediction error: {str(e)}")
            return self._create_fallback_response(location, prediction_days, f"Error: {str(e)}")
    
    def _run_rag_enhanced_chain(self, location, prediction_days, current_conditions, season, analysis_date):
        """Run RAG-enhanced LangChain prediction with enhanced fallbacks"""
        try:
            # Retrieve similar weather patterns with quota-aware handling
            query = f"weather conditions {current_conditions} {season} season {location}"
            similar_patterns = []
            
            # Try RAG retrieval with graceful quota handling
            if self.rag_service and hasattr(self.rag_service, 'retrieve_similar_weather'):
                try:
                    similar_patterns = self.rag_service.retrieve_similar_weather(query, k=5)
                    logger.info(f"✅ RAG retrieved {len(similar_patterns)} patterns successfully")
                except Exception as rag_error:
                    logger.warning(f"⚠️ RAG retrieval failed, continuing without historical context: {rag_error}")
                    # Continue with empty patterns rather than failing completely
            
            # Format historical context (empty if RAG failed)
            historical_context = self._format_historical_patterns(similar_patterns)
            
            logger.info(f"🔍 Retrieved {len(similar_patterns)} similar patterns for LangChain analysis")
            
            # Create comprehensive weather input
            conversation_history = self.memory.chat_memory.messages[-3:] if hasattr(self.memory, 'chat_memory') else []
            history_text = "\n".join([f"- {msg}" for msg in conversation_history]) if conversation_history else "No previous conversations"
            
            # Enhanced context with fallback messaging
            rag_status = "✅ Historical patterns retrieved via RAG" if similar_patterns else "⚠️ No historical patterns (RAG unavailable)"
            
            weather_input = f"""CONVERSATION HISTORY:
{history_text}

RAG STATUS: {rag_status}

HISTORICAL WEATHER PATTERNS (Retrieved via RAG):
{historical_context}

CURRENT WEATHER CONDITIONS:
{current_conditions}

PREDICTION REQUEST:
- Location: {location}
- Forecast Period: {prediction_days} days
- Current Season: {season}
- Analysis Date: {analysis_date}

ANALYSIS INSTRUCTIONS:
- Use available historical patterns for context (if any)
- Apply meteorological expertise for {location} region
- Consider {season} seasonal patterns
- Provide confidence levels based on available data
- Generate detailed daily forecasts with reasoning"""
            
            # Run the RAG chain with single input
            result = self.rag_chain.run(weather_input=weather_input)
            
            logger.info("✅ LangChain + RAG prediction generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ RAG-enhanced chain failed: {e}")
            # Return None to trigger fallback, don't raise exception
            return None
    
    def _run_direct_chain(self, location, prediction_days, current_conditions, season, analysis_date):
        """Run direct LangChain prediction (no RAG) with enhanced error handling"""
        try:
            weather_input = f"""CURRENT WEATHER CONDITIONS:
{current_conditions}

PREDICTION REQUEST:
- Location: {location}
- Forecast Period: {prediction_days} days
- Current Season: {season}
- Analysis Date: {analysis_date}

ANALYSIS FRAMEWORK:
1. Current condition assessment for {location}
2. {season} seasonal pattern consideration
3. Geographic and topological factors for Japan
4. Multi-day meteorological trend prediction
5. Confidence assessment based on current data

INSTRUCTIONS:
- Provide detailed {prediction_days}-day forecast for {location}
- Include daily temperature, humidity, and weather patterns
- Apply scientific meteorological principles
- Consider seasonal characteristics of {season}
- Assign confidence levels to each prediction day"""
            
            result = self.direct_chain.run(weather_input=weather_input)
            
            logger.info("✅ Direct LangChain prediction generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Direct chain failed: {e}")
            # Log the specific error details for debugging
            logger.error(f"🔍 Direct chain failure details - Location: {location}, Days: {prediction_days}")
            return None
    
    def _format_historical_patterns(self, patterns):
        """Format RAG retrieved patterns for LangChain context"""
        if not patterns:
            return "No similar historical patterns found."
        
        formatted = "RETRIEVED HISTORICAL WEATHER PATTERNS:\n\n"
        
        for i, pattern in enumerate(patterns[:5], 1):  # Use top 5 patterns
            metadata = pattern.metadata
            content = pattern.page_content[:200] + "..." if len(pattern.page_content) > 200 else pattern.page_content
            
            formatted += f"Pattern {i}:\n"
            formatted += f"- Temperature: {metadata.get('temperature', 'N/A')}°C\n"
            formatted += f"- Humidity: {metadata.get('humidity', 'N/A')}%\n"
            formatted += f"- Season: {metadata.get('season', 'N/A')}\n"
            formatted += f"- Wind Speed: {metadata.get('wind_speed', 'N/A')} m/s\n"
            formatted += f"- Context: {content}\n\n"
        
        return formatted
    
    def _get_current_conditions_summary(self):
        """Get comprehensive current weather conditions"""
        try:
            recent_data = self.weather_service.get_recent_weather_data(days=1)
            if recent_data is not None and len(recent_data) > 0:
                latest = recent_data.iloc[-1]
                return f"""Temperature: {latest['Actual_Temperature(°C)']:.1f}°C
Humidity: {latest['Actual_Humidity(%)']:.1f}%
Wind Speed: {latest['Actual_WindSpeed(m/s)']:.1f} m/s
Solar Radiation: {latest['Actual_Solar(kWh/m²/day)']:.2f} kWh/m²/day
Rainfall: {latest['Actual_Rainfall(mm)']:.1f} mm
Cloud Cover: {latest['Actual_CloudCover(0-10)']:.1f}/10"""
            
            return "Current weather conditions unavailable"
            
        except Exception as e:
            logger.error(f"❌ Error getting current conditions: {e}")
            return "Current weather conditions unavailable"
    
    def _get_current_season(self):
        """Get current season based on month"""
        month = datetime.now().month
        seasons = {
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer",
            9: "Autumn", 10: "Autumn", 11: "Autumn"
        }
        return seasons.get(month, "Unknown")
    
    def _extract_confidence_level(self, prediction_text):
        """Extract confidence level from prediction text"""
        text_lower = prediction_text.lower()
        
        if any(phrase in text_lower for phrase in ["high confidence", "very confident", "strong confidence"]):
            return "High"
        elif any(phrase in text_lower for phrase in ["low confidence", "uncertain", "limited confidence"]):
            return "Low"
        elif any(phrase in text_lower for phrase in ["medium confidence", "moderate confidence"]):
            return "Medium"
        
        # Default based on text characteristics
        if len(prediction_text) > 1000:  # Detailed predictions often indicate higher confidence
            return "Medium"
        return "Medium"
    
    def _create_fallback_response(self, location, prediction_days, reason):
        """Create enhanced fallback response when LangChain + RAG fails"""
        logger.warning(f"⚠️ LangChain + RAG fallback: {reason}")
        
        # Provide helpful guidance for different failure reasons
        fallback_message = "LangChain + RAG prediction unavailable. "
        
        if "quota" in reason.lower():
            fallback_message += "Google API quota exceeded. Try using 'RAG + Local LLM' or 'Local LLM Only' methods for unlimited predictions."
        elif "chain failed" in reason.lower():
            fallback_message += "Processing chain encountered an error. Try using 'Hybrid Smart Fallback' for automatic method selection."
        elif "embedding" in reason.lower():
            fallback_message += "Embedding service unavailable. Local embeddings fallback attempted."
        else:
            fallback_message += "Service temporarily unavailable. Please try another prediction method."
        
        return {
            "prediction": fallback_message,
            "location": location,
            "prediction_days": prediction_days,
            "timeframe": prediction_days,
            "generated_at": datetime.now().isoformat(),
            "model_used": "LangChain + RAG (Unavailable)",
            "source": "fallback",
            "method": "langchain_rag_fallback",
            "enhancement": f"Service unavailable: {reason}",
            "confidence_level": "N/A",
            "features": [
                "Service Unavailable",
                "Fallback Guidance Provided",
                "Alternative Methods Suggested"
            ],
            "success": False,
            "fallback_used": True,
            "error": reason,
            "suggested_alternatives": [
                "🏠 Local LLM Only (unlimited)",
                "🧠 RAG + Local LLM (best local option)", 
                "🔄 Hybrid Smart Fallback (recommended)"
            ]
        }
    
    def get_conversation_history(self):
        """Get recent conversation history"""
        try:
            if hasattr(self.memory, 'buffer'):
                return self.memory.buffer
            return "No conversation history available"
        except Exception as e:
            logger.error(f"❌ Error retrieving conversation history: {e}")
            return "Error retrieving conversation history"
    
    def clear_conversation_memory(self):
        """Clear conversation memory"""
        try:
            self.memory.clear()
            logger.info("🧹 LangChain conversation memory cleared")
        except Exception as e:
            logger.error(f"❌ Error clearing memory: {e}")
    
    def get_service_status(self):
        """Get detailed service status"""
        status = {
            "service": "LangChain + RAG Weather Prediction",
            "available": self.available,
            "langchain_available": LANGCHAIN_AVAILABLE,
            "components": {}
        }
        
        if self.available:
            status["components"] = {
                "llm_wrapper": hasattr(self, 'llm') and self.llm is not None,
                "rag_chain": hasattr(self, 'rag_chain') and self.rag_chain is not None,
                "direct_chain": hasattr(self, 'direct_chain') and self.direct_chain is not None,
                "conversation_memory": hasattr(self, 'memory') and self.memory is not None,
                "lm_studio_integration": self.lm_studio_service and self.lm_studio_service.available,
                "rag_service_integration": self.rag_service is not None
            }
        
        return status

# Global instance for easy access
langchain_rag_service = None

def get_langchain_rag_service(weather_service=None, lm_studio_service=None, rag_service=None):
    """Get the global LangChain + RAG service instance"""
    global langchain_rag_service
    
    if langchain_rag_service is None and all([weather_service, lm_studio_service, rag_service]):
        try:
            langchain_rag_service = LangChainRAGService(
                weather_service=weather_service,
                lm_studio_service=lm_studio_service,
                rag_service=rag_service
            )
            if langchain_rag_service.available:
                logger.info("🧠 LangChain + RAG service initialized successfully")
            else:
                logger.warning("⚠️ LangChain + RAG service initialized but not available")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LangChain + RAG service: {e}")
            langchain_rag_service = None
    
    return langchain_rag_service

# Test function for development
def test_langchain_rag():
    """Test function to verify LangChain + RAG integration"""
    print("🧠 Testing LangChain + RAG Service...")
    print(f"LangChain Available: {LANGCHAIN_AVAILABLE}")
    
    if not LANGCHAIN_AVAILABLE:
        print("❌ LangChain not available - install with: pip install langchain")
        return
    
    print("✅ LangChain dependencies available")
    print("Note: Full test requires weather service, LM Studio, and RAG service integration")

if __name__ == "__main__":
    test_langchain_rag()