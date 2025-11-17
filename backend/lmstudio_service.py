"""
LM Studio Local LLM Service Integrates with LM Studio's OpenAI-compatible API for local AI inference
"""

import os
import requests
import json
import logging
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class LMStudioService:
    """Service for interacting with LM Studio local LLM API"""
    
    def __init__(self):
        """Initialize LM Studio service with configuration from environment"""
        self.api_url = os.getenv('LM_STUDIO_API_URL', 'http://127.0.0.1:1239')
        self.timeout = 120  # Increased timeout for complex predictions
        self.available = False
        self.model_info = {}
        
        logger.info(f"🏠 Initializing LM Studio service at {self.api_url}")
        self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if LM Studio server is running and model is loaded"""
        try:
            # Check if server is running
            response = requests.get(f"{self.api_url}/v1/models", timeout=5)
            
            if response.status_code == 200:
                models_data = response.json()
                
                if 'data' in models_data and len(models_data['data']) > 0:
                    self.model_info = models_data['data'][0]
                    model_name = self.model_info.get('id', 'unknown')
                    self.available = True
                    logger.info(f"✅ LM Studio available with model: {model_name}")
                    return True
                else:
                    logger.warning("⚠️ LM Studio running but no model loaded")
                    return False
            else:
                logger.warning(f"⚠️ LM Studio server responded with status: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            logger.warning("⚠️ LM Studio server not reachable - ensure LM Studio is running")
            return False
        except Exception as e:
            logger.warning(f"⚠️ LM Studio availability check failed: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model"""
        if not self.available:
            return {"error": "LM Studio not available"}
        
        try:
            response = requests.get(f"{self.api_url}/v1/models", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Failed to get model info: {response.status_code}"}
        except Exception as e:
            return {"error": f"Model info request failed: {str(e)}"}
    
    def generate_text(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.3) -> Optional[str]:
        """Generate text using the local LLM with simple completion"""
        if not self.available:
            logger.error("❌ LM Studio not available for text generation")
            return None
        
        try:
            payload = {
                "model": self.model_info.get('id', 'local-model'),
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
                "stop": ["\n\n", "User:", "Human:"]
            }
            
            logger.info("🤖 Calling LM Studio for text completion...")
            response = requests.post(
                f"{self.api_url}/v1/completions",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result['choices'][0]['text'].strip()
                logger.info("✅ LM Studio text generation successful")
                return generated_text
            else:
                logger.error(f"❌ LM Studio completion error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ LM Studio text generation failed: {str(e)}")
            return None
    
    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: int = 1000, temperature: float = 0.3) -> Optional[str]:
        """Generate response using chat completion format"""
        if not self.available:
            logger.error("❌ LM Studio not available for chat generation")
            return None
        
        try:
            payload = {
                "model": self.model_info.get('id', 'local-model'),
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            logger.info("💬 Calling LM Studio for chat completion...")
            logger.info(f"🔧 Using timeout: {self.timeout}s, max_tokens: {max_tokens}")
            
            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    generated_text = result['choices'][0]['message']['content']
                    logger.info("✅ LM Studio chat generation successful")
                    logger.info(f"📝 Generated {len(generated_text)} characters")
                    return generated_text
                else:
                    logger.error("❌ LM Studio response missing choices")
                    return None
            else:
                logger.error(f"❌ LM Studio chat error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error(f"❌ LM Studio timeout after {self.timeout}s - prediction may be too complex")
            return "⚠️ Prediction generation timed out. The model was processing but took longer than expected. Try with a shorter timeframe or simpler request."
        except requests.exceptions.ConnectionError:
            logger.error("❌ LM Studio connection failed - check if LM Studio is running")
            return None
        except Exception as e:
            logger.error(f"❌ LM Studio chat generation failed: {str(e)}")
            return None
    
    def generate_weather_prediction(self, prompt: str) -> Optional[str]:
        """Generate weather prediction using optimized parameters"""
        messages = [
            {
                "role": "system", 
                "content": "You are an expert meteorologist and weather prediction specialist. Provide accurate, detailed weather forecasts based on the provided historical data and current conditions. Format your predictions clearly with daily breakdowns and confidence levels."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        return self.generate_chat(
            messages=messages,
            max_tokens=1500,
            temperature=0.2  # Lower temperature for more consistent weather predictions
        )
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection and return detailed status"""
        logger.info("🔍 Testing LM Studio connection...")
        
        # Refresh availability
        self._check_availability()
        
        status = {
            "available": self.available,
            "api_url": self.api_url,
            "timestamp": str(logger.info("Testing at this time")),
        }
        
        if self.available:
            status.update({
                "status": "✅ Connected",
                "model": self.model_info.get('id', 'unknown'),
                "model_info": self.model_info
            })
            
            # Test with a simple generation
            try:
                test_response = self.generate_text("Hello", max_tokens=10)
                status["test_generation"] = "✅ Success" if test_response else "❌ Failed"
                if test_response:
                    status["test_response_length"] = len(test_response)
            except Exception as e:
                status["test_generation"] = f"❌ Error: {str(e)}"
        else:
            status["status"] = "❌ Not available"
            status["help"] = [
                "1. Ensure LM Studio is running",
                "2. Load a model in LM Studio",
                "3. Start the local server in LM Studio",
                f"4. Verify server is running on {self.api_url}"
            ]
        
        return status
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics and health info"""
        stats = {
            "service": "LM Studio Local LLM",
            "api_url": self.api_url,
            "available": self.available,
            "timeout": self.timeout
        }
        
        if self.available and self.model_info:
            stats.update({
                "model_id": self.model_info.get('id'),
                "model_object": self.model_info.get('object'),
                "model_owned_by": self.model_info.get('owned_by', 'local'),
            })
        
        return stats

# Global instance for easy access
lm_studio_service = None

def get_lm_studio_service() -> Optional[LMStudioService]:
    """Get the global LM Studio service instance"""
    global lm_studio_service
    
    if lm_studio_service is None:
        try:
            lm_studio_service = LMStudioService()
            if lm_studio_service.available:
                logger.info("🏠 LM Studio service initialized successfully")
            else:
                logger.warning("⚠️ LM Studio service initialized but not available")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LM Studio service: {e}")
            lm_studio_service = None
    
    return lm_studio_service

# Test function for development
def test_lm_studio():
    """Test function to verify LM Studio integration"""
    service = get_lm_studio_service()
    
    if not service:
        print("❌ LM Studio service not available")
        return
    
    print(f"🔍 Testing LM Studio at {service.api_url}")
    status = service.test_connection()
    
    print("\n📊 Connection Status:")
    for key, value in status.items():
        if isinstance(value, list):
            print(f"  {key}:")
            for item in value:
                print(f"    {item}")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    # Run test when executed directly
    test_lm_studio()