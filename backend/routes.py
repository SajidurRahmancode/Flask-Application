from flask import Blueprint, request, jsonify, render_template, session
from flask_jwt_extended import jwt_required, get_jwt_identity
from flask_wtf.csrf import CSRFProtect
from backend.models import db, User, Image
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api = Blueprint('api', __name__)

# Initialize CSRF protection
csrf = CSRFProtect()

# Initialize weather service
weather_service = None
try:
    from backend.weather_service import WeatherPredictionService
    weather_service = WeatherPredictionService()
    print("✅ Weather prediction service initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize weather service: {e}")
    print(f"   Error details: {type(e).__name__}: {str(e)}")
    logger.error(f"Weather service initialization failed: {str(e)}")
    weather_service = None

@api.route('/users', methods=['GET'])
@jwt_required()
def get_users():
    """Get all users (protected route)"""
    users = User.query.all()
    return jsonify([user.to_dict() for user in users])

@api.route('/users/<int:user_id>', methods=['GET'])
@jwt_required()
def get_user(user_id):
    """Get a specific user"""
    user = User.query.get_or_404(user_id)
    return jsonify(user.to_dict())

@api.route('/users/<int:user_id>', methods=['PUT'])
@jwt_required()
def update_user(user_id):
    """Update a user"""
    current_user_id = get_jwt_identity()
    user = User.query.get_or_404(user_id)
    
    # Only allow users to update their own profile or admin functionality
    # For now, allow any authenticated user to update any user
    data = request.get_json()
    
    if 'username' in data:
        # Check if username is already taken by another user
        existing_user = User.query.filter_by(username=data['username']).first()
        if existing_user and existing_user.id != user_id:
            return jsonify({'message': 'Username already exists'}), 400
        user.username = data['username']
    
    if 'email' in data:
        # Check if email is already taken by another user
        existing_user = User.query.filter_by(email=data['email']).first()
        if existing_user and existing_user.id != user_id:
            return jsonify({'message': 'Email already exists'}), 400
        user.email = data['email']
    
    if 'is_active' in data:
        user.is_active = bool(data['is_active'])
    
    try:
        db.session.commit()
        return jsonify({
            'message': 'User updated successfully',
            'user': user.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': 'Error updating user'}), 500

@api.route('/users/<int:user_id>', methods=['DELETE'])
@jwt_required()
def delete_user(user_id):
    """Delete a user and all associated images"""
    user = User.query.get_or_404(user_id)
    
    try:
        # Associated images will be deleted automatically due to cascade
        db.session.delete(user)
        db.session.commit()
        return jsonify({'message': 'User deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': 'Error deleting user'}), 500

@api.route('/images', methods=['GET'])
@jwt_required()
def get_images():
    """Get all images (protected route)"""
    images = Image.query.all()
    return jsonify([image.to_dict() for image in images])

@api.route('/images', methods=['POST'])
@jwt_required()
def create_image():
    """Create a new image"""
    data = request.get_json()
    
    if not data or 'title' not in data or 'url' not in data or 'user_id' not in data:
        return jsonify({'message': 'Title, URL, and user_id are required'}), 400
    
    # Check if user exists
    if not User.query.get(data['user_id']):
        return jsonify({'message': 'User does not exist'}), 400
    
    new_image = Image(
        title=data['title'],
        url=data['url'],
        user_id=data['user_id']
    )
    
    try:
        db.session.add(new_image)
        db.session.commit()
        return jsonify({
            'id': new_image.id,
            'title': new_image.title,
            'url': new_image.url,
            'user_id': new_image.user_id,
            'message': 'Image created successfully'
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': 'Error creating image'}), 500

@api.route('/images/<int:image_id>', methods=['DELETE'])
@jwt_required()
def delete_image(image_id):
    """Delete an image"""
    image = Image.query.get_or_404(image_id)
    
    try:
        db.session.delete(image)
        db.session.commit()
        return jsonify({'message': 'Image deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': 'Error deleting image'}), 500

@api.route('/users/<int:user_id>/images', methods=['GET'])
@jwt_required()
def get_user_images(user_id):
    """Get all images for a specific user"""
    user = User.query.get_or_404(user_id)
    images = Image.query.filter_by(user_id=user_id).all()
    return jsonify([image.to_dict() for image in images])

@api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint (public)"""
    return jsonify({'status': 'healthy', 'message': 'Flask API is running'})

@api.route('/weather/test', methods=['GET', 'POST'])
def test_weather_api():
    """Test endpoint to verify API connectivity"""
    try:
        logger.info("🧪 Weather API test endpoint called")
        
        # Check session authentication
        from flask import session
        if 'user_id' not in session:
            logger.warning("❌ Unauthorized test request")
            return jsonify({'error': 'Authentication required', 'endpoint': 'test'}), 401
        
        # Check if service is available
        service_status = "available" if weather_service is not None else "unavailable"
        
        # Get request info
        method = request.method
        data = None
        
        if method == 'POST':
            if request.is_json:
                data = request.get_json()
                logger.info(f"📊 Received JSON: {data}")
            else:
                data = request.form.to_dict()
                logger.info(f"📊 Received form: {data}")
        else:
            data = request.args.to_dict()
            logger.info(f"📊 Received query: {data}")
        
        logger.info("✅ Weather API test completed successfully")
        
        return jsonify({
            'status': 'success',
            'message': 'Weather API endpoint is working',
            'method': method,
            'data_received': data,
            'weather_service': service_status,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Weather API test failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'API test failed: {str(e)}',
            'endpoint': 'test'
        }), 500

@api.route('/weather/test', methods=['GET'])
def test_weather_service():
    """Test weather service functionality"""
    from flask import session
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    if weather_service is None:
        return jsonify({
            'error': 'Weather service not available',
            'status': 'failed'
        })
    
    try:
        # Test basic functionality
        summary = weather_service.get_data_summary()
        recent_data = weather_service.get_recent_weather_data(3)
        
        return jsonify({
            'status': 'success',
            'message': 'Weather service is working',
            'data_available': summary is not None,
            'recent_data_available': recent_data is not None,
            'records_count': len(weather_service.data) if weather_service.data is not None else 0
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Weather service test failed: {str(e)}'
        }), 500

@api.route('/weather/predict', methods=['GET', 'POST'])
def predict_weather():
    """Weather prediction endpoint using LangChain and Gemini AI"""
    try:
        logger.info("🔮 Weather prediction request received")
        
        # Check session-based authentication
        from flask import session
        if 'user_id' not in session:
            logger.warning("❌ Unauthorized weather prediction request")
            return jsonify({'error': 'Authentication required'}), 401
        
        if weather_service is None:
            logger.error("❌ Weather service not available")
            return jsonify({
                'error': 'Weather service not available',
                'message': 'Please check Gemini API key configuration'
            }), 500
        
        # Get parameters from request
        if request.method == 'POST':
            if request.is_json:
                data = request.get_json() or {}
                logger.info(f"📊 Received JSON data: {data}")
            else:
                data = request.form.to_dict()
                logger.info(f"📊 Received form data: {data}")
        else:
            data = request.args.to_dict()
            logger.info(f"📊 Received query params: {data}")
        
        location = data.get('location', 'Tokyo')
        prediction_days = int(data.get('prediction_days', 3))
        
        logger.info(f"🌍 Predicting weather for {location}, {prediction_days} days")
        
        # Validate input
        if prediction_days < 1 or prediction_days > 10:
            logger.warning(f"❌ Invalid prediction days: {prediction_days}")
            return jsonify({
                'error': 'Invalid prediction days',
                'message': 'Prediction days must be between 1 and 10'
            }), 400
        
        # Generate prediction
        logger.info("🤖 Calling weather service for prediction...")
        result = weather_service.predict_weather(location, prediction_days)
        
        logger.info(f"📈 Prediction result status: {result.get('success', False)}")
        
        if result.get('success'):
            logger.info("✅ Weather prediction completed successfully")
            return jsonify(result), 200
        else:
            logger.error(f"❌ Weather prediction failed: {result.get('error', 'Unknown error')}")
            return jsonify(result), 500
            
    except ValueError as e:
        logger.error(f"❌ ValueError in predict_weather: {e}")
        return jsonify({
            'error': 'Invalid input data',
            'message': str(e),
            'success': False
        }), 400
    except Exception as e:
        logger.error(f"❌ Exception in predict_weather: {e}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e),
            'success': False
        }), 500

@api.route('/weather/data-summary', methods=['GET'])
def get_weather_data_summary():
    """Get summary of the weather dataset"""
    # Check session-based authentication
    from flask import session
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    if weather_service is None:
        return jsonify({
            'error': 'Weather service not available'
        }), 500
    
    try:
        print("Getting weather data summary...")  # Debug log
        summary = weather_service.get_data_summary()
        print(f"Summary result: {summary}")  # Debug log
        
        if summary:
            return jsonify(summary)
        else:
            return jsonify({
                'error': 'Unable to load data summary'
            }), 500
    except Exception as e:
        print(f"Exception in get_weather_data_summary: {e}")
        return jsonify({
            'error': 'Failed to get data summary',
            'message': str(e)
        }), 500

@api.route('/weather/recent-data', methods=['GET'])
def get_recent_weather_data():
    """Get recent weather data for visualization"""
    # Check session-based authentication
    from flask import session
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    if weather_service is None:
        return jsonify({
            'error': 'Weather service not available'
        }), 500
    
    try:
        days = int(request.args.get('days', 7))
        recent_data = weather_service.get_recent_weather_data(days)
        
        if recent_data is not None:
            # Convert to JSON-serializable format
            result = {
                'data': recent_data.to_dict('records'),
                'dates': [str(date) for date in recent_data.index],
                'days_requested': days
            }
            return jsonify(result)
        else:
            return jsonify({
                'error': 'Unable to load recent data'
            }), 500
    except Exception as e:
        return jsonify({
            'error': 'Failed to get recent data',
            'message': str(e)
        }), 500

# ===================== RAG-ENHANCED WEATHER ENDPOINTS =====================

@api.route('/weather/predict-rag', methods=['POST'])
@csrf.exempt
def predict_weather_rag():
    """Enhanced weather prediction using RAG + historical patterns"""
    try:
        logger.info("🧠 RAG-enhanced weather prediction request received")
        
        # Check authentication
        if 'user_id' not in session:
            logger.warning("❌ Unauthorized RAG prediction request")
            return jsonify({'error': 'Authentication required'}), 401
        
        if not weather_service:
            return jsonify({'error': 'Weather service not available'}), 500
        
        # Get request data
        data = request.get_json() if request.is_json else {}
        location = data.get('location', 'Tokyo')
        timeframe = int(data.get('timeframe', 7))
        
        # Validate timeframe
        if timeframe < 1 or timeframe > 14:
            return jsonify({'error': 'Timeframe must be between 1 and 14 days'}), 400
        
        logger.info(f"🔍 Starting RAG prediction: {location}, {timeframe} days")
        
        # Generate RAG-enhanced prediction
        prediction = weather_service.predict_weather_with_rag(location, timeframe)
        
        return jsonify({
            'prediction': prediction,
            'location': location,
            'timeframe': timeframe,
            'method': 'RAG-Enhanced',
            'timestamp': datetime.now().isoformat(),
            'success': True
        }), 200
        
    except Exception as e:
        logger.error(f"❌ RAG prediction error: {str(e)}")
        return jsonify({'error': f'RAG prediction failed: {str(e)}'}), 500

@api.route('/weather/rag-search', methods=['POST'])
@csrf.exempt
def rag_search_patterns():
    """Search historical weather patterns using RAG"""
    try:
        logger.info("🔍 RAG weather pattern search request received")
        
        # Check authentication
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        
        if not weather_service or not weather_service.rag_service:
            return jsonify({'error': 'RAG service not available'}), 500
        
        # Get request data
        data = request.get_json() if request.is_json else {}
        query = data.get('query', '')
        limit = int(data.get('limit', 5))
        
        if not query:
            return jsonify({'error': 'Query parameter required'}), 400
        
        if limit > 20:
            limit = 20  # Cap results
        
        logger.info(f"🔍 RAG searching patterns: '{query}', limit: {limit}")
        
        # Search using RAG service
        results = weather_service.rag_service.retrieve_similar_weather(query, k=limit)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'content': result.get('content', ''),
                'metadata': result.get('metadata', {}),
                'doc_type': result.get('doc_type', 'unknown'),
                'relevance_score': result.get('relevance_score', 0.0)
            })
        
        return jsonify({
            'success': True,
            'query': query,
            'results': formatted_results,
            'count': len(formatted_results),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ RAG search error: {str(e)}")
        return jsonify({'error': f'RAG search failed: {str(e)}'}), 500

@api.route('/weather/rag-stats', methods=['GET'])
def get_rag_stats():
    """Get RAG service statistics and status"""
    try:
        logger.info("📊 RAG statistics request received")
        
        # Check authentication
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        
        if not weather_service:
            return jsonify({'error': 'Weather service not available'}), 500
        
        # Get RAG service stats
        if weather_service.rag_service:
            rag_stats = weather_service.rag_service.get_stats()
            rag_available = weather_service.rag_service.is_available()
            
            return jsonify({
                'success': True,
                'rag_available': rag_available,
                'stats': rag_stats,
                'service_status': 'active' if rag_available else 'inactive',
                'timestamp': datetime.now().isoformat()
            }), 200
        else:
            return jsonify({
                'success': False,
                'rag_available': False,
                'stats': {},
                'service_status': 'not_initialized',
                'error': 'RAG service not initialized',
                'timestamp': datetime.now().isoformat()
            }), 200
        
    except Exception as e:
        logger.error(f"❌ RAG stats error: {str(e)}")
        return jsonify({'error': f'RAG stats failed: {str(e)}'}), 500

# ===== LOCAL LLM ENDPOINTS =====

@api.route('/weather/predict-local', methods=['POST'])
@csrf.exempt  # Exempt from CSRF for API
def predict_weather_local():
    """Local LLM weather prediction endpoint"""
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401
            
        logger.info("🏠 Local LLM weather prediction request received")
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        logger.info(f"📊 Received JSON data: {data}")
        
        location = data.get('location', 'Tokyo')
        timeframe = int(data.get('timeframe', 3))
        
        # Validate inputs
        if timeframe < 1 or timeframe > 10:
            return jsonify({"error": "Timeframe must be between 1 and 10 days"}), 400
            
        logger.info(f"🏠 Predicting weather for {location}, {timeframe} days using local LLM")
        logger.info("🤖 Calling weather service for local LLM prediction...")
        
        # Get local LLM prediction
        result = weather_service.predict_weather_with_local_llm(location, timeframe)
        
        if result and result.get('success'):
            logger.info("✅ Local LLM weather prediction successful")
            return jsonify(result), 200
        else:
            logger.error(f"❌ Local LLM weather prediction failed: {result}")
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"❌ Local LLM prediction endpoint error: {str(e)}")
        return jsonify({
            "error": f"Server error: {str(e)}",
            "success": False
        }), 500

@api.route('/weather/predict-rag-local', methods=['POST'])
@csrf.exempt  # Exempt from CSRF for API
def predict_weather_rag_local():
    """RAG + Local LLM weather prediction endpoint"""
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401
            
        logger.info("🧠 RAG + Local LLM weather prediction request received")
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        location = data.get('location', 'Tokyo')
        timeframe = int(data.get('timeframe', 7))
        
        # Validate inputs
        if timeframe < 1 or timeframe > 10:
            return jsonify({"error": "Timeframe must be between 1 and 10 days"}), 400
            
        logger.info(f"🧠 Predicting weather for {location}, {timeframe} days using RAG + Local LLM")
        
        # Get RAG + Local LLM prediction
        result = weather_service.predict_weather_with_rag_local_llm(location, timeframe)
        
        if result and result.get('success'):
            logger.info("✅ RAG + Local LLM weather prediction successful")
            return jsonify(result), 200
        else:
            logger.error(f"❌ RAG + Local LLM prediction failed")
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"❌ RAG + Local LLM prediction endpoint error: {str(e)}")
        return jsonify({
            "error": f"Server error: {str(e)}",
            "success": False
        }), 500

@api.route('/weather/predict-hybrid', methods=['POST'])
@csrf.exempt  # Exempt from CSRF for API  
def predict_weather_hybrid():
    """Hybrid prediction endpoint with intelligent fallback"""
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401
            
        logger.info("🔄 Hybrid weather prediction request received")
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        location = data.get('location', 'Tokyo')
        timeframe = int(data.get('timeframe', 3))
        prefer_local = data.get('prefer_local', True)
        
        # Validate inputs
        if timeframe < 1 or timeframe > 10:
            return jsonify({"error": "Timeframe must be between 1 and 10 days"}), 400
            
        logger.info(f"🔄 Hybrid prediction for {location}, {timeframe} days (prefer_local={prefer_local})")
        
        # Get hybrid prediction
        result = weather_service.predict_weather_hybrid(location, timeframe, prefer_local)
        
        if result and result.get('success'):
            logger.info("✅ Hybrid weather prediction successful")
            return jsonify(result), 200
        else:
            logger.error(f"❌ Hybrid prediction failed")
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"❌ Hybrid prediction endpoint error: {str(e)}")
        return jsonify({
            "error": f"Server error: {str(e)}",
            "success": False
        }), 500

@api.route('/weather/lm-studio-status', methods=['GET'])
def get_lm_studio_status():
    """Get LM Studio service status and connection info"""
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401
            
        logger.info("📊 LM Studio status request received")
        
        # Get LM Studio status
        status = weather_service.get_lm_studio_status()
        
        return jsonify({
            'success': True,
            'lm_studio_status': status,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ LM Studio status error: {str(e)}")
        return jsonify({
            'error': f'LM Studio status check failed: {str(e)}',
            'success': False
        }), 500

@api.route('/weather/predict-langchain-rag', methods=['POST'])
@csrf.exempt  # Exempt from CSRF for API
def predict_weather_langchain_rag():
    """Ultimate weather prediction using LangChain + RAG orchestration"""
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401
            
        logger.info("🧠 LangChain + RAG weather prediction request received")
        
        # Check if weather service is available
        if weather_service is None:
            logger.error("❌ Weather service not initialized")
            return jsonify({
                "error": "Weather service not available. Please check server configuration.",
                "suggestion": "The service requires proper API keys. Try using local-only methods.",
                "alternatives": [
                    "Local LLM Only (if LM Studio is running)",
                    "Statistical Analysis (always available)"
                ],
                "success": False
            }), 503
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        location = data.get('location', 'Tokyo')
        timeframe = int(data.get('timeframe', 3))
        
        # Validate inputs
        if timeframe < 1 or timeframe > 10:
            return jsonify({"error": "Timeframe must be between 1 and 10 days"}), 400
            
        logger.info(f"🧠 LangChain + RAG prediction for {location}, {timeframe} days")
        
        # Check if the method exists on the weather service
        if not hasattr(weather_service, 'predict_weather_langchain_rag'):
            logger.error("❌ LangChain + RAG method not available on weather service")
            return jsonify({
                "error": "LangChain + RAG method not available",
                "suggestion": "Service may not be fully initialized. Try another prediction method.",
                "alternatives": [
                    "🏠 Local LLM Only",
                    "🧠 RAG + Local LLM", 
                    "🔄 Hybrid Smart Fallback"
                ],
                "success": False
            }), 503
        
        # Get LangChain + RAG prediction
        result = weather_service.predict_weather_langchain_rag(location, timeframe)
        
        if result and result.get('success'):
            logger.info("✅ LangChain + RAG weather prediction successful")
            return jsonify(result), 200
        else:
            logger.error(f"❌ LangChain + RAG prediction failed")
            
            # Determine error type for better frontend handling
            error_response = result or {"error": "Prediction failed", "success": False}
            
            # Check for timeout conditions
            if result and (
                result.get('timeout_occurred') or
                (result.get('error') and 'timeout' in str(result.get('error')).lower()) or
                (result.get('note') and 'taking longer than expected' in str(result.get('note')).lower())
            ):
                error_response['error_type'] = 'timeout'
                if not error_response.get('error'):
                    error_response['error'] = 'Prediction is taking longer than expected'
            
            # Check for service unavailable conditions
            elif result and (
                'not available' in str(result.get('error', '')).lower() or
                'service unavailable' in str(result.get('error', '')).lower() or
                'not properly initialized' in str(result.get('error', '')).lower()
            ):
                error_response['error_type'] = 'service_unavailable'
                if not error_response.get('error'):
                    error_response['error'] = 'AI service is not currently available'
            
            return jsonify(error_response), 500
            
    except Exception as e:
        logger.error(f"❌ LangChain + RAG prediction endpoint error: {str(e)}")
        return jsonify({
            "error": f"Server error: {str(e)}",
            "success": False
        }), 500

@api.route('/weather/langchain-rag-status', methods=['GET'])
def get_langchain_rag_status():
    """Get LangChain + RAG service status and capabilities"""
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401
            
        logger.info("📊 LangChain + RAG status request received")
        
        # Get LangChain + RAG status
        status = weather_service.get_langchain_rag_status()
        
        return jsonify({
            'success': True,
            'langchain_rag_status': status,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ LangChain + RAG status error: {str(e)}")
        return jsonify({
            'error': f'LangChain + RAG status check failed: {str(e)}',
            'success': False
        }), 500

@api.route('/weather/service-overview', methods=['GET'])
def get_service_overview():
    """Get comprehensive overview of all weather prediction services"""
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401
            
        logger.info("📈 Service overview request received")
        
        overview = {
            'timestamp': datetime.now().isoformat(),
            'services': {}
        }
        
        # Check Gemini service
        overview['services']['gemini'] = {
            'name': 'Google Gemini AI',
            'available': hasattr(weather_service, 'llm') and weather_service.llm is not None,
            'type': 'cloud',
            'quota_limited': True,
            'endpoints': ['/weather/predict']
        }
        
        # Check RAG service
        rag_available = weather_service.rag_service is not None
        overview['services']['rag'] = {
            'name': 'RAG (Retrieval Augmented Generation)',
            'available': rag_available,
            'type': 'enhancement',
            'quota_limited': True,  # Because it uses Gemini embeddings
            'endpoints': ['/weather/predict-rag', '/weather/rag-search']
        }
        
        # Check LM Studio service
        lm_studio_available = (
            hasattr(weather_service, 'lm_studio') and 
            weather_service.lm_studio is not None and 
            weather_service.lm_studio.available
        )
        overview['services']['lm_studio'] = {
            'name': 'LM Studio Local LLM',
            'available': lm_studio_available,
            'type': 'local',
            'quota_limited': False,
            'endpoints': ['/weather/predict-local', '/weather/predict-rag-local']
        }
        
        # Check hybrid service
        overview['services']['hybrid'] = {
            'name': 'Hybrid Intelligent Fallback',
            'available': True,  # Always available with statistical fallback
            'type': 'intelligent',
            'quota_limited': False,
            'endpoints': ['/weather/predict-hybrid'],
            'fallback_chain': ['RAG + Local LLM', 'Local LLM', 'RAG + Gemini', 'Standard Gemini', 'Statistical Analysis']
        }
        
        # Check statistical fallback
        overview['services']['statistical'] = {
            'name': 'Statistical Analysis',
            'available': True,
            'type': 'fallback',
            'quota_limited': False,
            'endpoints': ['Embedded in all prediction methods']
        }
        
        # Summary
        available_services = sum(1 for service in overview['services'].values() if service['available'])
        quota_free_services = sum(1 for service in overview['services'].values() if service['available'] and not service['quota_limited'])
        
        overview['summary'] = {
            'total_services': len(overview['services']),
            'available_services': available_services,
            'quota_free_services': quota_free_services,
            'recommended_endpoint': '/weather/predict-hybrid',
            'status': '✅ Operational' if available_services >= 2 else '⚠️ Limited'
        }
        
        return jsonify({
            'success': True,
            'overview': overview
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Service overview error: {str(e)}")
        return jsonify({
            'error': f'Service overview failed: {str(e)}',
            'success': False
        }), 500