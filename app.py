from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_wtf.csrf import CSRFProtect
from backend.models import db
from backend.routes import api
from backend.auth import auth_bp
from dotenv import load_dotenv
import os
from datetime import timedelta

# Load environment variables
load_dotenv()

def create_app():
    """Application factory pattern"""
    app = Flask(__name__, 
                template_folder='frontend/templates',
                static_folder='frontend/static')
    CORS(app) 

    # Database configuration for MySQL/XAMPP
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '3306')
    DB_NAME = os.getenv('DB_NAME', 'flask_react_app')
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')

    # MySQL connection string
    app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')

    # Session configuration
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

    # JWT Configuration
    app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'jwt-secret-string-change-this-in-production')
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = False  # Tokens don't expire for development

    # CSRF Protection
    app.config['WTF_CSRF_SECRET_KEY'] = os.getenv('CSRF_SECRET_KEY', 'csrf-secret-change-in-production')
    app.config['WTF_CSRF_TIME_LIMIT'] = None  # CSRF tokens don't expire

    # Initialize extensions
    jwt = JWTManager(app)
    csrf = CSRFProtect(app)
    db.init_app(app)

    # Context processors for templates
    @app.context_processor
    def inject_current_user():
        from flask import session
        from backend.models import User
        current_user = None
        if 'user_id' in session:
            current_user = User.query.get(session['user_id'])
        return dict(current_user=current_user)
    
    @app.context_processor
    def inject_csrf_token():
        from flask_wtf.csrf import generate_csrf
        return dict(csrf_token=generate_csrf)

    # Register blueprints
    app.register_blueprint(api, url_prefix='/api')
    app.register_blueprint(auth_bp, url_prefix='/auth')

    # Root route redirect to login
    @app.route('/')
    def index():
        from flask import session, redirect, url_for
        if 'user_id' in session:
            return redirect(url_for('auth.dashboard'))
        return redirect(url_for('auth.login'))

    return app

def initialize_database_and_tables():
    """Initialize database and create tables if they don't exist"""
    try:
        # Import and run database initialization
        from init_database import initialize_database
        success = initialize_database()
        
        if not success:
            print("❌ Database initialization failed!")
            return False
            
        print("✅ Database initialization successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error during database initialization: {e}")
        print("🔧 Make sure XAMPP MySQL is running and try again")
        return False

app = create_app()

# Initialize database and create tables
with app.app_context():
    try:
        print("🔄 Initializing database...")
        if initialize_database_and_tables():
            db.create_all()
            print("✅ Flask-SQLAlchemy tables synchronized!")
        else:
            print("⚠️  Database initialization failed, but continuing...")
            print("    You may need to set up the database manually.")
    except Exception as e:
        print(f"⚠️  Database setup error: {e}")
        print("    Continuing anyway - you may need to set up the database manually.")

if __name__ == '__main__':
    print("🚀 Starting Flask Template Authentication App...")
    print("=" * 50)
    print(f"🌐 Web App will be available at: http://localhost:5000")
    print(f"🔑 Auth pages: http://localhost:5000/auth/login, /auth/signup, /auth/dashboard")
    print(f"📊 API endpoints: http://localhost:5000/api/users, /api/images")
    print(f"🎯 Template-based frontend with Bootstrap 5 and Jinja2")
    print("=" * 50)
    
    app.run(debug=True, port=5000)
