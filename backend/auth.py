from flask import Blueprint, request, jsonify, render_template, redirect, url_for, flash, session
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from backend.models import db, User
from backend.forms import LoginForm, SignupForm, ProfileForm, AddUserForm, EditUserForm
import re

auth_bp = Blueprint('auth', __name__)

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """Validate password strength"""
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    if not re.search(r'[A-Za-z]', password):
        return False, "Password must contain at least one letter"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    return True, "Valid password"

# Helper function to check if user is logged in
def is_logged_in():
    return 'user_id' in session

def get_current_user():
    if is_logged_in():
        return User.query.get(session['user_id'])
    return None

# Template routes
@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Login page and form handling"""
    if is_logged_in():
        return redirect(url_for('auth.dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        username_or_email = form.username_or_email.data.strip()
        password = form.password.data
        
        # Find user by username or email
        user = User.query.filter(
            (User.username == username_or_email) | 
            (User.email == username_or_email)
        ).first()
        
        if user and user.check_password(password):
            if user.is_active:
                session['user_id'] = user.id
                session.permanent = form.remember_me.data
                flash('Login successful!', 'success')
                return redirect(url_for('auth.dashboard'))
            else:
                flash('Account is deactivated.', 'error')
        else:
            flash('Invalid username/email or password.', 'error')
    
    return render_template('login.html', form=form)

@auth_bp.route('/test_signup', methods=['GET', 'POST'])
def test_signup():
    """Test signup page"""
    form = SignupForm()
    print(f"Test - Form submitted: {request.method == 'POST'}")
    print(f"Test - Form data: {request.form}")
    print(f"Test - Form errors: {form.errors}")
    
    if request.method == 'POST':
        print("POST request received!")
        if form.validate_on_submit():
            print("Form validation passed!")
        else:
            print("Form validation failed!")
            print(f"Validation errors: {form.errors}")
    
    return render_template('test_signup.html', form=form)

@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    """Signup page and form handling"""
    if is_logged_in():
        return redirect(url_for('auth.dashboard'))
    
    form = SignupForm()
    
    if form.validate_on_submit():
        # Manual validation for unique username and email
        if User.query.filter_by(username=form.username.data).first():
            flash('Username already exists. Please choose a different one.', 'error')
            return render_template('signup.html', form=form)
        
        if User.query.filter_by(email=form.email.data).first():
            flash('Email already registered. Please use a different email address.', 'error')
            return render_template('signup.html', form=form)
        
        try:
            new_user = User(
                username=form.username.data,
                email=form.email.data
            )
            new_user.set_password(form.password.data)
            
            db.session.add(new_user)
            db.session.commit()
            
            session['user_id'] = new_user.id
            flash('Account created successfully! Welcome!', 'success')
            return redirect(url_for('auth.dashboard'))
            
        except Exception as e:
            db.session.rollback()
            flash('Error creating account. Please try again.', 'error')
    else:
        if request.method == 'POST':
            flash('Please check your form data and try again.', 'error')
            print(f"Form validation errors: {form.errors}")
    
    return render_template('signup.html', form=form)

@auth_bp.route('/dashboard')
def dashboard():
    """User dashboard"""
    if not is_logged_in():
        return redirect(url_for('auth.login'))
    
    current_user = get_current_user()
    if not current_user:
        session.clear()
        return redirect(url_for('auth.login'))
    
    # Get all users for management
    users = User.query.all()
    total_users = len(users)
    active_sessions = len([u for u in users if u.is_active])
    
    return render_template('dashboard.html', 
                         current_user=current_user,
                         users=users,
                         total_users=total_users,
                         active_sessions=active_sessions)

@auth_bp.route('/weather')
def weather_dashboard():
    """Weather prediction dashboard"""
    if not is_logged_in():
        return redirect(url_for('auth.login'))
    
    current_user = get_current_user()
    if not current_user:
        session.clear()
        return redirect(url_for('auth.login'))
    
    return render_template('weather_dashboard.html', current_user=current_user)

@auth_bp.route('/realtime')
def realtime_dashboard():
    """Real-time LangGraph weather prediction dashboard with WebSocket"""
    if not is_logged_in():
        return redirect(url_for('auth.login'))
    
    current_user = get_current_user()
    if not current_user:
        session.clear()
        return redirect(url_for('auth.login'))
    
    return render_template('realtime_dashboard.html', current_user=current_user)

@auth_bp.route('/profile', methods=['GET', 'POST'])
def profile():
    """User profile page"""
    if not is_logged_in():
        return redirect(url_for('auth.login'))
    
    current_user = get_current_user()
    if not current_user:
        session.clear()
        return redirect(url_for('auth.login'))
    
    form = ProfileForm(obj=current_user)
    
    if form.validate_on_submit():
        # Manual validation for unique username and email
        if form.username.data != current_user.username:
            if User.query.filter_by(username=form.username.data).first():
                flash('Username already exists. Please choose a different one.', 'error')
                return render_template('profile.html', form=form, current_user=current_user)
        
        if form.email.data != current_user.email:
            if User.query.filter_by(email=form.email.data).first():
                flash('Email already registered. Please use a different email address.', 'error')
                return render_template('profile.html', form=form, current_user=current_user)
        
        try:
            # Update basic info
            current_user.username = form.username.data
            current_user.email = form.email.data
            
            # Update password if provided
            if form.new_password.data:
                if current_user.check_password(form.current_password.data):
                    current_user.set_password(form.new_password.data)
                else:
                    flash('Current password is incorrect.', 'error')
                    return render_template('profile.html', form=form, current_user=current_user)
            
            db.session.commit()
            flash('Profile updated successfully!', 'success')
            return redirect(url_for('auth.profile'))
            
        except Exception as e:
            db.session.rollback()
            flash('Error updating profile. Please try again.', 'error')
    
    return render_template('profile.html', form=form, current_user=current_user)

@auth_bp.route('/add_user', methods=['POST'])
def add_user():
    """Add new user (admin function)"""
    if not is_logged_in():
        return redirect(url_for('auth.login'))
    
    form = AddUserForm()
    if form.validate_on_submit():
        # Manual validation for unique username and email
        if User.query.filter_by(username=form.username.data).first():
            flash('Username already exists. Please choose a different one.', 'error')
            return redirect(url_for('auth.dashboard'))
        
        if User.query.filter_by(email=form.email.data).first():
            flash('Email already registered. Please use a different email address.', 'error')
            return redirect(url_for('auth.dashboard'))
        
        try:
            new_user = User(
                username=form.username.data,
                email=form.email.data
            )
            new_user.set_password(form.password.data)
            
            db.session.add(new_user)
            db.session.commit()
            flash(f'User "{new_user.username}" added successfully!', 'success')
            
        except Exception as e:
            db.session.rollback()
            flash('Error adding user. Please try again.', 'error')
    else:
        for field, errors in form.errors.items():
            for error in errors:
                flash(f'{field}: {error}', 'error')
    
    return redirect(url_for('auth.dashboard'))

@auth_bp.route('/edit_user/<int:user_id>', methods=['POST'])
def edit_user(user_id):
    """Edit user (admin function)"""
    if not is_logged_in():
        return redirect(url_for('auth.login'))
    
    user = User.query.get_or_404(user_id)
    form = EditUserForm()
    
    if form.validate_on_submit():
        # Manual validation for unique username and email
        if form.username.data != user.username:
            if User.query.filter_by(username=form.username.data).first():
                flash('Username already exists. Please choose a different one.', 'error')
                return redirect(url_for('auth.dashboard'))
        
        if form.email.data != user.email:
            if User.query.filter_by(email=form.email.data).first():
                flash('Email already registered. Please use a different email address.', 'error')
                return redirect(url_for('auth.dashboard'))
        
        try:
            user.username = form.username.data
            user.email = form.email.data
            
            if form.password.data:
                user.set_password(form.password.data)
            
            db.session.commit()
            flash(f'User "{user.username}" updated successfully!', 'success')
            
        except Exception as e:
            db.session.rollback()
            flash('Error updating user. Please try again.', 'error')
    else:
        for field, errors in form.errors.items():
            for error in errors:
                flash(f'{field}: {error}', 'error')
    
    return redirect(url_for('auth.dashboard'))

@auth_bp.route('/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    """Delete user (admin function)"""
    if not is_logged_in():
        return redirect(url_for('auth.login'))
    
    current_user = get_current_user()
    if current_user and current_user.id == user_id:
        flash('You cannot delete your own account from here.', 'error')
        return redirect(url_for('auth.dashboard'))
    
    user = User.query.get_or_404(user_id)
    username = user.username
    
    try:
        db.session.delete(user)
        db.session.commit()
        flash(f'User "{username}" deleted successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash('Error deleting user. Please try again.', 'error')
    
    return redirect(url_for('auth.dashboard'))

@auth_bp.route('/delete_account', methods=['POST'])
def delete_account():
    """Delete current user's account"""
    if not is_logged_in():
        return redirect(url_for('auth.login'))
    
    current_user = get_current_user()
    if not current_user:
        return redirect(url_for('auth.login'))
    
    password = request.form.get('password')
    if not password or not current_user.check_password(password):
        flash('Invalid password. Account deletion cancelled.', 'error')
        return redirect(url_for('auth.profile'))
    
    try:
        db.session.delete(current_user)
        db.session.commit()
        session.clear()
        flash('Your account has been deleted successfully.', 'info')
        return redirect(url_for('auth.login'))
    except Exception as e:
        db.session.rollback()
        flash('Error deleting account. Please try again.', 'error')
        return redirect(url_for('auth.profile'))

@auth_bp.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('auth.login'))

# Keep API routes for backward compatibility
@auth_bp.route('/api/signup', methods=['POST'])
def api_signup():
    """Register a new user (API)"""
    data = request.get_json()
    
    if not data:
        return jsonify({'message': 'No data provided'}), 400
    
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '')
    
    # Validation
    if not username or not email or not password:
        return jsonify({'message': 'Username, email, and password are required'}), 400
    
    if len(username) < 3:
        return jsonify({'message': 'Username must be at least 3 characters long'}), 400
    
    if not validate_email(email):
        return jsonify({'message': 'Invalid email format'}), 400
    
    is_valid, password_message = validate_password(password)
    if not is_valid:
        return jsonify({'message': password_message}), 400
    
    # Check if user already exists
    if User.query.filter_by(username=username).first():
        return jsonify({'message': 'Username already exists'}), 400
    
    if User.query.filter_by(email=email).first():
        return jsonify({'message': 'Email already exists'}), 400
    
    # Create new user
    try:
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        
        db.session.add(new_user)
        db.session.commit()
        
        # Create access token
        access_token = create_access_token(identity=new_user.id)
        
        return jsonify({
            'message': 'User created successfully',
            'access_token': access_token,
            'user': new_user.to_dict_safe()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': 'Error creating user'}), 500

@auth_bp.route('/api/login', methods=['POST'])
def api_login():
    """Authenticate user and return JWT token (API)"""
    data = request.get_json()
    
    if not data:
        return jsonify({'message': 'No data provided'}), 400
    
    username_or_email = data.get('username', '').strip()
    password = data.get('password', '')
    
    if not username_or_email or not password:
        return jsonify({'message': 'Username/email and password are required'}), 400
    
    # Find user by username or email
    user = User.query.filter(
        (User.username == username_or_email) | 
        (User.email == username_or_email)
    ).first()
    
    if not user or not user.check_password(password):
        return jsonify({'message': 'Invalid credentials'}), 401
    
    if not user.is_active:
        return jsonify({'message': 'Account is deactivated'}), 401
    
    # Create access token
    access_token = create_access_token(identity=user.id)
    
    return jsonify({
        'message': 'Login successful',
        'access_token': access_token,
        'user': user.to_dict_safe()
    }), 200

@auth_bp.route('/api/profile', methods=['GET'])
@jwt_required()
def api_get_profile():
    """Get current user profile (API)"""
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)
    
    if not user:
        return jsonify({'message': 'User not found'}), 404
    
    return jsonify({'user': user.to_dict()}), 200

@auth_bp.route('/api/users', methods=['GET'])
@jwt_required()
def api_get_all_users():
    """Get all users (admin functionality) (API)"""
    users = User.query.all()
    return jsonify([user.to_dict() for user in users]), 200