from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from backend.models import db, User, Image

api = Blueprint('api', __name__)

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