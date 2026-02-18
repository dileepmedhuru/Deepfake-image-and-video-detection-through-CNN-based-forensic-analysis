from flask import Blueprint, request, jsonify
from database import db
from models import User
import jwt, re
from datetime import datetime, timedelta
from config import Config

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

def create_token(user_id: int) -> str:
    payload = {'user_id': user_id,
               'exp': datetime.utcnow() + timedelta(days=30)}
    return jwt.encode(payload, Config.JWT_SECRET_KEY, algorithm='HS256')

def _validate_password_strength(password: str):
    if len(password) < 8:
        return 'Password must be at least 8 characters.'
    if not re.search(r'[A-Z]', password):
        return 'Password must contain at least one uppercase letter.'
    if not re.search(r'[a-z]', password):
        return 'Password must contain at least one lowercase letter.'
    if not re.search(r'[0-9]', password):
        return 'Password must contain at least one number.'
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return 'Password must contain at least one special character.'
    return None

@auth_bp.route('/signup', methods=['POST'])
def signup():
    try:
        data      = request.get_json() or {}
        full_name = data.get('full_name', '').strip()
        email     = data.get('email', '').strip().lower()
        password  = data.get('password', '')
        if not full_name or not email or not password:
            return jsonify({'error': 'All fields are required.'}), 400
        if not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
            return jsonify({'error': 'Invalid email address.'}), 400
        err = _validate_password_strength(password)
        if err:
            return jsonify({'error': err}), 400
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already registered.'}), 400
        new_user = User(full_name=full_name, email=email,
                        is_admin=False, is_verified=True, force_password_change=False)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        try:
            from email_service import send_welcome_email
            send_welcome_email(email, full_name)
        except Exception:
            pass
        token = create_token(new_user.id)
        return jsonify({'message': 'Registration successful.',
                        'token': token, 'user': new_user.to_dict()}), 201
    except Exception as e:
        db.session.rollback()
        print(f'Signup error: {e}')
        return jsonify({'error': 'Registration failed. Please try again.'}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    try:
        data     = request.get_json() or {}
        email    = data.get('email', '').strip().lower()
        password = data.get('password', '')
        if not email or not password:
            return jsonify({'error': 'Email and password are required.'}), 400
        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            return jsonify({'error': 'Invalid email or password.'}), 401
        user.updated_at = datetime.utcnow()
        db.session.commit()
        token = create_token(user.id)
        return jsonify({'message': 'Login successful.',
                        'token': token, 'user': user.to_dict()}), 200
    except Exception as e:
        print(f'Login error: {e}')
        return jsonify({'error': 'Login failed. Please try again.'}), 500

@auth_bp.route('/check-token', methods=['GET'])
def check_token():
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '').strip()
        if not token:
            return jsonify({'error': 'No token provided.'}), 401
        payload = jwt.decode(token, Config.JWT_SECRET_KEY, algorithms=['HS256'])
        user    = User.query.get(payload['user_id'])
        if not user:
            return jsonify({'error': 'User not found.'}), 404
        return jsonify({'valid': True, 'user': user.to_dict()}), 200
    except jwt.ExpiredSignatureError:
        return jsonify({'error': 'Token expired.'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'error': 'Invalid token.'}), 401
    except Exception:
        return jsonify({'error': 'Token verification failed.'}), 500

@auth_bp.route('/change-password', methods=['POST'])
def change_password():
    from utils import verify_token as _verify
    user = _verify()
    if not user:
        return jsonify({'error': 'Unauthorized.'}), 401
    try:
        data    = request.get_json() or {}
        current = data.get('current_password', '')
        new_pw  = data.get('new_password', '')
        if not current or not new_pw:
            return jsonify({'error': 'Both passwords are required.'}), 400
        if not user.check_password(current):
            return jsonify({'error': 'Current password is incorrect.'}), 401
        err = _validate_password_strength(new_pw)
        if err:
            return jsonify({'error': err}), 400
        user.set_password(new_pw)
        user.force_password_change = False
        user.updated_at = datetime.utcnow()
        db.session.commit()
        return jsonify({'message': 'Password changed successfully.'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Password change failed.'}), 500

@auth_bp.route('/forgot-password', methods=['POST'])
def forgot_password():
    try:
        data  = request.get_json() or {}
        email = data.get('email', '').strip().lower()
        if not email:
            return jsonify({'error': 'Email is required.'}), 400
        user = User.query.filter_by(email=email).first()
        if user:
            from email_service import generate_reset_token, send_password_reset_email
            token = generate_reset_token(user.id)
            send_password_reset_email(user.email, user.full_name, token)
        return jsonify({'message': 'If that email is registered, a reset link has been sent.'}), 200
    except Exception as e:
        print(f'forgot-password error: {e}')
        return jsonify({'error': 'Request failed. Please try again.'}), 500

@auth_bp.route('/reset-password', methods=['POST'])
def reset_password():
    try:
        data   = request.get_json() or {}
        token  = data.get('token', '')
        new_pw = data.get('new_password', '')
        if not token or not new_pw:
            return jsonify({'error': 'Token and new password are required.'}), 400
        from email_service import verify_reset_token
        user_id = verify_reset_token(token)
        if not user_id:
            return jsonify({'error': 'Reset link is invalid or has expired.'}), 400
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found.'}), 404
        err = _validate_password_strength(new_pw)
        if err:
            return jsonify({'error': err}), 400
        user.set_password(new_pw)
        user.force_password_change = False
        user.updated_at = datetime.utcnow()
        db.session.commit()
        return jsonify({'message': 'Password reset successfully. You can now log in.'}), 200
    except Exception as e:
        db.session.rollback()
        print(f'reset-password error: {e}')
        return jsonify({'error': 'Reset failed. Please try again.'}), 500
