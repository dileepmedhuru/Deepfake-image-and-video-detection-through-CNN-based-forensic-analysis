from flask import Blueprint, request, jsonify
from database import db
from models import User, Detection
from utils import verify_token
from sqlalchemy import func
from datetime import datetime, timedelta

admin_bp = Blueprint('admin', __name__, url_prefix='/api/admin')


def _require_admin():
    """Return User if admin, else None."""
    user = verify_token()
    if not user or not user.is_admin:
        return None
    return user


# ── Users ─────────────────────────────────────────────────────────────────────

@admin_bp.route('/users', methods=['GET'])
def get_all_users():
    admin = _require_admin()
    if not admin:
        return jsonify({'error': 'Unauthorized – admin access required.'}), 403

    try:
        page     = request.args.get('page', 1,  type=int)
        per_page = request.args.get('limit', 20, type=int)
        per_page = min(per_page, 100)

        pagination = User.query.order_by(User.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )

        users_data = []
        for u in pagination.items:
            d = u.to_dict()
            d['detection_count'] = Detection.query.filter_by(user_id=u.id).count()
            users_data.append(d)

        return jsonify({
            'users':    users_data,
            'total':    pagination.total,
            'page':     page,
            'per_page': per_page,
            'pages':    pagination.pages,
        }), 200

    except Exception as e:
        print(f'get_all_users error: {e}')
        return jsonify({'error': 'Failed to load users.'}), 500


@admin_bp.route('/user/<int:user_id>', methods=['GET'])
def get_user_detail(user_id):
    admin = _require_admin()
    if not admin:
        return jsonify({'error': 'Unauthorized – admin access required.'}), 403

    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found.'}), 404

    try:
        page     = request.args.get('page', 1,  type=int)
        per_page = request.args.get('limit', 10, type=int)
        per_page = min(per_page, 50)

        pagination = (
            Detection.query
            .filter_by(user_id=user_id)
            .order_by(Detection.created_at.desc())
            .paginate(page=page, per_page=per_page, error_out=False)
        )

        user_data = user.to_dict()
        # Exclude file_path from admin view too
        user_data['detections']      = [d.to_dict() for d in pagination.items]
        user_data['detection_count'] = pagination.total

        return jsonify({'user': user_data}), 200

    except Exception as e:
        print(f'get_user_detail error: {e}')
        return jsonify({'error': 'Failed to load user detail.'}), 500


@admin_bp.route('/user/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete a user and all their detections."""
    admin = _require_admin()
    if not admin:
        return jsonify({'error': 'Unauthorized – admin access required.'}), 403

    if admin.id == user_id:
        return jsonify({'error': 'You cannot delete your own account.'}), 400

    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found.'}), 404

    try:
        db.session.delete(user)   # cascade deletes detections
        db.session.commit()
        return jsonify({'message': f'User {user.email} deleted.'}), 200
    except Exception as e:
        db.session.rollback()
        print(f'delete_user error: {e}')
        return jsonify({'error': 'Delete failed.'}), 500


# ── Detections ────────────────────────────────────────────────────────────────

@admin_bp.route('/detections', methods=['GET'])
def get_all_detections():
    admin = _require_admin()
    if not admin:
        return jsonify({'error': 'Unauthorized – admin access required.'}), 403

    try:
        page     = request.args.get('page', 1,  type=int)
        per_page = request.args.get('limit', 20, type=int)
        per_page = min(per_page, 100)

        pagination = (
            db.session.query(Detection, User.full_name, User.email)
            .join(User, Detection.user_id == User.id)
            .order_by(Detection.created_at.desc())
            .paginate(page=page, per_page=per_page, error_out=False)
        )

        detections_data = []
        for det, full_name, email in pagination.items:
            d = det.to_dict()   # file_path already excluded in to_dict()
            d['full_name'] = full_name
            d['email']     = email
            detections_data.append(d)

        return jsonify({
            'detections': detections_data,
            'total':      pagination.total,
            'page':       page,
            'per_page':   per_page,
            'pages':      pagination.pages,
        }), 200

    except Exception as e:
        print(f'get_all_detections error: {e}')
        return jsonify({'error': 'Failed to load detections.'}), 500


# ── Dashboard stats (merged /stats + /dashboard-stats) ───────────────────────

@admin_bp.route('/dashboard-stats', methods=['GET'])
def get_dashboard_stats():
    """
    Comprehensive stats:
      – totals (users, detections, fake, real)
      – daily detection counts for the last 7 days (for chart)
    """
    admin = _require_admin()
    if not admin:
        return jsonify({'error': 'Unauthorized – admin access required.'}), 403

    try:
        total_users      = User.query.count()
        total_detections = Detection.query.count()
        fake_detections  = Detection.query.filter_by(result='fake').count()
        real_detections  = Detection.query.filter_by(result='real').count()

        # Per-day counts for the last 7 days
        daily = []
        for days_ago in range(6, -1, -1):
            day_start = datetime.utcnow().replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=days_ago)
            day_end = day_start + timedelta(days=1)

            count = Detection.query.filter(
                Detection.created_at >= day_start,
                Detection.created_at <  day_end,
            ).count()

            daily.append({
                'date':  day_start.strftime('%b %d'),
                'count': count,
            })

        return jsonify({
            'total_users':      total_users,
            'total_detections': total_detections,
            'fake_detections':  fake_detections,
            'real_detections':  real_detections,
            'daily_activity':   daily,
        }), 200

    except Exception as e:
        print(f'get_dashboard_stats error: {e}')
        return jsonify({'error': 'Failed to load stats.'}), 500


# Keep /stats as a lightweight alias (won't duplicate logic)
@admin_bp.route('/stats', methods=['GET'])
def get_stats():
    return get_dashboard_stats()
