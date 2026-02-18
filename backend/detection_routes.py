from flask import Blueprint, request, jsonify, Response
from database import db
from models import Detection
from utils import verify_token, allowed_file, save_upload_file
import os, time, csv, io
from datetime import datetime

detection_bp = Blueprint('detection', __name__, url_prefix='/api/detection')

ML_MODEL   = None
MODEL_IS_DEMO = True

def load_ml_model():
    global ML_MODEL, MODEL_IS_DEMO
    try:
        from tensorflow.keras.models import load_model
        from config import Config
        p = str(Config.MODEL_PATH)
        if os.path.exists(p):
            ML_MODEL = load_model(p)
            MODEL_IS_DEMO = False
            print(f'✔ ML Model loaded from {p}')
        else:
            print(f'⚠  Model not found at {p} — DEMO mode.')
    except Exception as e:
        print(f'⚠  Could not load model ({e}) — DEMO mode.')

load_ml_model()

def _demo_prediction():
    import random
    return random.choice(['fake','real']), round(random.uniform(70,95), 2)

def predict_image(image_path):
    start = time.time()
    if ML_MODEL is None:
        r, c = _demo_prediction()
        return r, c, round(time.time()-start,2), True
    try:
        import cv2, numpy as np
        img = cv2.imread(image_path)
        img = cv2.resize(img,(224,224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255.0
        img = np.expand_dims(img,0)
        pred = float(ML_MODEL.predict(img,verbose=0)[0][0])
        if pred > 0.5:
            return 'fake', round(pred*100,2), round(time.time()-start,2), False
        return 'real', round((1-pred)*100,2), round(time.time()-start,2), False
    except Exception as e:
        print(f'Image predict error: {e}')
        r,c = _demo_prediction()
        return r,c,round(time.time()-start,2),True

def predict_video(video_path, num_frames=10):
    start = time.time()
    if ML_MODEL is None:
        r,c = _demo_prediction()
        return r,c,round(time.time()-start,2),True
    try:
        import cv2, numpy as np
        cap   = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs  = np.linspace(0,total-1,num_frames,dtype=int)
        preds = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx))
            ret,frame = cap.read()
            if ret:
                frame = cv2.resize(frame,(224,224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32)/255.0
                frame = np.expand_dims(frame,0)
                preds.append(float(ML_MODEL.predict(frame,verbose=0)[0][0]))
        cap.release()
        avg = float(np.mean(preds)) if preds else 0.5
        if avg > 0.5:
            return 'fake',round(avg*100,2),round(time.time()-start,2),False
        return 'real',round((1-avg)*100,2),round(time.time()-start,2),False
    except Exception as e:
        print(f'Video predict error: {e}')
        r,c = _demo_prediction()
        return r,c,round(time.time()-start,2),True

# ── Upload single image ───────────────────────────────────────────────────
@detection_bp.route('/upload-image', methods=['POST'])
def upload_image():
    user = verify_token()
    if not user: return jsonify({'error':'Unauthorized.'}),401
    if 'file' not in request.files: return jsonify({'error':'No file uploaded.'}),400
    file = request.files['file']
    if not file.filename: return jsonify({'error':'No file selected.'}),400
    if not allowed_file(file,'image'): return jsonify({'error':'Invalid image file.'}),400
    try:
        fp = save_upload_file(file,'images')
        r,c,pt,demo = predict_image(fp)
        det = Detection(user_id=user.id, file_name=file.filename, file_type='image',
                        file_path=fp, result=r, confidence=c, processing_time=pt, is_demo=demo)
        db.session.add(det); db.session.commit()
        try:
            from email_service import send_detection_result_email
            send_detection_result_email(user.email, user.full_name, file.filename, r, c, det.id)
        except Exception: pass
        return jsonify({'message':'Image analysed.','result':r,'confidence':c,
                        'processing_time':pt,'detection_id':det.id,'is_demo':demo}),200
    except Exception as e:
        db.session.rollback(); print(f'upload-image error:{e}')
        return jsonify({'error':'Analysis failed.'}),500

# ── Upload single video ───────────────────────────────────────────────────
@detection_bp.route('/upload-video', methods=['POST'])
def upload_video():
    user = verify_token()
    if not user: return jsonify({'error':'Unauthorized.'}),401
    if 'file' not in request.files: return jsonify({'error':'No file uploaded.'}),400
    file = request.files['file']
    if not file.filename: return jsonify({'error':'No file selected.'}),400
    if not allowed_file(file,'video'): return jsonify({'error':'Invalid video file.'}),400
    try:
        fp = save_upload_file(file,'videos')
        r,c,pt,demo = predict_video(fp)
        det = Detection(user_id=user.id, file_name=file.filename, file_type='video',
                        file_path=fp, result=r, confidence=c, processing_time=pt, is_demo=demo)
        db.session.add(det); db.session.commit()
        try:
            from email_service import send_detection_result_email
            send_detection_result_email(user.email, user.full_name, file.filename, r, c, det.id)
        except Exception: pass
        return jsonify({'message':'Video analysed.','result':r,'confidence':c,
                        'processing_time':pt,'detection_id':det.id,'is_demo':demo}),200
    except Exception as e:
        db.session.rollback(); print(f'upload-video error:{e}')
        return jsonify({'error':'Analysis failed.'}),500

# ── Bulk upload (multiple images) ─────────────────────────────────────────
@detection_bp.route('/upload-bulk', methods=['POST'])
def upload_bulk():
    user = verify_token()
    if not user: return jsonify({'error':'Unauthorized.'}),401
    files = request.files.getlist('files')
    if not files: return jsonify({'error':'No files uploaded.'}),400
    if len(files) > 10: return jsonify({'error':'Maximum 10 files per batch.'}),400

    results = []
    for file in files:
        if not file.filename: continue
        ftype = 'image' if file.content_type.startswith('image/') else 'video'
        if not allowed_file(file, ftype):
            results.append({'file_name':file.filename,'error':'Invalid file type.'})
            continue
        try:
            sub = 'images' if ftype=='image' else 'videos'
            fp  = save_upload_file(file, sub)
            if ftype=='image':
                r,c,pt,demo = predict_image(fp)
            else:
                r,c,pt,demo = predict_video(fp)
            det = Detection(user_id=user.id, file_name=file.filename, file_type=ftype,
                            file_path=fp, result=r, confidence=c, processing_time=pt, is_demo=demo)
            db.session.add(det)
            db.session.flush()
            results.append({'file_name':file.filename,'result':r,'confidence':c,
                            'processing_time':pt,'detection_id':det.id,'is_demo':demo})
        except Exception as e:
            results.append({'file_name':file.filename,'error':str(e)})

    db.session.commit()
    return jsonify({'results': results, 'total': len(results)}), 200

# ── History (search + sort + pagination) ─────────────────────────────────
@detection_bp.route('/history', methods=['GET'])
def get_history():
    user = verify_token()
    if not user: return jsonify({'error':'Unauthorized.'}),401
    try:
        page        = request.args.get('page',   1,    type=int)
        per_page    = min(request.args.get('limit', 20, type=int), 100)
        search      = request.args.get('search', '').strip()
        sort_by     = request.args.get('sort',   'date')   # date|confidence|result
        order       = request.args.get('order',  'desc')   # asc|desc
        ftype       = request.args.get('type',   'all')
        fresult     = request.args.get('result', 'all')

        q = Detection.query.filter_by(user_id=user.id)

        # Filters
        if ftype   != 'all': q = q.filter(Detection.file_type == ftype)
        if fresult != 'all': q = q.filter(Detection.result    == fresult)

        # Search by filename
        if search:
            q = q.filter(Detection.file_name.ilike(f'%{search}%'))

        # Sort
        col_map = {'date':'created_at','confidence':'confidence','result':'result'}
        col = getattr(Detection, col_map.get(sort_by,'created_at'))
        q   = q.order_by(col.asc() if order=='asc' else col.desc())

        pagination = q.paginate(page=page, per_page=per_page, error_out=False)
        return jsonify({
            'history':  [d.to_dict() for d in pagination.items],
            'total':    pagination.total,
            'page':     page,
            'per_page': per_page,
            'pages':    pagination.pages,
            'has_next': pagination.has_next,
            'has_prev': pagination.has_prev,
        }),200
    except Exception as e:
        print(f'history error:{e}')
        return jsonify({'error':'Failed to load history.'}),500

# ── Export history as CSV ─────────────────────────────────────────────────
@detection_bp.route('/export-csv', methods=['GET'])
def export_csv():
    user = verify_token()
    if not user: return jsonify({'error':'Unauthorized.'}),401
    try:
        detections = Detection.query.filter_by(user_id=user.id)\
                        .order_by(Detection.created_at.desc()).all()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['ID','File Name','Type','Result','Confidence (%)','Processing Time (s)','Demo','Date'])
        for d in detections:
            writer.writerow([d.id, d.file_name, d.file_type, d.result,
                             round(d.confidence,2), round(d.processing_time,2),
                             'Yes' if d.is_demo else 'No',
                             d.created_at.strftime('%Y-%m-%d %H:%M:%S') if d.created_at else ''])
        output.seek(0)
        return Response(output.getvalue(), mimetype='text/csv',
                        headers={'Content-Disposition':'attachment;filename=detection_history.csv'})
    except Exception as e:
        return jsonify({'error':'Export failed.'}),500

# ── Single detection ──────────────────────────────────────────────────────
@detection_bp.route('/detection/<int:detection_id>', methods=['GET'])
def get_detection(detection_id):
    user = verify_token()
    if not user: return jsonify({'error':'Unauthorized.'}),401
    det = Detection.query.filter_by(id=detection_id, user_id=user.id).first()
    if not det: return jsonify({'error':'Detection not found.'}),404
    return jsonify({'detection': det.to_dict()}),200

# ── Delete detection ──────────────────────────────────────────────────────
@detection_bp.route('/detection/<int:detection_id>', methods=['DELETE'])
def delete_detection(detection_id):
    user = verify_token()
    if not user: return jsonify({'error':'Unauthorized.'}),401
    det = Detection.query.filter_by(id=detection_id, user_id=user.id).first()
    if not det: return jsonify({'error':'Detection not found.'}),404
    try:
        if det.file_path and os.path.exists(det.file_path):
            os.remove(det.file_path)
        db.session.delete(det); db.session.commit()
        return jsonify({'message':'Deleted successfully.'}),200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error':'Delete failed.'}),500

# ── Stats ─────────────────────────────────────────────────────────────────
@detection_bp.route('/stats', methods=['GET'])
def get_stats():
    user = verify_token()
    if not user: return jsonify({'error':'Unauthorized.'}),401
    try:
        all_d = Detection.query.filter_by(user_id=user.id).all()
        total = len(all_d)
        fake  = sum(1 for d in all_d if d.result=='fake')
        avg_c = round(sum(d.confidence for d in all_d)/total,2) if total else 0
        # Weekly trend (last 7 days)
        from datetime import timedelta
        weekly = []
        for i in range(6,-1,-1):
            day_start = datetime.utcnow().replace(hour=0,minute=0,second=0,microsecond=0) - timedelta(days=i)
            day_end   = day_start + timedelta(days=1)
            cnt = sum(1 for d in all_d if d.created_at and day_start <= d.created_at < day_end)
            weekly.append({'date': day_start.strftime('%b %d'), 'count': cnt})
        return jsonify({'stats':{
            'total_detections': total,
            'fake_count':       fake,
            'real_count':       total-fake,
            'avg_confidence':   avg_c,
            'weekly_trend':     weekly,
        }}),200
    except Exception as e:
        return jsonify({'error':'Failed to load stats.'}),500
