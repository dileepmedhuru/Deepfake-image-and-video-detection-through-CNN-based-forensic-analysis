from flask import Blueprint, request, jsonify, Response
from database import db
from models import Detection
from utils import verify_token, allowed_file, save_upload_file
import os, time, csv, io, json
from datetime import datetime
import cv2, numpy as np

detection_bp = Blueprint('detection', __name__, url_prefix='/api/detection')

ML_MODEL = None
MODEL_IS_DEMO = True

def load_ml_model():
    global ML_MODEL, MODEL_IS_DEMO
    try:
        from tensorflow.keras.models import load_model
        from config import Config
        p = str(Config.MODEL_PATH)
        if os.path.exists(p):
            ML_MODEL = load_model(p)
            try:
                test_img = np.zeros((1, 224, 224, 3), dtype=np.float32)
                test_pred = float(ML_MODEL.predict(test_img, verbose=0)[0][0])
                print(f'✔ ML Model loaded from {p}  (test pred={test_pred:.4f})')
                if abs(test_pred - 0.5) < 0.001:
                    print('⚠  Model outputs exactly 0.5 on blank image — '
                          'may be untrained. Will validate per-image at runtime.')
            except Exception as ve:
                print(f'⚠  Model load-time validation failed: {ve}')
            MODEL_IS_DEMO = False
        else:
            print(f'⚠  Model not found at {p} — DEMO mode.')
    except Exception as e:
        print(f'⚠  Could not load model ({e}) — DEMO mode.')

load_ml_model()


def _demo_prediction():
    import random
    result     = random.choice(['fake', 'real'])
    confidence = round(random.uniform(55, 95), 2)
    return result, confidence


def _is_constant_output_model(pred_value):
    # Only fallback if model outputs EXACTLY 0.5 (untrained model)
    # Threshold 0.005 was too tight — legitimate uncertain predictions triggered fallback
    return abs(pred_value - 0.5) < 0.002


# ══════════════════════════════════════════════════════════════════════════════
# EDITED IMAGE CLASSIFICATION LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def _forensic_suspicion_score(quality_metrics: dict) -> float:
    """
    Pure forensic suspicion score 0-100, independent of ML model.
    Low score (< 20) = strong evidence of a real photograph.
    High score (> 50) = suspicious signals present.
    Used to veto ML false positives on genuine mobile photos.
    """
    if not quality_metrics:
        return 50.0
    score = 0.0
    sat_cv  = quality_metrics.get('sat_cv', 1.0)
    sat_mean= quality_metrics.get('sat_mean', 0)
    bly     = quality_metrics.get('blockiness', 1.0)
    bpp     = quality_metrics.get('bytes_per_pixel', 1.0)

    if sat_cv < 0.50 and sat_mean > 75:  score += 35
    elif sat_cv < 0.60 and sat_mean > 85: score += 20
    if bly > 1.6 and bpp < 0.08:         score += 20
    elif bly > 1.4 and bpp < 0.16:       score += 8
    tv = quality_metrics.get('texture_variance', 999)
    if tv < 20:   score += 18
    elif tv < 35: score += 8
    nr = quality_metrics.get('noise_residual', 999)
    if nr < 1.5:  score += 15
    elif nr < 3.0: score += 6
    ed = quality_metrics.get('edge_density', 0)
    if ed > 0.20:  score += 12
    elif ed > 0.15: score += 5
    lv = quality_metrics.get('face_lighting_variance') or quality_metrics.get('lighting_variance', 0)
    if lv > 45:   score += 10
    elif lv > 35: score += 5
    if quality_metrics.get('freq_ratio', 0) > 0.18:    score += 8
    if quality_metrics.get('channel_imbalance', 0) > 40: score += 8
    if quality_metrics.get('blur_score', 999) < 40:    score += 8
    return min(100.0, score)


def _classify_result(ml_result: str, ml_confidence: float, quality_metrics: dict) -> tuple:
    """
    Three-way classification with face-manipulation and AI-generation signals.

    Returns (result, confidence, was_reclassified)
    result is one of: 'real' | 'fake' | 'ai_generated'

    Logic:
    1. Check for AI-generation signals (saturation CV, blockiness) FIRST
       → If strong AI-gen signals present: return 'ai_generated'
    2. If ML says 'fake' but forensic score is very low (< 20):
       → Override to REAL — model is biased (Celeb-DF trained, poor on mobile photos)
    3. If ML says 'fake' AND confidence > 75 AND forensic score >= 20: keep 'fake'
    4. If ML says 'real': keep 'real'
    """
    if quality_metrics is None:
        return ml_result, ml_confidence, False

    # ── AI-Generation check (independent of ML result) ────────────────
    sat_cv          = quality_metrics.get('sat_cv',          1.0)
    sat_mean        = quality_metrics.get('sat_mean',        100)
    blockiness      = quality_metrics.get('blockiness',      1.0)
    bytes_per_pixel = quality_metrics.get('bytes_per_pixel', 0.5)

    ai_gen_score = 0

    # Primary signal: saturation uniformity
    # Raised sat_mean threshold 60→80 and tightened sat_cv 0.65→0.50
    # to avoid false positives on night photos and indoor shots.
    # Real AI images have VERY uniform saturation AND high colour richness.
    if sat_cv < 0.50 and sat_mean > 75:
        ai_gen_score += 2   # primary signal

    # Supporting signal: compression pattern
    # Lowered bytes_per_pixel threshold 0.16→0.08 to avoid flagging
    # WhatsApp/Telegram compressed real photos (0.08–0.15 range).
    # Only counts when BOTH blockiness is high AND file is very small.
    if blockiness > 1.6 and bytes_per_pixel < 0.08:
        ai_gen_score += 1   # supporting signal

    # Strong AI-generation signature → flag regardless of ML verdict
    if ai_gen_score >= 2:
        ai_conf = round(min(92.0, 70 + (0.50 - sat_cv) * 100), 2)
        return 'ai_generated', ai_conf, True

    if ml_result == 'real':
        # Even when ML says real, check for strong forensic evidence of manipulation.
        # High-quality deepfakes fool the model but still leave forensic traces:
        # very low noise (no sensor noise), extreme saturation uniformity, smooth skin.
        forensic_score_real = _forensic_suspicion_score(quality_metrics)
        if forensic_score_real >= 50:
            # Strong forensic evidence overrides ML 'real' verdict
            print(f'⚠  Forensic override: score={forensic_score_real:.0f} >= 50 '
                  f'despite ML saying real @ {ml_confidence:.1f}% — flagging as fake')
            return 'fake', round(min(90.0, 50 + forensic_score_real * 0.35), 2), True
        return ml_result, ml_confidence, False

    # ── ML says FAKE — apply forensic veto before accepting ───────────
    # The model (trained on Celeb-DF v2 celebrity footage) over-detects on:
    #   • Indian skin tones  • Mobile camera portraits  • Bokeh backgrounds
    #   • Colorful shirts    • Outdoor bright-sky shots
    # We use pure forensic signals as a reality check.

    forensic_score = _forensic_suspicion_score(quality_metrics)

    # VETO 1: Forensic score very low → no manipulation evidence → real photo
    # This catches model bias on genuine mobile photos regardless of ML confidence
    if forensic_score < 20:
        real_conf = round(min(88.0, max(65.0, 92 - forensic_score * 1.4)), 2)
        print(f'ℹ  Veto: forensic_score={forensic_score:.0f} < 20 → real photo (ML was {ml_confidence:.1f}% fake)')
        return 'real', real_conf, True

    # VETO 2: Forensic score moderate (20-35) AND ML confidence not extreme → uncertain → real
    if forensic_score < 35 and ml_confidence < 92:
        real_conf = round(min(82.0, max(60.0, 85 - forensic_score * 0.8)), 2)
        print(f'ℹ  Veto: forensic_score={forensic_score:.0f} moderate, ML {ml_confidence:.1f}% → marking real')
        return 'real', real_conf, True

    # High forensic score (≥35) — there ARE real manipulation signals
    # Check face-specific signals before confirming fake
    has_skin_anomaly  = quality_metrics.get('texture_variance', 999) < 35
    has_boundary_seam = quality_metrics.get('edge_density', 0) > 0.15
    face_lv = quality_metrics.get('face_lighting_variance')
    has_face_lighting_issue = (face_lv is not None and face_lv > 35)
    has_eye_anomaly = (
        quality_metrics.get('eye_symmetry_score') == 0
        and quality_metrics.get('faces_detected', 0) > 0
        and (has_skin_anomaly or has_boundary_seam)
    )
    face_manipulation_score = sum([has_skin_anomaly, has_boundary_seam,
                                   has_face_lighting_issue, has_eye_anomaly])

    if face_manipulation_score == 0 and ml_confidence < 75.0:
        real_confidence = round(min(80.0, max(55.0, 100 - ml_confidence + 10)), 2)
        return 'real', real_confidence, True

    return ml_result, ml_confidence, False


# ══════════════════════════════════════════════════════════════════════════════
# FORENSIC RISK SCORE DIMENSIONS  (replaces confidence plot in frontend)
# ══════════════════════════════════════════════════════════════════════════════

def compute_forensic_risk_scores(quality_metrics: dict) -> dict:
    """
    Compute two dedicated forensic scores:
      deepfake_score  — face-swap / face-manipulation signals (0-100)
      ai_gen_score    — full AI image synthesis signals (0-100)

    Each score is independent. An image can score high on both
    (AI-generated deepfake), one, or neither.
    Brightness gate prevents dark/transition frames from inflating scores.
    """
    if not quality_metrics:
        return {'deepfake_score': 0, 'ai_gen_score': 0}

    brightness = quality_metrics.get('brightness', 100)
    bright_enough = brightness > 40   # dark frames suppress noise/blur/sat signals

    # ── DEEPFAKE SCORE — face manipulation signals ────────────────────
    df = 0

    tv = quality_metrics.get('texture_variance', 50)
    if tv < 15:   df += 25
    elif tv < 25: df += 18
    elif tv < 35: df += 10
    elif tv < 45: df += 4

    if bright_enough:
        nr = quality_metrics.get('noise_residual', 5)
        if nr < 1.5:   df += 20
        elif nr < 2.5: df += 13
        elif nr < 3.5: df += 6

    ed = quality_metrics.get('edge_density', 0)
    if ed > 0.20:   df += 15
    elif ed > 0.15: df += 9
    elif ed > 0.12: df += 4

    lv = quality_metrics.get('face_lighting_variance') or quality_metrics.get('lighting_variance', 0)
    if lv > 45:   df += 15
    elif lv > 35: df += 9
    elif lv > 25: df += 4

    fr = quality_metrics.get('freq_ratio', 0)
    if fr > 0.18:   df += 10
    elif fr > 0.13: df += 5

    if bright_enough:
        bs = quality_metrics.get('blur_score', 999)
        if bs < 20:   df += 10
        elif bs < 40: df += 6
        elif bs < 80: df += 2

    ci = quality_metrics.get('channel_imbalance', 0)
    if ci > 40:   df += 5
    elif ci > 25: df += 2

    deepfake_score = min(100, round(df))

    # ── AI-GENERATION SCORE — synthesis signals ───────────────────────
    ai = 0

    if bright_enough:
        sat_cv   = quality_metrics.get('sat_cv', 1.0)
        sat_mean = quality_metrics.get('sat_mean', 0)
        if sat_cv < 0.25 and sat_mean > 100:   ai += 45
        elif sat_cv < 0.35 and sat_mean > 80:  ai += 38
        elif sat_cv < 0.45 and sat_mean > 75:  ai += 28
        elif sat_cv < 0.50 and sat_mean > 70:  ai += 16
        elif sat_cv < 0.55 and sat_mean > 85:  ai += 18

    bly = quality_metrics.get('blockiness', 1.0)
    bpp = quality_metrics.get('bytes_per_pixel', 1.0)
    if bly > 1.6 and bpp < 0.08:    ai += 25
    elif bly > 1.5 and bpp < 0.12:  ai += 18
    elif bly > 1.4 and bpp < 0.16:  ai += 10

    sp = quality_metrics.get('shadow_pct', 0.1)
    hp = quality_metrics.get('highlight_pct', 0.1)
    if sp < 0.01 and hp < 0.02:   ai += 20
    elif sp < 0.03 and hp < 0.05: ai += 10

    if bright_enough:
        nr = quality_metrics.get('noise_residual', 5)
        if nr < 1.5:   ai += 10
        elif nr < 2.5: ai += 5

    ai_gen_score = min(100, round(ai))

    return {
        'deepfake_score': deepfake_score,
        'ai_gen_score':   ai_gen_score,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FORENSIC ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def analyze_image_quality(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        h, w    = img.shape[:2]
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        b, g, r = cv2.split(img)

        blur_score       = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness       = float(np.mean(gray))
        texture_variance = float(np.std(gray))

        face_cascade   = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces          = face_cascade.detectMultiScale(gray, 1.1, 4)
        faces_detected = len(faces)
        face_regions   = faces.tolist() if len(faces) > 0 else []

        edges        = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0) / edges.size)

        channel_stds      = [float(np.std(b)), float(np.std(g)), float(np.std(r))]
        color_consistency = float(np.mean(channel_stds))
        channel_imbalance = float(max(channel_stds) - min(channel_stds))

        dct_gray         = cv2.resize(gray, (224, 224))
        dct              = cv2.dct(np.float32(dct_gray))
        high_freq_energy = float(np.sum(np.abs(dct[112:, 112:])))
        total_energy     = float(np.sum(np.abs(dct)) + 1e-9)
        freq_ratio       = round(high_freq_energy / total_energy, 4)

        smoothed       = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_residual = float(np.std(gray.astype(np.float32) - smoothed.astype(np.float32)))

        hist       = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist_norm  = hist / (hist.sum() + 1e-9)
        entropy    = float(-np.sum(hist_norm * np.log2(hist_norm + 1e-9)))

        ycrcb      = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        skin_mask  = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        skin_ratio = float(np.sum(skin_mask > 0) / (h * w))

        sobelx  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely  = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        grad_std = float(np.std(grad_mag))

        mid_h, mid_w   = h // 2, w // 2
        quadrant_means = [
            float(np.mean(gray[:mid_h, :mid_w])),
            float(np.mean(gray[:mid_h, mid_w:])),
            float(np.mean(gray[mid_h:, :mid_w])),
            float(np.mean(gray[mid_h:, mid_w:])),
        ]
        lighting_variance = float(np.std(quadrant_means))

        # ── Face-region lighting variance (much more accurate than full-image) ──
        # Full-image lighting variance is misleading for outdoor photos:
        # bright sky vs dark ground creates natural variance that looks
        # "suspicious" but is completely normal (e.g. golden hour, outdoor shots).
        # We compute lighting variance INSIDE the largest face bounding box only.
        face_lighting_variance = None
        eye_symmetry_score     = None

        if faces_detected > 0:
            eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            # Use the largest face for all per-face metrics
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            fx, fy, fw, fh = largest_face
            face_roi = gray[fy:fy+fh, fx:fx+fw]

            # Face-region lighting variance
            rh, rw = face_roi.shape
            if rh >= 4 and rw >= 4:
                mfh, mfw = rh // 2, rw // 2
                fq = [
                    float(np.mean(face_roi[:mfh, :mfw])),
                    float(np.mean(face_roi[:mfh, mfw:])),
                    float(np.mean(face_roi[mfh:, :mfw])),
                    float(np.mean(face_roi[mfh:, mfw:])),
                ]
                face_lighting_variance = round(float(np.std(fq)), 2)

            eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 5)
            eye_symmetry_score = int(len(eyes))

        # ── AI-generation specific metrics ─────────────────────────────
        # Modern AI generators (Midjourney/SDXL/Gemini) fool traditional
        # forensic checks but leave these measurable signatures:

        # 1. Saturation uniformity — AI artificially boosts saturation uniformly
        # Real photos: natural variation (sat_cv > 0.65). AI: low CV < 0.65
        hsv         = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat_channel = hsv[:, :, 1].astype(np.float32)
        sat_mean    = float(np.mean(sat_channel))
        sat_cv      = float(np.std(sat_channel) / (sat_mean + 1e-9))

        # 2. JPEG blockiness (proxy for compression quality)
        # AI outputs saved at 50-75% quality → blockiness ratio > 1.4
        block_diffs = []
        for bi in range(8, h - 8, 8):
            row_diff = float(np.mean(np.abs(
                gray[bi, :].astype(float) - gray[bi-1, :].astype(float)
            )))
            inter = float(np.mean(np.abs(
                gray[bi-1, :].astype(float) - gray[bi-2, :].astype(float)
            )))
            block_diffs.append(row_diff / (inter + 0.1))
        blockiness = float(np.mean(block_diffs)) if block_diffs else 1.0

        # 3. Tonal distribution (AI "cinematic" colour grading)
        shadow_pct    = float(np.sum(hist[:30]))  / (h * w)
        highlight_pct = float(np.sum(hist[220:])) / (h * w)

        # 4. Bytes per pixel
        bytes_per_pixel = float(os.path.getsize(image_path)) / (h * w + 1e-9)

        return {
            'blur_score':           round(blur_score,        2),
            'brightness':           round(brightness,        2),
            'texture_variance':     round(texture_variance,  2),
            'faces_detected':       faces_detected,
            'face_regions':         face_regions,
            'edge_density':         round(edge_density,      4),
            'color_consistency':    round(color_consistency, 2),
            'channel_imbalance':    round(channel_imbalance, 2),
            'compression_artifacts':round(high_freq_energy / 1000, 2),
            'freq_ratio':           freq_ratio,
            'noise_residual':       round(noise_residual,    3),
            'entropy':              round(entropy,           3),
            'skin_ratio':           round(skin_ratio,        4),
            'grad_std':             round(grad_std,          2),
            'lighting_variance':      round(lighting_variance, 2),
            'face_lighting_variance': face_lighting_variance,
            'eye_symmetry_score':     eye_symmetry_score,
            'file_size_mb':         round(os.path.getsize(image_path) / (1024*1024), 2),
            'resolution':           f'{w}x{h}',
            # AI-generation signals
            'sat_mean':             round(sat_mean,          2),
            'sat_cv':               round(sat_cv,            4),
            'blockiness':           round(blockiness,        4),
            'shadow_pct':           round(shadow_pct,        4),
            'highlight_pct':        round(highlight_pct,     4),
            'bytes_per_pixel':      round(bytes_per_pixel,   5),
        }
    except Exception as e:
        print(f'Quality analysis error: {e}')
        return None


def _heuristic_confidence(quality_metrics):
    """
    Forensic heuristic used when ML model is unavailable or outputs near-0.5.
    Scores suspiciousness 0-100; >=50 = fake/ai_generated.
    Uses all available forensic signals including AI-generation indicators.
    """
    import random
    if quality_metrics is None:
        return _demo_prediction()

    score = 0.0   # start at 0, build up evidence

    # ── AI-generation signals (strongest indicators) ──────────────────
    sat_cv          = quality_metrics.get('sat_cv', 1.0)
    sat_mean        = quality_metrics.get('sat_mean', 0)
    blockiness      = quality_metrics.get('blockiness', 1.0)
    bytes_per_pixel = quality_metrics.get('bytes_per_pixel', 1.0)

    if sat_cv < 0.50 and sat_mean > 75:
        score += 35   # very strong AI signal
    elif sat_cv < 0.60 and sat_mean > 85:
        score += 20   # moderate AI signal

    if blockiness > 1.6 and bytes_per_pixel < 0.08:
        score += 20   # AI compression pattern
    elif blockiness > 1.4 and bytes_per_pixel < 0.16:
        score += 8    # mild compression signal

    # ── Deepfake / manipulation signals ───────────────────────────────
    if quality_metrics.get('texture_variance', 999) < 20:
        score += 18
    elif quality_metrics.get('texture_variance', 999) < 35:
        score += 8

    if quality_metrics.get('noise_residual', 999) < 1.5:
        score += 15
    elif quality_metrics.get('noise_residual', 999) < 3.0:
        score += 6

    if quality_metrics.get('edge_density', 0) > 0.20:
        score += 12
    elif quality_metrics.get('edge_density', 0) > 0.15:
        score += 5

    lv = quality_metrics.get('face_lighting_variance') or quality_metrics.get('lighting_variance', 0)
    if lv > 45:
        score += 10
    elif lv > 35:
        score += 5

    if quality_metrics.get('freq_ratio', 0) > 0.18:
        score += 8
    if quality_metrics.get('channel_imbalance', 0) > 40:
        score += 8
    if quality_metrics.get('blur_score', 999) < 40:
        score += 8

    # Small random jitter ±4 to avoid exact repeats
    score += random.uniform(-4, 4)
    score  = max(0.0, min(100.0, score))

    if score >= 50:
        result     = 'fake'
        confidence = round(min(97.0, 50 + score * 0.47), 2)
    else:
        result     = 'real'
        confidence = round(max(55.0, min(92.0, 92 - score * 0.74)), 2)

    return result, confidence


# ── Clue icon map ─────────────────────────────────────────────────────────
_CLUE_ICONS = {
    'skin_texture':   '🧬',
    'boundary':       '🔲',
    'lighting':       '💡',
    'compression':    '📦',
    'noise':          '📡',
    'frequency':      '🌊',
    'eye_anomaly':    '👁️',
    'color_splice':   '🎨',
    'blur':           '🌫️',
    'entropy':        '🔀',
    'resolution':     '📐',
}


def run_forensic_analysis(quality_metrics, result, confidence):
    """
    Forensic clue engine.
    NOTE: face-count clues (no_face / multiple_faces) intentionally removed.
    Returns list of forensic clue dicts.
    """
    if not quality_metrics:
        return []

    clues   = []
    is_fake = (result == 'fake')

    # 1. Skin Texture
    tv = quality_metrics['texture_variance']
    if tv < 20:
        clues.append({
            'clue_type':   'skin_texture',
            'icon':        _CLUE_ICONS['skin_texture'],
            'severity':    'critical',
            'title':       'Abnormal Skin Texture',
            'description': 'Texture variance is extremely low — strong indicator of AI-generated '
                           'skin smoothing, characteristic of GAN-based deepfake models.',
            'evidence':    f'Texture variance: {tv:.1f} (threshold < 20)',
            'technical':   'GANs typically produce skin with unnaturally uniform pixel distributions.',
        })
    elif tv < 35 and is_fake:
        clues.append({
            'clue_type':   'skin_texture',
            'icon':        _CLUE_ICONS['skin_texture'],
            'severity':    'warning',
            'title':       'Suspicious Skin Smoothness',
            'description': 'Texture variance below normal range, suggesting possible AI skin '
                           'smoothing or heavy post-processing.',
            'evidence':    f'Texture variance: {tv:.1f} (normal range > 35)',
            'technical':   'Real facial photographs have measurable pore-level texture noise.',
        })

    # 2. Face Boundary Artifacts
    ed = quality_metrics['edge_density']
    if ed > 0.20 and is_fake:
        clues.append({
            'clue_type':   'boundary',
            'icon':        _CLUE_ICONS['boundary'],
            'severity':    'critical',
            'title':       'Face Boundary Blending Artifacts',
            'description': 'Abnormally high edge density at face boundaries indicates compositing '
                           'or face-swap seams where pasted regions do not blend seamlessly.',
            'evidence':    f'Edge density: {ed:.4f} (threshold > 0.20)',
            'technical':   'Face-swap algorithms often leave detectable high-frequency edges at the splice boundary.',
        })
    elif ed > 0.15 and is_fake:
        clues.append({
            'clue_type':   'boundary',
            'icon':        _CLUE_ICONS['boundary'],
            'severity':    'warning',
            'title':       'Possible Boundary Inconsistency',
            'description': 'Elevated edge density may indicate incomplete blending at face boundaries.',
            'evidence':    f'Edge density: {ed:.4f} (normal < 0.15)',
            'technical':   'Blending masks in deepfake pipelines rarely achieve perfect frequency matching.',
        })

    # 3. Lighting Mismatch — use face-region variance, not full-image
    # Full-image variance is naturally high for outdoor photos (sky vs ground)
    lv = quality_metrics.get('face_lighting_variance') or quality_metrics['lighting_variance']
    # Only flag if face-specific lighting mismatch (threshold raised accordingly)
    lv_threshold_critical = 45
    lv_threshold_warning  = 35
    if lv > lv_threshold_critical:
        clues.append({
            'clue_type':   'lighting',
            'icon':        _CLUE_ICONS['lighting'],
            'severity':    'critical' if is_fake else 'warning',
            'title':       'Inconsistent Face Illumination',
            'description': 'Significant brightness variation within the face region suggests '
                           'the face was lit differently from the background — a common sign '
                           'of a composited or face-swapped image.',
            'evidence':    f'Face lighting variance: {lv:.1f} (threshold > {lv_threshold_critical})',
            'technical':   'Real photographs show natural illumination falloff; composites often '
                           'show mismatched light direction or temperature.',
        })
    elif lv > lv_threshold_warning and is_fake:
        clues.append({
            'clue_type':   'lighting',
            'icon':        _CLUE_ICONS['lighting'],
            'severity':    'warning',
            'title':       'Mild Face Lighting Inconsistency',
            'description': 'Uneven illumination detected within face region — may indicate '
                           'imperfect relighting after face replacement.',
            'evidence':    f'Face lighting variance: {lv:.1f} (threshold > {lv_threshold_warning})',
            'technical':   'Many deepfake models lack 3D-aware relighting, causing subtle mismatches.',
        })

    # 4. Compression / Re-encoding
    ca = quality_metrics['compression_artifacts']
    fr = quality_metrics['freq_ratio']
    if fr > 0.18 and confidence > 70:
        clues.append({
            'clue_type':   'compression',
            'icon':        _CLUE_ICONS['compression'],
            'severity':    'warning',
            'title':       'JPEG Re-encoding Artifacts',
            'description': 'High-frequency DCT energy pattern suggests the image has been '
                           're-encoded multiple times — a trace left when manipulated images '
                           'are saved and re-saved.',
            'evidence':    f'High-freq energy ratio: {fr:.4f} (threshold > 0.18)',
            'technical':   'Each JPEG encode-decode cycle degrades different frequency bands.',
        })
    elif ca > 80:
        clues.append({
            'clue_type':   'compression',
            'icon':        _CLUE_ICONS['compression'],
            'severity':    'info',
            'title':       'Heavy Compression Detected',
            'description': 'Strong JPEG compression may mask or introduce visual artefacts '
                           'that affect detection accuracy.',
            'evidence':    f'Compression energy: {ca:.1f}',
            'technical':   'Lossy compression reduces detection reliability.',
        })

    # 5. Noise Residual
    nr = quality_metrics['noise_residual']
    if nr < 1.5 and is_fake:
        clues.append({
            'clue_type':   'noise',
            'icon':        _CLUE_ICONS['noise'],
            'severity':    'critical',
            'title':       'Unnatural Noise Pattern',
            'description': 'Extremely low noise residual — the image may have been synthetically '
                           'generated. Real camera sensors always introduce measurable photon noise.',
            'evidence':    f'Noise residual: {nr:.3f} (real images typically > 2.0)',
            'technical':   'GAN-generated images lack authentic camera sensor noise (PRNU) patterns.',
        })
    elif nr < 3.0 and is_fake:
        clues.append({
            'clue_type':   'noise',
            'icon':        _CLUE_ICONS['noise'],
            'severity':    'warning',
            'title':       'Low Sensor Noise',
            'description': 'Below-average noise residual — may indicate AI synthesis or '
                           'excessive denoising applied to hide manipulation traces.',
            'evidence':    f'Noise residual: {nr:.3f}',
            'technical':   'Photo Response Non-Uniformity (PRNU) analysis can confirm camera source.',
        })

    # 6. Colour Channel Splice
    ci = quality_metrics['channel_imbalance']
    if ci > 40 and is_fake:
        clues.append({
            'clue_type':   'color_splice',
            'icon':        _CLUE_ICONS['color_splice'],
            'severity':    'warning',
            'title':       'Colour Channel Imbalance',
            'description': 'Large spread between RGB channel standard deviations suggests '
                           'regions from different source images have been composited.',
            'evidence':    f'Channel imbalance: {ci:.1f} (threshold > 40)',
            'technical':   'Spliced images from different cameras show mismatched chromatic noise.',
        })

    # 7. Eye Anomaly (kept — it IS a face manipulation signal)
    eyes = quality_metrics.get('eye_symmetry_score')
    if eyes is not None:
        # Only flag eye anomaly when exactly 1 face detected (not 0, not 4)
        # Multiple face detections usually means background false positives
        if eyes == 0 and quality_metrics['faces_detected'] == 1:
            clues.append({
                'clue_type':   'eye_anomaly',
                'icon':        _CLUE_ICONS['eye_anomaly'],
                'severity':    'warning',
                'title':       'Eye Region Anomaly',
                'description': 'No eyes detected within the face region. Deepfake models '
                               'frequently distort or fail to reconstruct the periocular region.',
                'evidence':    f'Eyes detected in face region: {eyes} (expected 2)',
                'technical':   'Eye blink temporal patterns are common failure points for generative models.',
            })
        elif eyes == 1 and is_fake:
            clues.append({
                'clue_type':   'eye_anomaly',
                'icon':        _CLUE_ICONS['eye_anomaly'],
                'severity':    'info',
                'title':       'Asymmetric Eye Detection',
                'description': 'Only one eye detected — may indicate facial asymmetry from '
                               'deepfake generation or pose-dependent reconstruction failure.',
                'evidence':    f'Eyes detected: {eyes} of expected 2',
                'technical':   'Temporal inconsistency in eye regions is a key forensic marker.',
            })

    # 8. Frequency Suppression
    if fr < 0.05 and is_fake:
        clues.append({
            'clue_type':   'frequency',
            'icon':        _CLUE_ICONS['frequency'],
            'severity':    'warning',
            'title':       'Suppressed High-Frequency Detail',
            'description': 'Unusually low high-frequency energy suggests over-smoothing by a '
                           'generative model — real photographs retain fine texture in the '
                           'frequency domain.',
            'evidence':    f'High-freq energy ratio: {fr:.4f} (real images typically > 0.08)',
            'technical':   'Upsampling layers in GAN decoders often blur high-frequency bands.',
        })

    # 9. Blur Artifact
    bs = quality_metrics['blur_score']
    if bs < 40 and is_fake:
        clues.append({
            'clue_type':   'blur',
            'icon':        _CLUE_ICONS['blur'],
            'severity':    'warning',
            'title':       'Artificial Blurring',
            'description': 'Laplacian variance is very low, indicating uniform blurring '
                           'inconsistent with natural camera optics — likely applied to '
                           'conceal manipulation traces.',
            'evidence':    f'Blur score (Laplacian var): {bs:.1f} (threshold < 40)',
            'technical':   'Natural lens blur produces non-uniform bokeh; AI smoothing is spatially uniform.',
        })

    # ── AI-GENERATION SPECIFIC CLUES ─────────────────────────────────────
    # These clues fire regardless of deepfake verdict — they indicate
    # AI-synthesised content (Midjourney, SDXL, Gemini, DALL-E 3) rather
    # than camera-captured photography.

    # 10. Saturation Uniformity (most reliable AI-gen signal)
    sat_cv   = quality_metrics.get('sat_cv',   1.0)
    sat_mean = quality_metrics.get('sat_mean', 100)
    if sat_cv < 0.50 and sat_mean > 75:
        severity = 'critical' if sat_cv < 0.35 else 'warning'
        clues.append({
            'clue_type':   'ai_generated',
            'icon':        '🤖',
            'severity':    severity,
            'title':       'AI Saturation Signature',
            'description': 'Colour saturation is unnaturally high and uniform across the image. '
                           'Real photographs show natural saturation variation; AI generators '
                           'apply stylised colour enhancement producing a "too perfect" palette.',
            'evidence':    f'Saturation CV={sat_cv:.3f} (real photos > 0.65), mean={sat_mean:.0f}/255',
            'technical':   'Midjourney, SDXL and Gemini image models apply learned colour grading '
                           'that consistently produces low saturation coefficient of variation.',
        })

    # 11. Compression Quality Mismatch (AI images downloaded/shared at low quality)
    blockiness     = quality_metrics.get('blockiness',     1.0)
    bytes_per_pixel = quality_metrics.get('bytes_per_pixel', 0.5)
    if blockiness > 1.4 and bytes_per_pixel < 0.16:
        clues.append({
            'clue_type':   'ai_generated',
            'icon':        '🤖',
            'severity':    'warning',
            'title':       'AI Output Compression Pattern',
            'description': 'High JPEG blockiness combined with low file size per pixel suggests '
                           'the image was generated by an AI tool then saved at reduced quality — '
                           'a very common pattern for AI-generated images shared on social media.',
            'evidence':    f'Blockiness ratio={blockiness:.2f} (>1.4 suspicious), bytes/pixel={bytes_per_pixel:.4f} (<0.16)',
            'technical':   'AI generator outputs are typically saved at 70-80% JPEG quality before '
                           'distribution, and further compressed by WhatsApp/Telegram to ~50-60%.',
        })

    # 12. Cinematic Tonal Compression (AI "artistic" look)
    shadow_pct    = quality_metrics.get('shadow_pct',    0.1)
    highlight_pct = quality_metrics.get('highlight_pct', 0.1)
    if shadow_pct < 0.01 and highlight_pct < 0.02:
        clues.append({
            'clue_type':   'ai_generated',
            'icon':        '🤖',
            'severity':    'info',
            'title':       'Cinematic Tonal Range',
            'description': 'The image has almost no true shadows or highlights — the tonal range '
                           'is compressed into the midtones. This "cinematic" look is a hallmark '
                           'of AI image generators trained on professionally edited stock photos.',
            'evidence':    f'Shadow pixels: {shadow_pct*100:.1f}% (<1%), Highlights: {highlight_pct*100:.1f}% (<2%)',
            'technical':   'AI training datasets are biased toward well-exposed, post-processed images, '
                           'causing the model to reproduce compressed tonal distributions.',
        })

    return clues


def predict_image(image_path):
    start = time.time()
    quality_metrics = analyze_image_quality(image_path)

    # No model → heuristic fallback
    if ML_MODEL is None:
        r, c = _heuristic_confidence(quality_metrics)
        # Apply edited-image override
        r, c, reclassified = _classify_result(r, c, quality_metrics)
        risk_scores = compute_forensic_risk_scores(quality_metrics)
        artifacts   = run_forensic_analysis(quality_metrics, r, c) if quality_metrics else []
        return r, c, round(time.time() - start, 2), True, quality_metrics, artifacts, risk_scores

    # Model loaded → run inference
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # Apply face crop — matches training preprocessing (face-cropped 224x224 frames)
        # Without this, full-image resize produces predictions near 0.5 → heuristic fallback
        gray_inf = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        fc_inf   = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces_inf = fc_inf.detectMultiScale(gray_inf, 1.1, 4, minSize=(40, 40))
        if len(faces_inf) > 0:
            x, y, w, h  = max(faces_inf, key=lambda f: f[2]*f[3])
            pad_x = int(w * 0.30); pad_y = int(h * 0.30)
            x1 = max(0, x-pad_x); y1 = max(0, y-pad_y)
            x2 = min(img_bgr.shape[1], x+w+pad_x); y2 = min(img_bgr.shape[0], y+h+pad_y)
            img_crop = img_bgr[y1:y2, x1:x2]
        else:
            # No face detected — use centre crop (same fallback as training)
            h_i, w_i = img_bgr.shape[:2]
            margin   = min(h_i, w_i) // 4
            img_crop = img_bgr[margin:h_i-margin, margin:w_i-margin]

        img = cv2.resize(img_crop, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)   # raw 0-255 for EfficientNet internal rescaling
        img = np.expand_dims(img, 0)

        pred = float(ML_MODEL.predict(img, verbose=0)[0][0])
        print(f'🔍 DEBUG predict_image: raw pred={pred:.6f}, face_detected={len(faces_inf) > 0}')

        if _is_constant_output_model(pred):
            print(f'⚠  Model returned pred={pred:.4f} (near 0.5) — using heuristic fallback')
            r, c = _heuristic_confidence(quality_metrics)
            r, c, reclassified = _classify_result(r, c, quality_metrics)
            risk_scores = compute_forensic_risk_scores(quality_metrics)
            artifacts   = run_forensic_analysis(quality_metrics, r, c) if quality_metrics else []
            return r, c, round(time.time() - start, 2), True, quality_metrics, artifacts, risk_scores

        # Class mapping: fake=0, real=1 (alphabetical folder order)
        # pred close to 0.0 = confident FAKE
        # pred close to 1.0 = confident REAL
        if pred > 0.5:
            r, c = 'real', round(pred * 100, 2)
        else:
            r, c = 'fake', round((1.0 - pred) * 100, 2)

        # Apply edited-image override
        r, c, reclassified = _classify_result(r, c, quality_metrics)
        if reclassified:
            print(f'ℹ  Reclassified as authentic (edited image, no face manipulation signals)')

        risk_scores = compute_forensic_risk_scores(quality_metrics)
        artifacts   = run_forensic_analysis(quality_metrics, r, c) if quality_metrics else []
        return r, c, round(time.time() - start, 2), False, quality_metrics, artifacts, risk_scores

    except Exception as e:
        print(f'Image predict error: {e}')
        r, c = _heuristic_confidence(quality_metrics)
        r, c, _ = _classify_result(r, c, quality_metrics)
        risk_scores = compute_forensic_risk_scores(quality_metrics)
        artifacts   = run_forensic_analysis(quality_metrics, r, c) if quality_metrics else []
        return r, c, round(time.time() - start, 2), True, quality_metrics, artifacts, risk_scores


def predict_video(video_path, num_frames=10):
    # Video classification uses ONLY the ML model average across sampled frames.
    # _classify_result and sat_cv AI-image checks are intentionally NOT applied
    # to videos — those heuristics are tuned for still images and produce wrong
    # results on video frames (a deepfake video can look authentic frame-by-frame).
    # Forensic metrics are collected for INFO display only, never to override result.
    start           = time.time()
    quality_metrics = None
    risk_scores     = {}
    forensic_clues  = []

    if ML_MODEL is None:
        r, c = _demo_prediction()
        return r, c, round(time.time() - start, 2), True, quality_metrics, forensic_clues, risk_scores

    try:
        cap   = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        idxs  = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)

        preds        = []
        frame_images = []

        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            frame_images.append(frame.copy())
            # Apply face crop at inference — matches training preprocessing
            gray_f  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fc      = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces_f = fc.detectMultiScale(gray_f, 1.1, 4, minSize=(60, 60))
            if len(faces_f) > 0:
                x, y, w, h = max(faces_f, key=lambda f: f[2]*f[3])
                pad_x = int(w * 0.30); pad_y = int(h * 0.30)
                x1 = max(0, x-pad_x); y1 = max(0, y-pad_y)
                x2 = min(frame.shape[1], x+w+pad_x); y2 = min(frame.shape[0], y+h+pad_y)
                crop = frame[y1:y2, x1:x2]
            else:
                h_f, w_f = frame.shape[:2]
                margin = min(h_f, w_f) // 4
                crop = frame[margin:h_f-margin, margin:w_f-margin]
            resized = cv2.resize(crop, (224, 224))
            rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
            preds.append(float(ML_MODEL.predict(np.expand_dims(rgb, 0), verbose=0)[0][0]))

        cap.release()
        avg = float(np.mean(preds)) if preds else 0.5

        if _is_constant_output_model(avg):
            print(f'⚠  Video model output avg={avg:.4f} — using demo fallback')
            r, c    = _demo_prediction()
            is_demo = True
        else:
            # Threshold raised 0.5 → 0.60 for video.
            # Videos with inconsistent frame scores (some real, some fake frames)
            # average to ~50-55% and get wrongly labelled FAKE at threshold 0.5.
            # Real deepfake videos score consistently high (80%+).
            # Threshold 0.60 eliminates borderline false positives.
            # Class mapping: fake=0, real=1 (alphabetical)
            # avg close to 0.0 = fake, close to 1.0 = real
            VIDEO_THRESHOLD = 0.40   # below 0.4 = fake (was 0.60 when inverted)
            if avg < VIDEO_THRESHOLD:
                r, c = 'fake', round((1.0 - avg) * 100, 2)
            else:
                r, c = 'real', round(avg * 100, 2)
            is_demo = False

        # ── Collect forensic INFO on the most representative frame ──────────────
        # This is purely for displaying metrics/clues in the UI.
        # It does NOT change r or c — the ML verdict is final for video.
        if frame_images and preds:
            best_idx  = int(np.argmin(np.abs(np.array(preds) - avg)))
            rep_frame = frame_images[best_idx]

            import tempfile
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    tmp_path = tmp.name
                cv2.imwrite(tmp_path, rep_frame)

                quality_metrics = analyze_image_quality(tmp_path)

                if quality_metrics:
                    # Add video-specific fields
                    quality_metrics['video_fps']          = round(float(fps), 2)
                    quality_metrics['video_frame_count']  = total
                    quality_metrics['video_duration_sec'] = round(total / fps, 1) if fps else 0
                    quality_metrics['frames_analyzed']    = len(preds)
                    quality_metrics['frame_scores']       = [round(p * 100, 1) for p in preds]
                    quality_metrics['resolution']         = f'{w_vid}x{h_vid}'

                    # Forensic clues and risk scores use the already-decided r/c
                    # _classify_result is deliberately NOT called here for video
                    forensic_clues = run_forensic_analysis(quality_metrics, r, c)
                    risk_scores    = compute_forensic_risk_scores(quality_metrics)
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

        return r, c, round(time.time() - start, 2), is_demo, quality_metrics, forensic_clues, risk_scores

    except Exception as e:
        print(f'Video predict error: {e}')
        import traceback; traceback.print_exc()
        r, c = _demo_prediction()
        return r, c, round(time.time() - start, 2), True, quality_metrics, forensic_clues, risk_scores


# ── Upload single image ────────────────────────────────────────────────────
@detection_bp.route('/upload-image', methods=['POST'])
def upload_image():
    user = verify_token()
    if not user: return jsonify({'error': 'Unauthorized.'}), 401
    if 'file' not in request.files: return jsonify({'error': 'No file uploaded.'}), 400
    file = request.files['file']
    if not file.filename: return jsonify({'error': 'No file selected.'}), 400
    if not allowed_file(file, 'image'): return jsonify({'error': 'Invalid image file.'}), 400

    try:
        fp = save_upload_file(file, 'images')
        r, c, pt, demo, quality, forensic_clues, risk_scores = predict_image(fp)

        metadata = {
            'quality_metrics': quality,
            'forensic_clues':  forensic_clues,
            'risk_scores':     risk_scores,
        }

        det = Detection(
            user_id=user.id, file_name=file.filename, file_type='image',
            file_path=fp, result=r, confidence=c, processing_time=pt,
            is_demo=demo, extra_data=json.dumps(metadata)
        )
        db.session.add(det)
        db.session.commit()

        try:
            from email_service import send_detection_result_email
            send_detection_result_email(user.email, user.full_name, file.filename, r, c, det.id)
        except Exception:
            pass

        return jsonify({
            'message':         'Image analysed.',
            'result':          r,
            'confidence':      c,
            'processing_time': pt,
            'detection_id':    det.id,
            'is_demo':         demo,
            'quality_metrics': quality,
            'forensic_clues':  forensic_clues,
            'risk_scores':     risk_scores,
        }), 200

    except Exception as e:
        db.session.rollback()
        print(f'upload-image error: {e}')
        return jsonify({'error': 'Analysis failed.'}), 500


# ── Upload single video ────────────────────────────────────────────────────
@detection_bp.route('/upload-video', methods=['POST'])
def upload_video():
    user = verify_token()
    if not user: return jsonify({'error': 'Unauthorized.'}), 401
    if 'file' not in request.files: return jsonify({'error': 'No file uploaded.'}), 400
    file = request.files['file']
    if not file.filename: return jsonify({'error': 'No file selected.'}), 400
    if not allowed_file(file, 'video'): return jsonify({'error': 'Invalid video file.'}), 400

    try:
        fp = save_upload_file(file, 'videos')
        r, c, pt, demo, quality, artifacts, risk_scores = predict_video(fp)

        det = Detection(
            user_id=user.id, file_name=file.filename, file_type='video',
            file_path=fp, result=r, confidence=c, processing_time=pt, is_demo=demo
        )
        db.session.add(det)
        db.session.commit()

        try:
            from email_service import send_detection_result_email
            send_detection_result_email(user.email, user.full_name, file.filename, r, c, det.id)
        except Exception:
            pass

        metadata = {
            'quality_metrics': quality,
            'forensic_clues':  artifacts,
            'risk_scores':     risk_scores,
        }
        # Persist forensic data so history view can display it
        det.extra_data = json.dumps(metadata)
        db.session.commit()

        return jsonify({
            'message':         'Video analysed.',
            'result':          r,
            'confidence':      c,
            'processing_time': pt,
            'detection_id':    det.id,
            'is_demo':         demo,
            'quality_metrics': quality,
            'forensic_clues':  artifacts,
            'risk_scores':     risk_scores,
        }), 200

    except Exception as e:
        db.session.rollback()
        print(f'upload-video error: {e}')
        return jsonify({'error': 'Analysis failed.'}), 500


# ── Bulk upload ─────────────────────────────────────────────────────────────
@detection_bp.route('/upload-bulk', methods=['POST'])
def upload_bulk():
    user = verify_token()
    if not user: return jsonify({'error': 'Unauthorized.'}), 401
    files = request.files.getlist('files')
    if not files: return jsonify({'error': 'No files uploaded.'}), 400
    if len(files) > 10: return jsonify({'error': 'Maximum 10 files per batch.'}), 400

    results = []
    for file in files:
        if not file.filename: continue
        ftype = 'image' if file.content_type.startswith('image/') else 'video'
        if not allowed_file(file, ftype):
            results.append({'file_name': file.filename, 'error': 'Invalid file type.'})
            continue
        try:
            sub = 'images' if ftype == 'image' else 'videos'
            fp  = save_upload_file(file, sub)
            if ftype == 'image':
                r, c, pt, demo, quality, artifacts, risk_scores = predict_image(fp)
            else:
                r, c, pt, demo, quality, artifacts, risk_scores = predict_video(fp)

            det = Detection(
                user_id=user.id, file_name=file.filename, file_type=ftype,
                file_path=fp, result=r, confidence=c, processing_time=pt, is_demo=demo
            )
            db.session.add(det)
            db.session.flush()
            results.append({
                'file_name': file.filename, 'result': r, 'confidence': c,
                'processing_time': pt, 'detection_id': det.id, 'is_demo': demo
            })
        except Exception as e:
            results.append({'file_name': file.filename, 'error': str(e)})

    db.session.commit()
    return jsonify({'results': results, 'total': len(results)}), 200


# ── History ─────────────────────────────────────────────────────────────────
@detection_bp.route('/history', methods=['GET'])
def get_history():
    user = verify_token()
    if not user: return jsonify({'error': 'Unauthorized.'}), 401

    try:
        page     = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('limit', 20, type=int), 100)
        search   = request.args.get('search', '').strip()
        sort_by  = request.args.get('sort',   'date')
        order    = request.args.get('order',  'desc')
        ftype    = request.args.get('type',   'all')
        fresult  = request.args.get('result', 'all')

        q = Detection.query.filter_by(user_id=user.id)
        if ftype   != 'all': q = q.filter(Detection.file_type == ftype)
        if fresult != 'all': q = q.filter(Detection.result    == fresult)
        if search:           q = q.filter(Detection.file_name.ilike(f'%{search}%'))

        col_map = {'date': 'created_at', 'confidence': 'confidence', 'result': 'result'}
        col = getattr(Detection, col_map.get(sort_by, 'created_at'))
        q   = q.order_by(col.asc() if order == 'asc' else col.desc())

        pagination = q.paginate(page=page, per_page=per_page, error_out=False)
        return jsonify({
            'history':  [d.to_dict() for d in pagination.items],
            'total':    pagination.total,
            'page':     page,
            'per_page': per_page,
            'pages':    pagination.pages,
            'has_next': pagination.has_next,
            'has_prev': pagination.has_prev,
        }), 200

    except Exception as e:
        print(f'history error: {e}')
        return jsonify({'error': 'Failed to load history.'}), 500


# ── Export CSV ──────────────────────────────────────────────────────────────
@detection_bp.route('/export-csv', methods=['GET'])
def export_csv():
    user = verify_token()
    if not user: return jsonify({'error': 'Unauthorized.'}), 401

    try:
        detections = Detection.query.filter_by(user_id=user.id) \
                        .order_by(Detection.created_at.desc()).all()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['ID', 'File', 'Type', 'Result', 'Confidence', 'Time', 'Demo', 'Date'])
        for d in detections:
            writer.writerow([
                d.id, d.file_name, d.file_type, d.result,
                round(d.confidence, 2), round(d.processing_time, 2),
                'Yes' if d.is_demo else 'No',
                d.created_at.strftime('%Y-%m-%d %H:%M:%S') if d.created_at else ''
            ])
        output.seek(0)
        return Response(
            output.getvalue(), mimetype='text/csv',
            headers={'Content-Disposition': 'attachment;filename=detection_history.csv'}
        )
    except Exception as e:
        return jsonify({'error': 'Export failed.'}), 500


# ── Single detection ────────────────────────────────────────────────────────
@detection_bp.route('/detection/<int:detection_id>', methods=['GET'])
def get_detection(detection_id):
    user = verify_token()
    if not user: return jsonify({'error': 'Unauthorized.'}), 401
    det = Detection.query.filter_by(id=detection_id, user_id=user.id).first()
    if not det: return jsonify({'error': 'Detection not found.'}), 404
    return jsonify({'detection': det.to_dict()}), 200


# ── Delete detection ────────────────────────────────────────────────────────
@detection_bp.route('/detection/<int:detection_id>', methods=['DELETE'])
def delete_detection(detection_id):
    user = verify_token()
    if not user: return jsonify({'error': 'Unauthorized.'}), 401
    det = Detection.query.filter_by(id=detection_id, user_id=user.id).first()
    if not det: return jsonify({'error': 'Detection not found.'}), 404

    try:
        if det.file_path and os.path.exists(det.file_path):
            os.remove(det.file_path)
        db.session.delete(det)
        db.session.commit()
        return jsonify({'message': 'Deleted successfully.'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Delete failed.'}), 500


# ── Stats ───────────────────────────────────────────────────────────────────
@detection_bp.route('/stats', methods=['GET'])
def get_stats():
    user = verify_token()
    if not user: return jsonify({'error': 'Unauthorized.'}), 401

    try:
        all_d = Detection.query.filter_by(user_id=user.id).all()
        total = len(all_d)
        fake  = sum(1 for d in all_d if d.result == 'fake')
        avg_c = round(sum(d.confidence for d in all_d) / total, 2) if total else 0

        from datetime import timedelta
        weekly = []
        for i in range(6, -1, -1):
            day_start = datetime.utcnow().replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=i)
            day_end = day_start + timedelta(days=1)
            cnt = sum(1 for d in all_d
                      if d.created_at and day_start <= d.created_at < day_end)
            weekly.append({'date': day_start.strftime('%b %d'), 'count': cnt})

        return jsonify({'stats': {
            'total_detections': total,
            'fake_count':       fake,
            'real_count':       total - fake,
            'avg_confidence':   avg_c,
            'weekly_trend':     weekly,
        }}), 200

    except Exception as e:
        return jsonify({'error': 'Failed to load stats.'}), 500