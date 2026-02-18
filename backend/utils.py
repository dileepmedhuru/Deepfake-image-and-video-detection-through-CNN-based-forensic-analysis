from flask import request
from models import User
import jwt
from config import Config
import os
import imghdr
from werkzeug.utils import secure_filename
import uuid


# ---------- Auth helpers ----------

def verify_token():
    """Verify JWT token from Authorization header and return User or None."""
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '').strip()
        if not token:
            return None
        payload = jwt.decode(token, Config.JWT_SECRET_KEY, algorithms=['HS256'])
        return User.query.get(payload['user_id'])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


# ---------- File validation ----------

# Mapping of first-bytes signatures to MIME types (magic numbers)
_IMAGE_SIGNATURES = {
    b'\xff\xd8\xff':    'image/jpeg',
    b'\x89PNG\r\n':     'image/png',
    b'GIF87a':          'image/gif',
    b'GIF89a':          'image/gif',
    b'BM':              'image/bmp',
    b'RIFF':            'image/webp',   # refined below
}

_VIDEO_CONTAINER_EXTS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}


def _sniff_image_mime(file_stream) -> str | None:
    """Read the first 16 bytes to identify image type. Rewinds stream."""
    header = file_stream.read(16)
    file_stream.seek(0)
    for sig, mime in _IMAGE_SIGNATURES.items():
        if header.startswith(sig):
            # Distinguish WebP from other RIFF files
            if sig == b'RIFF' and header[8:12] == b'WEBP':
                return 'image/webp'
            elif sig == b'RIFF':
                return None
            return mime
    return None


def _sniff_video_mime(filename: str) -> str | None:
    """
    Video containers are complex binary formats; sniffing them reliably
    requires a library like python-magic. As a practical fallback we trust
    the extension ONLY after confirming the extension is in the safe set.
    """
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext in _VIDEO_CONTAINER_EXTS:
        return f'video/{ext}'
    return None


def allowed_file(file, file_type: str = 'image') -> bool:
    """
    Validate a FileStorage object by:
      1. Checking the filename extension is in the allowed set.
      2. For images: reading magic bytes from the stream to confirm real type.
      3. For videos: trusting the extension (container formats vary too much).
    """
    filename = file.filename or ''
    if '.' not in filename:
        return False

    ext = filename.rsplit('.', 1)[1].lower()

    if file_type == 'image':
        if ext not in Config.ALLOWED_IMAGE_EXTENSIONS:
            return False
        # Magic-byte check
        mime = _sniff_image_mime(file.stream)
        return mime in Config.ALLOWED_IMAGE_MIMES

    elif file_type == 'video':
        if ext not in Config.ALLOWED_VIDEO_EXTENSIONS:
            return False
        mime = _sniff_video_mime(filename)
        return mime is not None

    return False


# ---------- File saving ----------

def save_upload_file(file, subfolder: str = 'images') -> str:
    """Save an uploaded FileStorage to disk, return the saved path."""
    upload_dir = os.path.join(Config.UPLOAD_FOLDER, subfolder)
    os.makedirs(upload_dir, exist_ok=True)

    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    file_path = os.path.join(upload_dir, unique_filename)
    file.save(file_path)
    return file_path
