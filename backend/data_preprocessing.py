"""
Data Preprocessing v2 – Celeb-DF Dataset
Extracts FACE-CROPPED frames from videos.
Face cropping is the single biggest improvement for deepfake detection —
the model sees faces instead of backgrounds/clothing.

Usage:
    python data_preprocessing_v2.py
    python data_preprocessing_v2.py --celebdf-dir ../Celeb-DF-v2 --frames 10
"""

import os
import cv2
import random
import argparse
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        print(f"{kwargs.get('desc','')}: processing {kwargs.get('total','?')} items...")
        return iterable

# Load face detector once globally
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def extract_face_frames(video_path: str, output_dir: str, num_frames: int = 10) -> int:
    """
    Extract face-cropped frames from a video.
    Falls back to full frame if no face detected (keeps data rather than losing it).
    """
    try:
        cap   = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0

        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return 0

        indices = (
            list(range(total)) if total < num_frames
            else [int(i * total / num_frames) for i in range(num_frames)]
        )

        stem  = Path(video_path).stem
        saved = 0

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1,
                                                  minNeighbors=4, minSize=(60, 60))

            if len(faces) > 0:
                # Use the largest face
                x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
                # Expand bounding box by 30% for context (forehead, chin, ears)
                pad_x = int(w * 0.30)
                pad_y = int(h * 0.30)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(frame.shape[1], x + w + pad_x)
                y2 = min(frame.shape[0], y + h + pad_y)
                crop = frame[y1:y2, x1:x2]
            else:
                # No face found — use centre crop (better than full frame)
                h_f, w_f = frame.shape[:2]
                margin = min(h_f, w_f) // 4
                crop = frame[margin:h_f-margin, margin:w_f-margin]

            # Resize to 224x224
            crop_resized = cv2.resize(crop, (224, 224))
            out_path = os.path.join(output_dir, f'{stem}_frame_{idx:06d}.jpg')
            cv2.imwrite(out_path, crop_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1

        cap.release()
        return saved

    except Exception as e:
        print(f'  Error: {video_path}: {e}')
        return 0


def collect_videos(directory: str) -> list:
    exts = {'.mp4', '.avi', '.mov', '.mkv'}
    directory = os.path.abspath(directory)   # fix Windows path resolution
    if not os.path.exists(directory):
        print(f'  ⚠  Directory not found: {directory}')
        return []
    videos = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if Path(f).suffix.lower() in exts
    ]
    print(f'  Found {len(videos)} videos in {os.path.basename(directory)}')
    return videos


def split(videos: list, train=0.70, val=0.15) -> dict:
    random.shuffle(videos)
    n     = len(videos)
    t_end = int(n * train)
    v_end = int(n * (train + val))
    return {
        'train':      videos[:t_end],
        'validation': videos[t_end:v_end],
        'test':       videos[v_end:],
    }


def process_dataset(celebdf_dir: str, output_dir: str, frames_per_video: int = 10):
    print('=' * 60)
    print('CELEB-DF PREPROCESSING v2 — FACE CROP MODE')
    print('=' * 60)

    real_videos = (
        collect_videos(os.path.join(celebdf_dir, 'Celeb-real')) +
        collect_videos(os.path.join(celebdf_dir, 'YouTube-real'))
    )
    fake_videos = collect_videos(os.path.join(celebdf_dir, 'Celeb-synthesis'))

    if not real_videos and not fake_videos:
        raise FileNotFoundError(f'No videos found in {celebdf_dir}')

    print(f'Real videos : {len(real_videos)}')
    print(f'Fake videos : {len(fake_videos)}')
    print(f'Frames/video: {frames_per_video}')
    print(f'Output dir  : {output_dir}')
    print()

    real_splits = split(real_videos)
    fake_splits = split(fake_videos)

    for sp in ('train', 'validation', 'test'):
        for cls in ('real', 'fake'):
            os.makedirs(os.path.join(output_dir, sp, cls), exist_ok=True)

    total_frames = 0

    for sp in ('train', 'validation', 'test'):
        for cls, video_list in (('real', real_splits[sp]), ('fake', fake_splits[sp])):
            out_dir = os.path.join(output_dir, sp, cls)
            desc    = f'{sp:>10} / {cls}'
            for vp in tqdm(video_list, desc=desc, total=len(video_list), unit='video'):
                total_frames += extract_face_frames(vp, out_dir, frames_per_video)

    print(f'\n✔ Done — {total_frames:,} face-cropped frames saved to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--celebdf-dir', default='../Celeb-DF-v2')
    parser.add_argument('--output-dir',  default='../processed_dataset')
    parser.add_argument('--frames',      type=int, default=10)
    args = parser.parse_args()

    if not os.path.exists(args.celebdf_dir):
        print(f'ERROR: Dataset not found at {args.celebdf_dir}')
        raise SystemExit(1)

    process_dataset(args.celebdf_dir, args.output_dir, args.frames)