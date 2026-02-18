"""
Data Preprocessing – Celeb-DF Dataset
Extracts frames from videos and organises them for training.

Usage:
    python data_preprocessing.py
    python data_preprocessing.py --celebdf-dir ../Celeb-DF-v1 --output-dir ../processed_dataset --frames 10
"""

import os
import cv2
import random
import argparse
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    # Graceful fallback if tqdm is not installed
    def tqdm(iterable, **kwargs):
        total = kwargs.get('total', '?')
        desc  = kwargs.get('desc', '')
        print(f'{desc}: processing {total} items...')
        return iterable


def extract_frames(video_path: str, output_dir: str, num_frames: int = 10) -> int:
    """Extract `num_frames` evenly-spaced frames from a video. Returns count saved."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f'  ⚠  Cannot open: {video_path}')
            return 0

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return 0

        indices = (
            list(range(total))
            if total < num_frames
            else [int(i * total / num_frames) for i in range(num_frames)]
        )

        stem    = Path(video_path).stem
        saved   = 0

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                out_path = os.path.join(output_dir, f'{stem}_frame_{idx:06d}.jpg')
                cv2.imwrite(out_path, frame)
                saved += 1

        cap.release()
        return saved

    except Exception as e:
        print(f'  Error processing {video_path}: {e}')
        return 0


def collect_videos(directory: str) -> list[str]:
    exts = {'.mp4', '.avi', '.mov', '.mkv'}
    if not os.path.exists(directory):
        return []
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if Path(f).suffix.lower() in exts
    ]


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
    print('CELEB-DF PREPROCESSING')
    print('=' * 60)

    # Collect videos
    real_videos = (
        collect_videos(os.path.join(celebdf_dir, 'Celeb-real')) +
        collect_videos(os.path.join(celebdf_dir, 'YouTube-real'))
    )
    fake_videos = collect_videos(os.path.join(celebdf_dir, 'Celeb-synthesis'))

    if not real_videos and not fake_videos:
        raise FileNotFoundError(f'No videos found in {celebdf_dir}')

    print(f'Real videos found : {len(real_videos)}')
    print(f'Fake videos found : {len(fake_videos)}')

    real_splits = split(real_videos)
    fake_splits = split(fake_videos)

    # Create output dirs
    for sp in ('train', 'validation', 'test'):
        for cls in ('real', 'fake'):
            os.makedirs(os.path.join(output_dir, sp, cls), exist_ok=True)

    total_frames = 0

    for sp in ('train', 'validation', 'test'):
        for cls, video_list in (('real', real_splits[sp]), ('fake', fake_splits[sp])):
            out_dir = os.path.join(output_dir, sp, cls)
            desc    = f'{sp:>10} / {cls}'
            for vp in tqdm(video_list, desc=desc, total=len(video_list), unit='video'):
                total_frames += extract_frames(vp, out_dir, frames_per_video)

    print(f'\n✔ Preprocessing complete — {total_frames} frames extracted.')
    print(f'  Output: {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--celebdf-dir', default='../Celeb-DF-v1')
    parser.add_argument('--output-dir',  default='../processed_dataset')
    parser.add_argument('--frames',      type=int, default=10)
    args = parser.parse_args()

    if not os.path.exists(args.celebdf_dir):
        print(f'ERROR: Dataset not found at {args.celebdf_dir}')
        raise SystemExit(1)

    process_dataset(args.celebdf_dir, args.output_dir, args.frames)
