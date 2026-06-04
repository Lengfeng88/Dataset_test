"""
Phase cache — save/load Phase 1+2 results to disk.
Avoids re-running 25min detection+embedding on every code change.
"""
import pickle, hashlib, os
from pathlib import Path

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def cache_key(video_path: str) -> str:
    stat = os.stat(video_path)
    raw = f"{video_path}_{stat.st_size}_{stat.st_mtime}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]

def save_phase12(video_path: str, embedded_crops):
    key = cache_key(video_path)
    path = CACHE_DIR / f"{key}_phase12.pkl"
    with open(path, "wb") as f:
        pickle.dump(embedded_crops, f)
    print(f"[Cache] Saved Phase 1+2 → {path}")

def load_phase12(video_path: str):
    key = cache_key(video_path)
    path = CACHE_DIR / f"{key}_phase12.pkl"
    if path.exists():
        print(f"[Cache] Loading Phase 1+2 from {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    return None
