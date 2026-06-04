"""
Phase 1 — Preprocessing
========================
TransNetV2  → shot boundary detection
Optical flow → keyframe extraction (τ = 8px, no model needed)
ByteTrack   → stable track_ids across frames
Grounding DINO → logo crops
RetinaFace  → face crops

In production: swap the stub classes for real model imports.
Stubs here produce deterministic synthetic data so the rest
of the pipeline can run end-to-end without GPUs.
"""

from __future__ import annotations
import uuid
import hashlib
import logging
from pathlib import Path
from typing import List, Iterator, Tuple
import numpy as np

from .models import Crop, EntityType, BoundingBox
from .config import PipelineConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model stubs — replace with real implementations
# ─────────────────────────────────────────────────────────────────────────────

class TransNetV2Stub:
    """
    PRODUCTION: from transnetv2 import TransNetV2; model = TransNetV2()
    Returns list of (start_frame, end_frame) scene boundaries.
    """
    def predict_scenes(self, video_path: str) -> List[Tuple[int, int]]:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=27.0))
        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()
        if not scene_list:
            # fallback: treat whole video as one scene
            import cv2
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return [(0, total)]
        return [(int(s.get_frames()), int(e.get_frames()))
                for s, e in scene_list]


class OpticalFlowKeyframeExtractor:
    """
    No model needed. Compute mean optical flow magnitude between consecutive
    frames. If magnitude > τ, the frame is a keyframe (significant motion).

    PRODUCTION:
        import cv2
        cap = cv2.VideoCapture(video_path)
        prev_gray = None
        while True:
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                        0.5, 3, 15, 3, 5, 1.2, 0)
                mag = np.mean(np.sqrt(flow[...,0]**2 + flow[...,1]**2))
                if mag > tau: yield frame_idx
            prev_gray = gray
    """
    def __init__(self, tau: float = 8.0):
        self.tau = tau

    def extract_keyframes(self, video_path: str,
                          scene: Tuple[int, int]) -> List[int]:
        rng = np.random.default_rng(hash(video_path + str(scene)) % 2**32)
        start, end = scene
        n_frames = end - start
        # ~30% of frames pass the flow threshold
        n_keyframes = max(1, int(n_frames * 0.30))
        keyframes = sorted(rng.choice(range(start, end), size=n_keyframes,
                                       replace=False).tolist())
        return keyframes


class ByteTrackStub:
    """IoU-based simple tracker — assigns stable track_ids across frames."""

    def __init__(self):
        self._id_counter = 0
        self._tracks: list = []  # list of {id, box, age}

    def update(self, frame_idx: int, boxes: List[BoundingBox]) -> List[int]:
        import numpy as np

        def iou(a: BoundingBox, b: BoundingBox) -> float:
            ix1 = max(a.x1, b.x1); iy1 = max(a.y1, b.y1)
            ix2 = min(a.x2, b.x2); iy2 = min(a.y2, b.y2)
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            union = a.width*a.height + b.width*b.height - inter
            return inter / (union + 1e-9)

        # Age out old tracks
        self._tracks = [t for t in self._tracks if t['age'] < 10]

        track_ids = []
        used = set()
        for box in boxes:
            best_iou, best_idx = 0.0, -1
            for i, t in enumerate(self._tracks):
                if i in used:
                    continue
                s = iou(box, t['box'])
                if s > best_iou:
                    best_iou, best_idx = s, i
            if best_iou >= 0.3 and best_idx >= 0:
                self._tracks[best_idx]['box'] = box
                self._tracks[best_idx]['age'] = 0
                track_ids.append(self._tracks[best_idx]['id'])
                used.add(best_idx)
            else:
                self._id_counter += 1
                self._tracks.append({'id': self._id_counter, 'box': box, 'age': 0})
                track_ids.append(self._id_counter)
        # Age remaining tracks
        for i, t in enumerate(self._tracks):
            if i not in used:
                t['age'] += 1
        return track_ids


class GroundingDINOStub:
    """Real Grounding DINO logo detector."""

    TEXT_PROMPT = (
        "nike logo . adidas logo . under armour logo . puma logo . "
        "jordan logo . reebok logo . new balance logo . gatorade logo . "
        "espn logo . lululemon logo . champion logo . brand logo . "
        "sports logo . company logo ."
    )

    def __init__(self):
        from groundingdino.util.inference import load_model
        import os
        cfg = ".venv/lib/python3.11/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        weights = "weights/groundingdino_swint_ogc.pth"
        self.model = load_model(cfg, weights)
        print("[GroundingDINO] Loaded SwinT model")
        self._frame_cache: dict = {}

    def _load_frame(self, video_path: str, frame_idx: int):
        import cv2
        from groundingdino.util.inference import load_image
        from PIL import Image
        import tempfile, os
        key = (video_path, frame_idx)
        if key in self._frame_cache:
            return self._frame_cache[key]
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None, None
        # Save to temp file for groundingdino
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(tmp.name, frame)
        img_src, img_tensor = load_image(tmp.name)
        os.unlink(tmp.name)
        result = (img_src, img_tensor)
        # Cache max 10 frames
        if len(self._frame_cache) > 10:
            self._frame_cache.pop(next(iter(self._frame_cache)))
        self._frame_cache[key] = result
        return result

    def detect(self, frame_idx: int, video_seed: int,
               threshold: float = 0.30,
               video_path: str = "") -> List[Tuple[BoundingBox, float, str]]:
        """Single-frame detect — delegates to detect_batch internally."""
        results = self.detect_batch([frame_idx], threshold, video_path)
        return results.get(frame_idx, [])

    def detect_batch(self, frame_indices: List[int],
                     threshold: float = 0.30,
                     video_path: str = "") -> dict:
        """
        Batch detect across multiple frames at once.
        Returns dict: frame_idx -> List[Tuple[BoundingBox, float, str]]
        """
        if not video_path or not frame_indices:
            return {}
        import torch
        import cv2
        from groundingdino.util.inference import predict
        

        # Read all frames in one cv2 VideoCapture pass
        cap = cv2.VideoCapture(video_path)
        frames = {}
        for idx in sorted(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()

        from groundingdino.util.inference import load_image
        import tempfile, os

        results = {}
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Process in batches of BATCH_SIZE
        BATCH_SIZE = 4
        idx_list = sorted(frames.keys())
        for i in range(0, len(idx_list), BATCH_SIZE):
            batch_indices = idx_list[i:i+BATCH_SIZE]
            for frame_idx in batch_indices:
                img = frames[frame_idx]
                h, w = img.shape[:2]
                # In-memory transform — no disk I/O
                from PIL import Image as _PIL
                import torchvision.transforms as _T
                _transform = _T.Compose([
                    _T.Resize(800),
                    _T.ToTensor(),
                    _T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
                ])
                pil_img = _PIL.fromarray(img)
                img_tensor = _transform(pil_img)
                try:
                    boxes, logits, phrases = predict(
                        model=self.model,
                        image=img_tensor,
                        caption=self.TEXT_PROMPT,
                        box_threshold=threshold,
                        text_threshold=0.25,
                        device=device,
                    )
                except Exception:
                    results[frame_idx] = []
                    continue
                frame_results = []
                for box, score, phrase in zip(boxes, logits, phrases):
                    cx, cy, bw, bh = box.tolist()
                    x1 = max(0, (cx - bw/2) * w)
                    y1 = max(0, (cy - bh/2) * h)
                    x2 = min(w, (cx + bw/2) * w)
                    y2 = min(h, (cy + bh/2) * h)
                    frame_results.append((BoundingBox(x1,y1,x2,y2), float(score), phrase))
                results[frame_idx] = frame_results
        return results


class RetinaFaceStub:
    """Real InsightFace RetinaFace detector."""

    def __init__(self):
        import insightface
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("[RetinaFace] Loaded InsightFace buffalo_l")
        self._frame_cache: dict = {}

    def _load_frame(self, video_path: str, frame_idx: int):
        import cv2
        key = (video_path, frame_idx)
        if key in self._frame_cache:
            return self._frame_cache[key]
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        if len(self._frame_cache) > 10:
            self._frame_cache.pop(next(iter(self._frame_cache)))
        self._frame_cache[key] = frame
        return frame

    def detect(self, frame_idx: int, video_seed: int,
               min_px: int = 24,
               video_path: str = "") -> List[Tuple[BoundingBox, float, str]]:
        if not video_path:
            return []
        frame = self._load_frame(video_path, frame_idx)
        if frame is None:
            return []
        try:
            faces = self.app.get(frame)
        except Exception:
            return []
        results = []
        for face in faces:
            x1,y1,x2,y2 = face.bbox.tolist()
            x1,y1,x2,y2 = max(0,x1),max(0,y1),max(0,x2),max(0,y2)
            box = BoundingBox(x1,y1,x2,y2)
            score = float(face.det_score)
            if box.width >= min_px and box.height >= min_px and score >= 0.5:
                results.append((box, score, "face"))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class PreprocessingPhase:
    """
    Input:  video file path
    Output: iterator of Crop objects with track_ids assigned

    Processing order:
    1. TransNetV2 splits video into scenes
    2. Optical flow extracts keyframes within each scene
    3. For each keyframe: run Grounding DINO + RetinaFace
    4. ByteTrack assigns track_ids to all detections
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):
        self.cfg = config
        self.scene_detector = TransNetV2Stub()
        self.keyframe_extractor = OpticalFlowKeyframeExtractor(config.optical_flow_tau)
        self.logo_detector = GroundingDINOStub()
        self.face_detector = RetinaFaceStub()
        self.tracker = ByteTrackStub()

    def process(self, video_path: str) -> Iterator[Crop]:
        """
        Main entry point. Yields Crop objects one at a time — memory-efficient
        for long videos (no need to load all frames at once).
        """
        video_seed = int(hashlib.md5(video_path.encode()).hexdigest()[:8], 16)
        logger.info(f"[Phase1] Processing: {video_path}")

        scenes = self.scene_detector.predict_scenes(video_path)
        logger.info(f"[Phase1] {len(scenes)} scenes detected")

        total_crops = 0
        total_keyframes = 0
        discarded_small = 0

        from tqdm import tqdm
        for scene_idx, scene in enumerate(tqdm(scenes, desc="Phase1 scenes")):
            keyframes = self.keyframe_extractor.extract_keyframes(video_path, scene)
            total_keyframes += len(keyframes)

            # Batch detect all keyframes in this scene at once
            logo_batch = self.logo_detector.detect_batch(
                keyframes, self.cfg.logo_det_threshold, video_path=video_path)

            for frame_idx in keyframes:
                # ── Detect logos ──────────────────────────────────────────
                logo_dets = logo_batch.get(frame_idx, [])
                if False:  # placeholder to keep structure
                    logo_dets = self.logo_detector.detect(
                        frame_idx, video_seed, self.cfg.logo_det_threshold, video_path=video_path)

                logo_boxes = [box for box, score, _ in logo_dets]
                logo_track_ids = self.tracker.update(frame_idx, logo_boxes)

                for (box, score, label), track_id in zip(logo_dets, logo_track_ids):
                    if box.width < self.cfg.min_logo_px:
                        discarded_small += 1
                        continue
                    crop = Crop(
                        crop_id=f"logo_{video_seed}_{frame_idx}_{track_id}",
                        video_path=video_path,
                        frame_idx=frame_idx,
                        timestamp_ms=frame_idx * 40.0,   # assume 25fps
                        track_id=track_id,
                        entity_type=EntityType.LOGO,
                        bbox=box,
                        det_score=score,
                    )
                    total_crops += 1
                    yield crop

                # ── Detect faces ──────────────────────────────────────────
                face_dets = self.face_detector.detect(
                    frame_idx, video_seed, self.cfg.min_face_px, video_path=video_path)

                face_boxes = [box for box, score, _ in face_dets]
                face_track_ids = self.tracker.update(frame_idx, face_boxes)

                for (box, score, face_id), track_id in zip(face_dets, face_track_ids):
                    crop = Crop(
                        crop_id=f"face_{video_seed}_{frame_idx}_{track_id}",
                        video_path=video_path,
                        frame_idx=frame_idx,
                        timestamp_ms=frame_idx * 40.0,
                        track_id=track_id + 10000,   # namespace faces away from logos
                        entity_type=EntityType.FACE,
                        bbox=box,
                        det_score=score,
                    )
                    total_crops += 1
                    yield crop

        logger.info(
            f"[Phase1] Done — keyframes: {total_keyframes}, "
            f"crops: {total_crops}, discarded (too small): {discarded_small}"
        )