"""
Phase 2 + 3 — Feature Extraction & Tracklet Aggregation
=========================================================
DINOv2 ViT-L/14  → 1024-dim logo embeddings
AdaFace IR-100    → 512-dim face embeddings (quality-adaptive)
Tracklet aggregation → mean embedding per track_id

Why this order matters:
- We embed individual crops first (Phase 2)
- Then group by track_id and average (Phase 3)
- The mean embedding of a 200-frame tracklet is far more stable
  than any single frame — this is the "free supervision" from video
"""

from __future__ import annotations
import logging
from collections import defaultdict
from typing import List, Dict, Iterator

import numpy as np

from .models import Crop, EmbeddedCrop, Tracklet, EntityType
from .config import PipelineConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding model stubs
# ─────────────────────────────────────────────────────────────────────────────

class DINOv2Stub:
    """Real DINOv2-large model for logo embeddings."""

    DIM = 1024

    def __init__(self):
        import torch
        import torch.nn.functional as F
        from transformers import AutoImageProcessor, AutoModel
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
        self.model = AutoModel.from_pretrained("facebook/dinov2-large").to(self.device).eval()
        self._F = F
        self._torch = torch
        print(f"[DINOv2] Loaded on {self.device}")

    def embed(self, crop: Crop) -> np.ndarray:
        import cv2
        from PIL import Image
        # Load crop image from video
        cap = cv2.VideoCapture(crop.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, crop.frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            # fallback to random if frame unreadable
            rng = np.random.default_rng(int(hash(crop.crop_id) % 2**31))
            vec = rng.standard_normal(self.DIM).astype(np.float32)
            return vec / (np.linalg.norm(vec) + 1e-9)
        x1 = max(0, int(crop.bbox.x1))
        y1 = max(0, int(crop.bbox.y1))
        x2 = min(frame.shape[1], int(crop.bbox.x2))
        y2 = min(frame.shape[0], int(crop.bbox.y2))
        crop_img = frame[y1:y2, x1:x2]
        if crop_img.size == 0:
            rng = np.random.default_rng(int(hash(crop.crop_id) % 2**31))
            vec = rng.standard_normal(self.DIM).astype(np.float32)
            return vec / (np.linalg.norm(vec) + 1e-9)
        # 确保 crop 足够大（最小 32x32）
        h, w = crop_img.shape[:2]
        if h < 32 or w < 32:
            crop_img = cv2.resize(crop_img, (max(w, 32), max(h, 32)))
        # BGR → RGB → PIL
        rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        if rgb.ndim == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(rgb).convert("RGB")
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        cls = outputs.last_hidden_state[:, 0]
        patches = outputs.last_hidden_state[:, 1:].mean(dim=1)
        feats = (cls + patches) / 2
        feats = self._F.normalize(feats, dim=-1)
        return feats.cpu().numpy()[0]


class AdaFaceStub:
    """
    PRODUCTION:
        import torch
        from face_alignment import align
        from net import build_model

        model = build_model('ir_100')
        model.load_state_dict(torch.load('adaface_ir100_ms1mv2.ckpt'))
        model.eval().cuda()

        def embed(aligned_face_img):
            # align first using 5-point keypoints from RetinaFace
            tensor = preprocess(aligned_face_img).unsqueeze(0).cuda()
            with torch.no_grad():
                feat, norm = model(tensor)
            return F.normalize(feat, dim=-1).cpu().numpy()[0]

    AdaFace key innovation: embedding norm encodes image quality.
    Low-quality image → low norm → margin shrinks → loss down-weights it.
    This prevents blurry/occluded frames from dominating the tracklet mean.
    """

    DIM = 512

    def embed(self, crop: Crop) -> tuple[np.ndarray, float]:
        """Returns (embedding, quality_score)."""
        seed = int(hash(crop.crop_id) % 2**31)
        rng = np.random.default_rng(seed)
        # Simulate quality: small boxes get lower quality
        quality = float(np.clip(
            (crop.bbox.width * crop.bbox.height) / (200 * 200), 0.1, 1.0
        ))
        vec = rng.standard_normal(self.DIM).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        return vec, quality


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: embed each crop
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingPhase:
    """
    Wraps each Crop with its embedding vector.
    Runs in batch for GPU efficiency (batch size configurable).
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG, batch_size: int = 64):
        self.cfg = config
        self.batch_size = batch_size
        self.logo_embedder = DINOv2Stub()
        self.face_embedder = AdaFaceStub()

    def process(self, crops: Iterator[Crop]) -> Iterator[EmbeddedCrop]:
        """
        PRODUCTION NOTE: accumulate into batches, call model.forward() once
        per batch (GPU efficiency). Here we call one at a time.
        """
        n = 0
        for crop in crops:
            if crop.entity_type == EntityType.LOGO:
                vec = self.logo_embedder.embed(crop)
                ec = EmbeddedCrop(crop=crop, embedding=vec)
            else:
                vec, quality = self.face_embedder.embed(crop)
                ec = EmbeddedCrop(crop=crop, embedding=vec, quality_score=quality)
            n += 1
            yield ec

        logger.info(f"[Phase2] Embedded {n} crops")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: aggregate crops into tracklets
# ─────────────────────────────────────────────────────────────────────────────

class TrackletAggregationPhase:
    """
    Groups EmbeddedCrops by (video_path, track_id) → Tracklet.
    Computes the mean embedding vector for each tracklet.

    This is the key step that converts noisy per-frame predictions
    into stable tracklet-level representations for clustering.
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):
        self.cfg = config

    def process(self, embedded_crops: Iterator[EmbeddedCrop]) -> List[Tracklet]:
        # Group by (video, track_id)
        buckets: Dict[tuple, Tracklet] = {}

        for ec in embedded_crops:
            key = (ec.crop.video_path, ec.crop.track_id)
            if key not in buckets:
                buckets[key] = Tracklet(
                    track_id=ec.crop.track_id,
                    entity_type=ec.crop.entity_type,
                    video_path=ec.crop.video_path,
                )
            buckets[key].crops.append(ec)

        tracklets = []
        for tracklet in buckets.values():
            tracklet.frame_count = len(tracklet.crops)
            tracklet.compute_mean_embedding()
            tracklets.append(tracklet)

        logo_count = sum(1 for t in tracklets if t.entity_type == EntityType.LOGO)
        face_count = sum(1 for t in tracklets if t.entity_type == EntityType.FACE)
        logger.info(
            f"[Phase3] {len(tracklets)} tracklets — "
            f"logos: {logo_count}, faces: {face_count}"
        )
        return tracklets