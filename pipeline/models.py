"""
Data models — typed containers that flow through the pipeline.
Every phase reads and writes these; nothing passes as raw dicts.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Tuple
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class EntityType(Enum):
    LOGO  = "logo"
    FACE  = "face"

class TrackResult(Enum):
    ACCEPT   = "auto_accept"     # score ≥ 0.85
    HITL     = "hitl"            # score 0.60–0.85
    DISCARD  = "discard"         # score < 0.60
    AMBIGUOUS = "ambiguous"      # track vote < 70%

class ClusterStatus(Enum):
    KNOWN     = "known"           # CLIP matched a known label
    UNKNOWN   = "unknown"         # CLIP score < threshold → brand_unknown_NNN
    HITL_PENDING = "hitl_pending"
    HITL_CONFIRMED = "hitl_confirmed"
    REJECTED  = "rejected"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 output
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class Crop:
    """
    One detected region from one frame.
    This is the atomic unit that flows through the pipeline.
    """
    crop_id:       str
    video_path:    str
    frame_idx:     int
    timestamp_ms:  float
    track_id:      int
    entity_type:   EntityType
    bbox:          BoundingBox
    det_score:     float            # raw detector confidence
    image_data:    Optional[np.ndarray] = None   # H×W×3 uint8


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2/3 output
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EmbeddedCrop:
    """Crop + its embedding vector."""
    crop:       Crop
    embedding:  np.ndarray          # L2-normalized, shape (D,)
    quality_score: float = 1.0      # AdaFace quality estimate for faces


@dataclass
class Tracklet:
    """
    All crops sharing a ByteTrack track_id, aggregated into one unit.
    This is what the clustering phase operates on.
    """
    track_id:      int
    entity_type:   EntityType
    video_path:    str
    crops:         List[EmbeddedCrop] = field(default_factory=list)

    # Computed during Phase 3 aggregation
    mean_embedding: Optional[np.ndarray] = None   # mean over all crops, L2-normed
    frame_count:    int = 0

    # Phase 5 track-vote result
    voted_label:    Optional[str]  = None
    vote_confidence: float         = 0.0
    vote_fraction:   float         = 0.0         # fraction of frames agreeing
    track_result:    Optional[TrackResult] = None

    def compute_mean_embedding(self) -> np.ndarray:
        """Average embeddings in the tracklet and re-normalize."""
        if not self.crops:
            dim = 1024 if self.entity_type.value == "logo" else 512
            self.mean_embedding = np.zeros(dim, dtype=np.float32)
            return self.mean_embedding
        expected_dim = self.crops[0].embedding.shape[0]
        valid_crops = [ec for ec in self.crops if ec.embedding.shape[0] == expected_dim]
        vecs = np.stack([ec.embedding for ec in valid_crops])
        mean = vecs.mean(axis=0)
        norm = np.linalg.norm(mean)
        self.mean_embedding = mean / norm if norm > 0 else mean
        return self.mean_embedding


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 output
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Cluster:
    """
    One Leiden community = one visual entity (brand or person).
    Human cost is O(clusters), not O(samples).
    """
    cluster_id:   str                  # e.g. "nike_001" or "brand_unknown_007"
    entity_type:  EntityType
    tracklets:    List[Tracklet]       = field(default_factory=list)

    # CLIP labeling
    clip_label:   Optional[str]        = None
    clip_score:   float                = 0.0
    status:       ClusterStatus        = ClusterStatus.HITL_PENDING

    # HITL
    hitl_label:   Optional[str]        = None
    reviewed_by:  Optional[str]        = None

    # Metrics
    tss:          float                = 0.0     # temporal stability score
    purity:       Optional[float]      = None    # if gold labels available

    @property
    def sample_count(self) -> int:
        return sum(len(t.crops) for t in self.tracklets)

    @property
    def final_label(self) -> Optional[str]:
        if self.hitl_label:
            return self.hitl_label
        if self.status == ClusterStatus.KNOWN:
            return self.clip_label
        return None

    def representative_crops(self, n: int = 5) -> List[EmbeddedCrop]:
        """Top-n crops closest to cluster centroid — shown to HITL reviewer."""
        if not self.tracklets:
            return []
        all_crops = [ec for t in self.tracklets for ec in t.crops]
        if len(all_crops) <= n:
            return all_crops
        # centroid
        vecs = np.stack([ec.embedding for ec in all_crops])
        centroid = vecs.mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-9)
        sims = vecs @ centroid
        top_idx = np.argsort(sims)[-n:][::-1]
        return [all_crops[i] for i in top_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 output — final dataset record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetRecord:
    """
    One row in the final labeled dataset.
    Only records that passed all funnel stages reach here.
    """
    record_id:      str
    cluster_id:     str
    label:          str
    entity_type:    EntityType
    video_path:     str
    frame_idx:      int
    timestamp_ms:   float
    track_id:       int
    bbox:           BoundingBox
    confidence:     float
    label_source:   str             # "auto_accept" | "hitl_confirmed"
    crop_path:      Optional[str]   = None    # saved image path


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline run summary
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunStats:
    video_path:           str
    total_crops:          int = 0
    after_track_vote:     int = 0
    after_conf_gate:      int = 0
    after_hitl:           int = 0
    final_dataset_records: int = 0
    clusters_total:       int = 0
    clusters_known:       int = 0
    clusters_unknown:     int = 0
    clusters_hitl:        int = 0
    discarded:            int = 0
    estimated_purity:     float = 0.0

    @property
    def pass_rate(self) -> float:
        if self.total_crops == 0:
            return 0.0
        return self.final_dataset_records / self.total_crops

    @property
    def hitl_rate(self) -> float:
        if self.total_crops == 0:
            return 0.0
        return self.clusters_hitl / max(self.total_crops, 1)