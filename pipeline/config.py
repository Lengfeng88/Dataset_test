"""
Pipeline configuration — every threshold in one place.
Change here, affects the whole system.
"""

from dataclasses import dataclass

@dataclass
class PipelineConfig:
    # ── Phase 1: Preprocessing ──────────────────────────────────────
    optical_flow_tau: float = 8.0          # px — keyframe extraction threshold
    min_face_px: int = 80                  # discard faces smaller than this
    min_logo_px: int = 32                  # discard logos smaller than this

    # ── Phase 2: Detection ──────────────────────────────────────────
    logo_det_threshold: float = 0.30       # Grounding DINO box score floor
    face_det_threshold: float = 0.90       # RetinaFace confidence floor

    # ── Phase 3: Embeddings ─────────────────────────────────────────
    logo_embed_dim: int = 1024             # DINOv2 ViT-L/14
    face_embed_dim: int = 512              # AdaFace IR-100

    # ── Phase 4: Clustering ─────────────────────────────────────────
    similarity_threshold: float = 0.75    # cosine sim edge threshold
    leiden_resolution: float = 0.72       # γ — tuned on validation set
    same_track_weight: float = 1.0        # edge weight for same track_id

    # ── Phase 5: Confidence Funnel ─────────────────────────────────
    track_vote_threshold: float = 0.70    # ≥70% frame agreement to assign label
    auto_accept_score: float = 0.20       # above → write to dataset immediately
    hitl_low_score: float = 0.13          # below → discard / quarantine
    # band 0.60–0.85 → HITL queue

    # ── Monitoring ──────────────────────────────────────────────────
    psi_warn_threshold: float = 0.25      # PSI population stability index
    psi_alert_threshold: float = 0.35
    gold_f1_min: float = 0.85             # per-class F1 floor
    gold_f1_drop_alert: float = 0.02      # 2pp drop triggers alert
    hitl_growth_alert: float = 0.20       # 20% week-over-week HITL growth
    samples_per_track_cap: int = 500      # prevent dominant-subject bias

    # ── CLIP labeling ────────────────────────────────────────────────
    clip_known_threshold: float = 0.15    # below → brand_unknown_NNN
    clip_ambiguous_threshold: float = 0.12

    # ── Continual learning ──────────────────────────────────────────
    pseudo_label_weight: float = 1.0
    hitl_label_weight: float = 2.0        # HITL labels count double
    fine_tune_interval_days: int = 7
    replay_buffer_size: int = 10000

    # ── Output ───────────────────────────────────────────────────────
    target_label_purity: float = 0.95


DEFAULT_CONFIG = PipelineConfig()