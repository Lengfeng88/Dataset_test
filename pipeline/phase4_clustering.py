"""
Phase 4 — Clustering & Zero-Shot Labeling
==========================================
1. Build similarity graph from tracklet embeddings
2. Leiden community detection → cluster_ids
3. CLIP / SiGLIP zero-shot labeling per cluster
4. Unknown clusters get brand_unknown_NNN labels

Key design choices explained in comments.
"""

from __future__ import annotations
import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import numpy as np

from .models import Tracklet, Cluster, ClusterStatus, EntityType
from .config import PipelineConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Leiden clustering stub
# ─────────────────────────────────────────────────────────────────────────────

class LeidenClustererStub:
    """
    PRODUCTION:
        import leidenalg
        import igraph as ig

        # Build graph
        sources, targets, weights = [], [], []
        for i, j, sim in high_sim_pairs:
            sources.append(i); targets.append(j); weights.append(float(sim))
            # Same track_id → forced edge weight 1.0
            if tracklets[i].track_id == tracklets[j].track_id:
                weights[-1] = 1.0

        g = ig.Graph(n=len(tracklets), edges=list(zip(sources, targets)),
                     edge_attrs={'weight': weights})

        # Leiden with resolution γ
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights='weight',
            resolution_parameter=gamma,
            seed=42
        )
        labels = partition.membership

    WHY LEIDEN over DBSCAN:
    - DBSCAN doesn't accept edge weights → can't encode track_id prior
    - Leiden operates on a graph → same-track edges get weight=1.0
      which strongly encourages same-track nodes to end up in same cluster
    - Leiden guarantees well-connected communities (Louvain can leave
      disconnected nodes in the same cluster)
    - Resolution γ gives a single interpretable knob to tune granularity
    """

    def __init__(self, resolution: float = 0.72, sim_threshold: float = 0.75):
        self.resolution = resolution
        self.sim_threshold = sim_threshold

    def cluster(self, tracklets: List[Tracklet]) -> List[int]:
        """
        Returns list of cluster IDs (int), one per tracklet.
        Same index = same cluster.
        """
        if not tracklets:
            return []

        n = len(tracklets)
        vecs = np.stack([t.mean_embedding for t in tracklets])  # (N, D)

        # ── Build similarity matrix ───────────────────────────────────────
        # Cosine similarity (embeddings are already L2-normalized)
        sim_matrix = vecs @ vecs.T   # (N, N)

        # ── Simplified Leiden stub via greedy graph partitioning ──────────
        # In production, use the real leidenalg library above.
        # This stub: union-find with similarity + track_id constraints.
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Merge pairs above threshold
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i, j])
                # Force merge if same track_id
                same_track = (tracklets[i].track_id == tracklets[j].track_id and
                              tracklets[i].video_path == tracklets[j].video_path)
                if sim >= self.sim_threshold or same_track:
                    # Resolution: higher γ = finer clusters
                    threshold = self.sim_threshold + (self.resolution - 0.5) * 0.1
                    if sim >= threshold or same_track:
                        union(i, j)

        # Assign consecutive integer cluster IDs
        root_to_id: Dict[int, int] = {}
        labels = []
        for i in range(n):
            root = find(i)
            if root not in root_to_id:
                root_to_id[root] = len(root_to_id)
            labels.append(root_to_id[root])

        return labels


# ─────────────────────────────────────────────────────────────────────────────
# CLIP zero-shot labeling stub
# ─────────────────────────────────────────────────────────────────────────────

class CLIPLabelerStub:
    """Real CLIP ViT-L/14 zero-shot labeler."""

    KNOWN_BRANDS = [
        "nike", "adidas", "under_armour", "puma", "jordan",
        "reebok", "new_balance", "gatorade", "espn", "lululemon",
        "champion", "state_farm", "crypto.com", "betway",
        "tissot", "wells_fargo", "penn_medicine",
    ]

    KNOWN_FACES = [
        "lebron james", "stephen curry", "kevin durant",
        "joel embiid", "giannis antetokounmpo", "luka doncic",
    ]

    def __init__(self, known_threshold: float = 0.60):
        import torch
        import open_clip
        self.known_threshold = known_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        self._torch = torch
        import torch.nn.functional as F
        self._F = F

        # Pre-compute text embeddings
        brand_prompts = [f"a photo of a {b} logo" for b in self.KNOWN_BRANDS]
        face_prompts  = [f"a photo of {f}" for f in self.KNOWN_FACES]
        self.brand_feats = self._encode_texts(brand_prompts)
        self.face_feats  = self._encode_texts(face_prompts)
        print(f"[CLIP] Loaded ViT-L/14 on {self.device}")

    def _encode_texts(self, texts):
        tokens = self.tokenizer(texts).to(self.device)
        with self._torch.no_grad():
            feats = self.model.encode_text(tokens)
            feats = self._F.normalize(feats, dim=-1)
        return feats.cpu().numpy()

    def label_cluster(self, cluster: Cluster) -> Tuple[Optional[str], float]:
        import cv2
        from PIL import Image

        rep_crops = cluster.representative_crops(n=5)
        if not rep_crops:
            return None, 0.0

        # Encode representative crop images
        img_feats = []
        for ec in rep_crops:
            cap = cv2.VideoCapture(ec.crop.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, ec.crop.frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                continue
            x1 = max(0, int(ec.crop.bbox.x1))
            y1 = max(0, int(ec.crop.bbox.y1))
            x2 = min(frame.shape[1], int(ec.crop.bbox.x2))
            y2 = min(frame.shape[0], int(ec.crop.bbox.y2))
            crop_img = frame[y1:y2, x1:x2]
            if crop_img.size == 0:
                continue
            h, w = crop_img.shape[:2]
            if h < 32 or w < 32:
                crop_img = cv2.resize(crop_img, (max(w,32), max(h,32)))
            pil_img = Image.fromarray(
                cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            ).convert("RGB")
            tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            with self._torch.no_grad():
                feat = self.model.encode_image(tensor)
                feat = self._F.normalize(feat, dim=-1)
            img_feats.append(feat.cpu().numpy()[0])

        if not img_feats:
            return None, 0.0

        mean_feat = np.mean(img_feats, axis=0)
        mean_feat = mean_feat / (np.linalg.norm(mean_feat) + 1e-9)

        if cluster.entity_type == EntityType.LOGO:
            text_feats = self.brand_feats
            known_list = self.KNOWN_BRANDS
        else:
            text_feats = self.face_feats
            known_list = self.KNOWN_FACES

        sims = text_feats @ mean_feat
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        best_label = known_list[best_idx].replace(" ", "_")

        if best_score >= self.known_threshold:
            return best_label, best_score
        return None, best_score


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Stability Score
# ─────────────────────────────────────────────────────────────────────────────

def compute_tss(cluster: Cluster) -> float:
    """
    Temporal Stability Score — measures how consistent embeddings are
    within a cluster across time.

    TSS = 1 - (std of cosine similarities between consecutive tracklets)

    High TSS (~0.95+): stable entity across video → trustworthy cluster
    Low TSS (<0.80):   mixed entity or tracking errors → route to HITL

    PRODUCTION NOTE: also consider temporal ordering of tracklets
    (sort by first_frame_idx before computing consecutive similarities).
    """
    all_tracklets = cluster.tracklets
    if len(all_tracklets) < 2:
        return 1.0

    vecs = np.stack([t.mean_embedding for t in all_tracklets])
    # Pairwise cosine similarities (vecs are L2-normalized)
    sims = (vecs @ vecs.T)
    # Take upper triangle
    n = len(all_tracklets)
    upper = [sims[i, j] for i in range(n) for j in range(i+1, n)]
    if not upper:
        return 1.0
    tss = float(np.mean(upper))
    return float(np.clip(tss, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class ClusteringPhase:
    """
    Input:  List[Tracklet]
    Output: List[Cluster] with labels assigned where possible
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):
        self.cfg = config
        self.clusterer = LeidenClustererStub(
            resolution=config.leiden_resolution,
            sim_threshold=config.similarity_threshold,
        )
        self.labeler = CLIPLabelerStub(
            known_threshold=config.clip_known_threshold,
        )
        self._unknown_logo_counter = 0
        self._unknown_face_counter = 0

    def process(self, tracklets: List[Tracklet]) -> List[Cluster]:
        if not tracklets:
            return []

        # ── Step 1: Leiden clustering — split by entity type first ─────────
        # Logo embeddings are 1024-dim, face are 512-dim; cannot stack together.
        logo_t = [t for t in tracklets if t.entity_type == EntityType.LOGO]
        face_t = [t for t in tracklets if t.entity_type == EntityType.FACE]
        logo_labels = self.clusterer.cluster(logo_t) if logo_t else []
        offset = (max(logo_labels) + 1) if logo_labels else 0
        face_labels = [l + offset for l in self.clusterer.cluster(face_t)] if face_t else []
        tracklets = logo_t + face_t
        labels = logo_labels + face_labels
        n_clusters = max(labels) + 1 if labels else 0
        logger.info(f"[Phase4] Leiden found {n_clusters} clusters from {len(tracklets)} tracklets")

        # ── Step 2: Assemble Cluster objects ──────────────────────────────
        cluster_buckets: Dict[int, List[Tracklet]] = defaultdict(list)
        for tracklet, label in zip(tracklets, labels):
            cluster_buckets[label].append(tracklet)

        clusters: List[Cluster] = []
        for cluster_idx, members in cluster_buckets.items():
            entity_type = members[0].entity_type
            cluster = Cluster(
                cluster_id=f"cluster_{cluster_idx:04d}",
                entity_type=entity_type,
                tracklets=members,
            )
            clusters.append(cluster)

        # ── Step 3: CLIP zero-shot labeling ───────────────────────────────
        known = unknown = 0
        for cluster in clusters:
            label, score = self.labeler.label_cluster(cluster)
            cluster.clip_score = score
            cluster.tss = compute_tss(cluster)

            if label is not None and score >= self.cfg.clip_known_threshold:
                cluster.clip_label = label
                cluster.status = ClusterStatus.KNOWN
                known += 1
            else:
                # Unknown entity → assign placeholder ID
                if cluster.entity_type == EntityType.LOGO:
                    self._unknown_logo_counter += 1
                    cluster.cluster_id = f"brand_unknown_{self._unknown_logo_counter:03d}"
                else:
                    self._unknown_face_counter += 1
                    cluster.cluster_id = f"face_unknown_{self._unknown_face_counter:03d}"
                cluster.status = ClusterStatus.UNKNOWN
                unknown += 1

        logger.info(
            f"[Phase4] Labels — known: {known}, unknown: {unknown} "
            f"({unknown/(len(clusters)+1e-9)*100:.1f}% unlabeled)"
        )
        return clusters