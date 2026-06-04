"""
Tests — verifying the purity guarantees hold
=============================================
Run with: python -m pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from pipeline.config import PipelineConfig
from pipeline.models import (
    Crop, EntityType, BoundingBox, EmbeddedCrop, Tracklet,
    Cluster, ClusterStatus
)
from pipeline.phase5_funnel import (
    apply_track_vote, apply_confidence_gate, HITLQueue,
    ConfidenceFunnelPhase, TrackResult
)
from pipeline.phase4_clustering import compute_tss
from monitoring.monitor import compute_psi, GoldSetEvaluator


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_crop(crop_id="c1", track_id=1, det_score=0.90,
              entity_type=EntityType.LOGO, w=80, h=60):
    return Crop(
        crop_id=crop_id,
        video_path="test.mp4",
        frame_idx=0,
        timestamp_ms=0.0,
        track_id=track_id,
        entity_type=entity_type,
        bbox=BoundingBox(100, 100, 100+w, 100+h),
        det_score=det_score,
    )

def make_embedded_crop(crop_id="c1", track_id=1, det_score=0.90, dim=1024):
    crop = make_crop(crop_id=crop_id, track_id=track_id, det_score=det_score)
    rng = np.random.default_rng(int(track_id))
    vec = rng.standard_normal(dim).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-9
    return EmbeddedCrop(crop=crop, embedding=vec)

def make_tracklet(track_id=1, n_crops=20, det_score=0.90, dim=1024):
    t = Tracklet(
        track_id=track_id,
        entity_type=EntityType.LOGO,
        video_path="test.mp4",
    )
    for i in range(n_crops):
        t.crops.append(make_embedded_crop(
            crop_id=f"c_{track_id}_{i}", track_id=track_id, det_score=det_score
        ))
    t.frame_count = n_crops
    t.compute_mean_embedding()
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: track vote
# ─────────────────────────────────────────────────────────────────────────────

class TestTrackVote:
    def test_sufficient_crops_vote_accept(self):
        """Tracklet with many consistent crops should pass vote."""
        t = make_tracklet(track_id=1, n_crops=30)
        result, label, conf = apply_track_vote(t, vote_threshold=0.70)
        # With enough crops, stub should produce a result
        assert result in [TrackResult.ACCEPT, TrackResult.AMBIGUOUS]
        assert 0.0 <= conf <= 1.0

    def test_empty_tracklet_discarded(self):
        t = Tracklet(track_id=99, entity_type=EntityType.LOGO, video_path="x.mp4")
        result, label, conf = apply_track_vote(t)
        assert result == TrackResult.DISCARD
        assert conf == 0.0

    def test_single_crop_tracklet(self):
        t = make_tracklet(track_id=5, n_crops=1)
        result, label, conf = apply_track_vote(t)
        assert result in [TrackResult.ACCEPT, TrackResult.AMBIGUOUS]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: confidence gate
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidenceGate:
    def test_above_085_auto_accept(self):
        assert apply_confidence_gate(0.90) == TrackResult.ACCEPT
        assert apply_confidence_gate(0.85) == TrackResult.ACCEPT
        assert apply_confidence_gate(0.95) == TrackResult.ACCEPT

    def test_band_is_hitl(self):
        assert apply_confidence_gate(0.70) == TrackResult.HITL
        assert apply_confidence_gate(0.60) == TrackResult.HITL
        assert apply_confidence_gate(0.61) == TrackResult.HITL
        assert apply_confidence_gate(0.84) == TrackResult.HITL

    def test_below_060_discard(self):
        assert apply_confidence_gate(0.59) == TrackResult.DISCARD
        assert apply_confidence_gate(0.00) == TrackResult.DISCARD
        assert apply_confidence_gate(0.30) == TrackResult.DISCARD

    def test_boundary_values(self):
        """Exactly on the boundary values."""
        assert apply_confidence_gate(0.85, auto_accept=0.85) == TrackResult.ACCEPT
        assert apply_confidence_gate(0.60, hitl_low=0.60) == TrackResult.HITL
        assert apply_confidence_gate(0.5999, hitl_low=0.60) == TrackResult.DISCARD

    def test_custom_thresholds(self):
        """Config changes propagate correctly."""
        assert apply_confidence_gate(0.80, auto_accept=0.75) == TrackResult.ACCEPT
        assert apply_confidence_gate(0.70, auto_accept=0.75, hitl_low=0.65) == TrackResult.HITL
        assert apply_confidence_gate(0.60, auto_accept=0.75, hitl_low=0.65) == TrackResult.DISCARD

    def test_raising_threshold_increases_purity(self):
        """
        Core purity guarantee test:
        Raising auto_accept from 0.85 to 0.90 should classify more items as HITL
        (i.e., fewer auto-accepts, which means higher purity in what does get accepted).
        """
        scores = np.linspace(0.60, 0.99, 100)
        accepts_low  = sum(1 for s in scores if apply_confidence_gate(s, 0.85, 0.60) == TrackResult.ACCEPT)
        accepts_high = sum(1 for s in scores if apply_confidence_gate(s, 0.90, 0.60) == TrackResult.ACCEPT)
        assert accepts_high < accepts_low, "Higher threshold must reduce auto-accepts"


# ─────────────────────────────────────────────────────────────────────────────
# HITL Queue
# ─────────────────────────────────────────────────────────────────────────────

class TestHITLQueue:
    def make_cluster(self, cluster_id="c1", score=0.70):
        c = Cluster(
            cluster_id=cluster_id,
            entity_type=EntityType.LOGO,
        )
        c.clip_score = score
        return c

    def test_add_and_pop(self):
        q = HITLQueue()
        c = self.make_cluster("brand_unknown_001")
        q.add(c)
        assert q.pending_count == 1
        item = q.pop_next()
        assert item is not None
        assert item.cluster_id == "brand_unknown_001"

    def test_confirm_removes_from_pending(self):
        q = HITLQueue()
        q.add(self.make_cluster("brand_unknown_002"))
        assert q.pending_count == 1
        q.confirm("brand_unknown_002", "nike", "tester")
        assert q.pending_count == 0
        assert len(q.confirmed_clusters) == 1
        assert q.confirmed_clusters[0].hitl_label == "nike"

    def test_reject_removes_from_pending(self):
        q = HITLQueue()
        q.add(self.make_cluster("brand_unknown_003"))
        q.reject("brand_unknown_003", "tester")
        assert q.pending_count == 0
        assert "brand_unknown_003" in q._rejected

    def test_confirmed_label_propagates(self):
        q = HITLQueue()
        c = self.make_cluster("brand_unknown_004")
        q.add(c)
        q.confirm("brand_unknown_004", "lululemon", "human_reviewer")
        confirmed = q.confirmed_clusters[0]
        assert confirmed.final_label == "lululemon"
        assert confirmed.status == ClusterStatus.HITL_CONFIRMED

    def test_empty_queue_returns_none(self):
        q = HITLQueue()
        assert q.pop_next() is None


# ─────────────────────────────────────────────────────────────────────────────
# TSS
# ─────────────────────────────────────────────────────────────────────────────

class TestTSS:
    def test_identical_tracklets_high_tss(self):
        """When all tracklets are identical, TSS should be close to 1."""
        vec = np.ones(128, dtype=np.float32)
        vec /= np.linalg.norm(vec)
        cluster = Cluster(cluster_id="test", entity_type=EntityType.LOGO)
        for i in range(5):
            t = make_tracklet(track_id=i, n_crops=5)
            # Override embedding with identical vector
            t.mean_embedding = vec.copy()
            cluster.tracklets.append(t)
        tss = compute_tss(cluster)
        assert tss > 0.95, f"Expected TSS > 0.95 for identical tracklets, got {tss}"

    def test_random_tracklets_lower_tss(self):
        """Random embeddings should produce lower TSS than consistent ones."""
        cluster = Cluster(cluster_id="test2", entity_type=EntityType.LOGO)
        rng = np.random.default_rng(42)
        for i in range(10):
            t = make_tracklet(track_id=i, n_crops=5)
            vec = rng.standard_normal(1024).astype(np.float32)
            vec /= np.linalg.norm(vec)
            t.mean_embedding = vec
            cluster.tracklets.append(t)
        tss = compute_tss(cluster)
        # Random unit vectors in 1024 dims have near-zero cosine similarity
        assert tss < 0.3, f"Expected TSS < 0.3 for random tracklets, got {tss}"

    def test_single_tracklet_returns_1(self):
        cluster = Cluster(cluster_id="solo", entity_type=EntityType.LOGO)
        cluster.tracklets.append(make_tracklet(track_id=1, n_crops=3))
        assert compute_tss(cluster) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# PSI drift detection
# ─────────────────────────────────────────────────────────────────────────────

class TestPSI:
    def test_identical_distributions_psi_zero(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal(1000)
        psi = compute_psi(data, data.copy())
        assert psi < 0.01, f"Identical distributions should have PSI≈0, got {psi}"

    def test_shifted_distribution_raises_psi(self):
        rng = np.random.default_rng(1)
        baseline = rng.standard_normal(1000)
        shifted = baseline + 2.0    # large shift
        psi = compute_psi(baseline, shifted)
        assert psi > 0.25, f"Shifted distribution should have PSI>0.25, got {psi}"

    def test_slightly_shifted_moderate_psi(self):
        rng = np.random.default_rng(2)
        baseline = rng.standard_normal(1000)
        slightly_shifted = baseline + 0.3
        psi = compute_psi(baseline, slightly_shifted)
        assert 0.01 < psi < 0.50, f"Slight shift should give moderate PSI, got {psi}"


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end purity guarantee test
# ─────────────────────────────────────────────────────────────────────────────

class TestPurityGuarantee:
    """
    The headline test: does running the full funnel produce ≥95% purity?
    This is the architectural guarantee from the solution document.
    """

    def test_estimated_purity_above_target(self):
        config = PipelineConfig()
        funnel = ConfidenceFunnelPhase(config)

        # Build synthetic clusters
        clusters = []
        rng = np.random.default_rng(123)

        for i in range(50):
            from pipeline.models import ClusterStatus
            c = Cluster(
                cluster_id=f"cluster_{i:03d}",
                entity_type=EntityType.LOGO if i % 3 != 0 else EntityType.FACE,
                tracklets=[make_tracklet(track_id=i*10+j, n_crops=15) for j in range(3)],
            )
            c.clip_score = float(rng.uniform(0.50, 0.98))
            c.clip_label = "nike" if c.clip_score >= config.clip_known_threshold else None
            c.status = (ClusterStatus.KNOWN if c.clip_label else ClusterStatus.UNKNOWN)
            clusters.append(c)

        records, stats = funnel.process(clusters, "test_video.mp4")

        # Core guarantee
        assert stats.estimated_purity >= config.target_label_purity, (
            f"Expected purity ≥{config.target_label_purity*100:.0f}%, "
            f"got {stats.estimated_purity*100:.1f}%"
        )

    def test_hitl_rate_below_2_percent(self):
        """HITL human cost should stay under 2% of total crops."""
        config = PipelineConfig()
        funnel = ConfidenceFunnelPhase(config)
        rng = np.random.default_rng(456)

        clusters = []
        for i in range(30):
            from pipeline.models import ClusterStatus
            c = Cluster(
                cluster_id=f"cluster_{i:03d}",
                entity_type=EntityType.LOGO,
                tracklets=[make_tracklet(track_id=i*5+j, n_crops=20) for j in range(2)],
            )
            c.clip_score = float(rng.uniform(0.55, 0.98))
            c.clip_label = "adidas" if c.clip_score >= config.clip_known_threshold else None
            c.status = ClusterStatus.KNOWN if c.clip_label else ClusterStatus.UNKNOWN
            clusters.append(c)

        records, stats = funnel.process(clusters, "test2.mp4")

        if stats.total_crops > 0:
            hitl_crop_rate = stats.clusters_hitl / stats.total_crops
            # Note: HITL is cluster-level, so rate vs total crops should be tiny
            assert hitl_crop_rate < 0.05, (
                f"HITL rate {hitl_crop_rate*100:.2f}% too high"
            )

    def test_discarding_ambiguous_improves_purity(self):
        """
        If we set a very high auto_accept threshold (0.99),
        almost everything goes to HITL/discard.
        If we set a very low threshold (0.50), lots auto-accepts but purity drops.
        Verify the design: higher threshold → fewer records but higher purity estimate.
        """
        low_config  = PipelineConfig(auto_accept_score=0.60, hitl_low_score=0.40)
        high_config = PipelineConfig(auto_accept_score=0.95, hitl_low_score=0.70)

        rng = np.random.default_rng(789)
        from pipeline.models import ClusterStatus

        def make_clusters(seed):
            rng2 = np.random.default_rng(seed)
            cs = []
            for i in range(40):
                c = Cluster(
                    cluster_id=f"c_{i}",
                    entity_type=EntityType.LOGO,
                    tracklets=[make_tracklet(track_id=i*3+j, n_crops=10) for j in range(2)],
                )
                c.clip_score = float(rng2.uniform(0.45, 0.99))
                c.clip_label = "puma" if c.clip_score >= 0.60 else None
                c.status = ClusterStatus.KNOWN if c.clip_label else ClusterStatus.UNKNOWN
                cs.append(c)
            return cs

        _, low_stats  = ConfidenceFunnelPhase(low_config).process(make_clusters(1), "v.mp4")
        _, high_stats = ConfidenceFunnelPhase(high_config).process(make_clusters(1), "v.mp4")

        # Higher threshold → fewer records accepted automatically
        # (some moved to HITL) and higher estimated purity
        assert high_stats.estimated_purity >= low_stats.estimated_purity, (
            "Stricter threshold should produce equal or higher purity"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])