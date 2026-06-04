"""
Monitoring — How to Keep 95%+ Purity Over Time
================================================
Three monitoring layers at different timescales:

Layer 1: Gold set evaluation  (every model update)
    - 50–100 human-annotated samples per class
    - F1, precision, recall per class
    - Circuit-breaker: halt deployment if F1 drops >2pp

Layer 2: Online proxy metrics (daily)
    - Auto-accept rate trend (↓ = model degrading)
    - HITL queue volume growth (↑ = confidence distribution shifting)
    - Unknown cluster rate (↑ = new entities entering content)

Layer 3: Embedding drift detection (weekly)
    - PSI (Population Stability Index) on embedding distributions
    - PSI > 0.25 → warning; PSI > 0.35 → halt + retrain

Continual learning loop:
    Every 7 days: fine-tune last 2 layers on accumulated pseudo-label pool
    Before deployment: run gold set evaluation
    On regression: rollback to previous weights
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque

import numpy as np

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from pipeline.config import PipelineConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Gold set record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GoldSample:
    """One human-annotated sample in the gold test set."""
    sample_id:    str
    true_label:   str
    entity_type:  str
    difficulty:   str    # "easy" | "partial_occlusion" | "blur" | "small"
    # Predicted label and score filled in during evaluation
    pred_label:   Optional[str]  = None
    pred_score:   float          = 0.0

@dataclass
class ClassMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp + 1e-9)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn + 1e-9)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: Gold Set Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class GoldSetEvaluator:
    """
    Maintains a gold test set and runs F1 evaluation after each model update.

    PRODUCTION:
    - Gold set: 50–100 human-annotated crops per class
    - Covers: frontal, partial occlusion, motion blur, small scale, multiple orientations
    - Run after: embedding fine-tune, threshold change, prompt library update
    - Alert condition: any per-class F1 drops more than 2pp between runs
    - Circuit-breaker: global F1 < 0.85 → halt deployment, rollback
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):
        self.cfg = config
        self.gold_set: List[GoldSample] = self._build_synthetic_gold_set()
        self._run_history: List[Dict] = []

    def _build_synthetic_gold_set(self) -> List[GoldSample]:
        """
        In production: load from a human-curated annotation file.
        Here: synthetic gold set for demonstration.
        """
        samples = []
        classes = {
            "logo": ["nike", "adidas", "under_armour", "puma", "jordan",
                     "reebok", "new_balance", "gatorade", "espn"],
            "face": ["lebron_james", "stephen_curry", "kevin_durant"],
        }
        difficulties = ["easy", "partial_occlusion", "blur", "small"]
        rng = np.random.default_rng(42)

        for entity_type, labels in classes.items():
            for label in labels:
                # 50 samples per class
                for i in range(50):
                    samples.append(GoldSample(
                        sample_id=f"gold_{label}_{i:03d}",
                        true_label=label,
                        entity_type=entity_type,
                        difficulty=difficulties[rng.integers(0, len(difficulties))],
                    ))
        return samples

    def evaluate(self, model_version: str = "current") -> Dict:
        """
        Run evaluation and return per-class F1 + global metrics.
        Fires alert if any class drops >2pp from last run.
        """
        rng = np.random.default_rng(hash(model_version) % 2**31)

        # Simulate predictions: ~92–96% accuracy on gold set
        class_metrics: Dict[str, ClassMetrics] = defaultdict(ClassMetrics)
        for sample in self.gold_set:
            # Simulate harder predictions for blur/small
            base_acc = 0.94
            if sample.difficulty == "blur":        base_acc = 0.89
            elif sample.difficulty == "small":     base_acc = 0.87
            elif sample.difficulty == "partial_occlusion": base_acc = 0.91

            correct = rng.random() < base_acc
            other_labels = [s.true_label for s in self.gold_set if s.true_label != sample.true_label]
            pred = sample.true_label if correct else rng.choice(other_labels)
            sample.pred_label = pred
            sample.pred_score = float(rng.uniform(0.7, 0.99) if correct else rng.uniform(0.3, 0.7))

            m = class_metrics[sample.true_label]
            if correct:
                m.tp += 1
            else:
                m.fp += 1
                class_metrics[pred].fn += 1

        results = {}
        f1_scores = []
        alerts = []

        for label, m in class_metrics.items():
            f1 = m.f1
            f1_scores.append(f1)
            results[label] = {
                "f1": round(f1, 3),
                "precision": round(m.precision, 3),
                "recall": round(m.recall, 3),
                "tp": m.tp, "fp": m.fp, "fn": m.fn,
            }
            # Per-class floor check
            if f1 < self.cfg.gold_f1_min:
                alerts.append(f"ALERT: {label} F1={f1:.3f} < threshold {self.cfg.gold_f1_min}")
                logger.warning(alerts[-1])

        global_f1 = float(np.mean(f1_scores))
        results["_global"] = {
            "f1": round(global_f1, 3),
            "model_version": model_version,
        }

        # Delta check against last run
        if self._run_history:
            last = self._run_history[-1]
            for label in class_metrics:
                if label in last:
                    delta = results[label]["f1"] - last[label]["f1"]
                    if delta < -self.cfg.gold_f1_drop_alert:
                        alerts.append(
                            f"REGRESSION: {label} F1 dropped {delta*100:.1f}pp"
                        )

        results["_alerts"] = alerts
        self._run_history.append(results)
        logger.info(f"[Monitor-L1] Gold set eval: global F1={global_f1:.3f}, alerts={len(alerts)}")
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: Online Proxy Metrics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DailySnapshot:
    date_str:          str
    total_crops:       int
    auto_accept:       int
    hitl_routed:       int
    discarded:         int
    unknown_clusters:  int
    total_clusters:    int

    @property
    def auto_accept_rate(self) -> float:
        return self.auto_accept / (self.total_crops + 1e-9)

    @property
    def hitl_rate(self) -> float:
        return self.hitl_routed / (self.total_crops + 1e-9)

    @property
    def unknown_cluster_rate(self) -> float:
        return self.unknown_clusters / (self.total_clusters + 1e-9)


class OnlineProxyMonitor:
    """
    Watches daily operational metrics as leading indicators.

    If HITL volume grows >20% WoW: model confidence distribution is shifting.
    If auto-accept rate drops >5pp: embedding quality degrading.
    If unknown cluster rate grows: new entities entering content.

    These fire earlier than gold set F1 regression because they're computed
    on every video batch, not just on the periodic gold set eval.
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):
        self.cfg = config
        self._history: deque = deque(maxlen=90)  # 90 days rolling

    def record(self, snapshot: DailySnapshot):
        self._history.append(snapshot)

    def check_alerts(self) -> List[str]:
        alerts = []
        if len(self._history) < 8:
            return alerts

        recent = list(self._history)

        # ── HITL growth ≥20% WoW ─────────────────────────────────────────
        this_week  = sum(s.hitl_routed for s in recent[-7:])
        last_week  = sum(s.hitl_routed for s in recent[-14:-7])
        if last_week > 0:
            hitl_growth = (this_week - last_week) / last_week
            if hitl_growth >= self.cfg.hitl_growth_alert:
                alerts.append(
                    f"HITL queue grew {hitl_growth*100:.0f}% WoW "
                    f"({last_week} → {this_week}) — model confidence shifting"
                )

        # ── Auto-accept rate drop ─────────────────────────────────────────
        recent_aa = np.mean([s.auto_accept_rate for s in recent[-7:]])
        baseline_aa = np.mean([s.auto_accept_rate for s in recent[-30:-7]])
        if baseline_aa > 0 and (baseline_aa - recent_aa) > 0.05:
            alerts.append(
                f"Auto-accept rate dropped {(baseline_aa-recent_aa)*100:.1f}pp "
                f"(baseline {baseline_aa*100:.1f}% → recent {recent_aa*100:.1f}%)"
            )

        return alerts

    def summary(self) -> Dict:
        if not self._history:
            return {}
        recent = list(self._history)[-7:]
        return {
            "avg_auto_accept_rate": round(np.mean([s.auto_accept_rate for s in recent]), 3),
            "avg_hitl_rate":        round(np.mean([s.hitl_rate for s in recent]), 4),
            "avg_unknown_rate":     round(np.mean([s.unknown_cluster_rate for s in recent]), 3),
            "days_tracked":         len(self._history),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3: PSI Drift Detection
# ─────────────────────────────────────────────────────────────────────────────

def compute_psi(expected: np.ndarray, actual: np.ndarray,
                n_buckets: int = 10) -> float:
    """
    Population Stability Index.
    Measures how much an embedding distribution has shifted from baseline.

    PSI interpretation:
        < 0.10  → no shift (OK)
        0.10–0.25 → moderate shift (monitor)
        > 0.25  → significant shift (alert, investigate retraining)
        > 0.35  → severe shift (halt, rollback)

    PRODUCTION:
    - Project high-dim embeddings to 1D using PCA first component
    - Or compute PSI on each dimension and take mean
    - Baseline: first 2 weeks of production data
    - Compare: rolling 1-week window vs baseline
    """
    # Project to 1D for PSI (use PCA in production)
    if expected.ndim > 1:
        expected = expected.mean(axis=1)
    if actual.ndim > 1:
        actual = actual.mean(axis=1)

    bins = np.percentile(expected, np.linspace(0, 100, n_buckets + 1))
    bins[0] -= 1e-9
    bins[-1] += 1e-9

    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual, bins=bins)

    exp_pct = exp_counts / (exp_counts.sum() + 1e-9)
    act_pct = act_counts / (act_counts.sum() + 1e-9)

    # Clip to avoid log(0)
    exp_pct = np.clip(exp_pct, 1e-9, None)
    act_pct = np.clip(act_pct, 1e-9, None)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


class EmbeddingDriftMonitor:
    """
    Weekly PSI check on logo and face embedding distributions.
    Compares current week against baseline (first 2 weeks of prod data).
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):
        self.cfg = config
        self._baseline_logo: Optional[np.ndarray] = None
        self._baseline_face: Optional[np.ndarray] = None
        self._weeks_seen = 0

    def record_weekly(self, logo_embeddings: np.ndarray,
                      face_embeddings: np.ndarray) -> Dict:
        self._weeks_seen += 1

        # First 2 weeks: build baseline
        if self._weeks_seen <= 2:
            self._baseline_logo = logo_embeddings
            self._baseline_face = face_embeddings
            return {"status": "building_baseline", "weeks": self._weeks_seen}

        alerts = []
        results = {}

        if self._baseline_logo is not None and len(logo_embeddings) > 0:
            psi_logo = compute_psi(self._baseline_logo, logo_embeddings)
            results["psi_logo"] = round(psi_logo, 3)
            if psi_logo > self.cfg.psi_alert_threshold:
                alerts.append(f"SEVERE logo drift PSI={psi_logo:.3f} — halt + retrain")
            elif psi_logo > self.cfg.psi_warn_threshold:
                alerts.append(f"Logo embedding drift PSI={psi_logo:.3f} — schedule fine-tune")

        if self._baseline_face is not None and len(face_embeddings) > 0:
            psi_face = compute_psi(self._baseline_face, face_embeddings)
            results["psi_face"] = round(psi_face, 3)
            if psi_face > self.cfg.psi_alert_threshold:
                alerts.append(f"SEVERE face drift PSI={psi_face:.3f} — halt + retrain")
            elif psi_face > self.cfg.psi_warn_threshold:
                alerts.append(f"Face embedding drift PSI={psi_face:.3f} — schedule fine-tune")

        results["alerts"] = alerts
        for a in alerts:
            logger.warning(f"[Monitor-L3] {a}")
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Continual Learning Loop
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PseudoLabelSample:
    embedding:   np.ndarray
    label:       str
    confidence:  float
    weight:      float   # 1.0 for auto-accept, 2.0 for HITL-confirmed


class ContinualLearningLoop:
    """
    Every 7 days: fine-tune the last 2 layers of DINOv2 / AdaFace
    on the accumulated pseudo-label pool.

    Key safeguards:
    1. HITL-confirmed labels have 2× sample weight (they're ground truth)
    2. Auto-accept labels only enter pool if score > 0.85
    3. Replay buffer: keep old-brand samples to prevent catastrophic forgetting
    4. Gold set evaluation before deploying new weights
    5. Rollback on regression

    PRODUCTION fine-tuning code (simplified):
        optimizer = torch.optim.AdamW(model.last_two_layers.parameters(), lr=1e-4)
        for epoch in range(5):
            for batch in DataLoader(pseudo_pool, batch_size=32):
                embeddings, labels, weights = batch
                loss = weighted_cross_entropy(model(embeddings), labels, weights)
                loss.backward()
                optimizer.step()
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):
        self.cfg = config
        self._pool: List[PseudoLabelSample] = []
        self._replay_buffer: List[PseudoLabelSample] = []

    def add_auto_accept(self, embedding: np.ndarray, label: str, score: float):
        if score >= self.cfg.auto_accept_score:
            self._pool.append(PseudoLabelSample(
                embedding=embedding, label=label, confidence=score,
                weight=self.cfg.pseudo_label_weight,
            ))

    def add_hitl_confirmed(self, embedding: np.ndarray, label: str):
        """HITL-confirmed labels: 2× weight, always included."""
        self._pool.append(PseudoLabelSample(
            embedding=embedding, label=label, confidence=1.0,
            weight=self.cfg.hitl_label_weight,
        ))

    def should_fine_tune(self, days_since_last: int) -> bool:
        return days_since_last >= self.cfg.fine_tune_interval_days

    def fine_tune(self) -> Dict:
        """
        Trigger weekly fine-tuning job.
        Returns summary of what was trained on.
        """
        total = len(self._pool)
        if total == 0:
            return {"status": "skipped", "reason": "empty pool"}

        label_counts: Dict[str, int] = defaultdict(int)
        for s in self._pool:
            label_counts[s.label] += 1

        hitl_count = sum(1 for s in self._pool if s.weight > 1.0)
        auto_count = total - hitl_count

        # Move current pool to replay buffer (prevent forgetting)
        self._replay_buffer.extend(self._pool[:self.cfg.replay_buffer_size])
        if len(self._replay_buffer) > self.cfg.replay_buffer_size:
            self._replay_buffer = self._replay_buffer[-self.cfg.replay_buffer_size:]
        self._pool.clear()

        logger.info(
            f"[ContinualLearning] Fine-tune job: {total} samples, "
            f"HITL: {hitl_count}, auto: {auto_count}"
        )
        return {
            "status": "completed",
            "total_samples": total,
            "hitl_confirmed": hitl_count,
            "auto_accept": auto_count,
            "label_distribution": dict(label_counts),
            "replay_buffer_size": len(self._replay_buffer),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Unified monitoring facade
# ─────────────────────────────────────────────────────────────────────────────

class PipelineMonitor:
    """
    Single interface wrapping all three monitoring layers.
    Attach to the pipeline and call .record() after each batch.
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):
        self.cfg = config
        self.gold_evaluator = GoldSetEvaluator(config)
        self.online_monitor = OnlineProxyMonitor(config)
        self.drift_monitor  = EmbeddingDriftMonitor(config)
        self.continual_loop = ContinualLearningLoop(config)

    def record_batch(self, snapshot: DailySnapshot,
                     logo_embeddings: Optional[np.ndarray] = None,
                     face_embeddings: Optional[np.ndarray] = None):
        self.online_monitor.record(snapshot)
        online_alerts = self.online_monitor.check_alerts()
        if online_alerts:
            for a in online_alerts:
                logger.warning(f"[Monitor-L2] {a}")

    def run_weekly_drift_check(self, logo_embeddings: np.ndarray,
                               face_embeddings: np.ndarray) -> Dict:
        return self.drift_monitor.record_weekly(logo_embeddings, face_embeddings)

    def run_gold_eval(self, model_version: str = "current") -> Dict:
        return self.gold_evaluator.evaluate(model_version)

    def summary_report(self) -> Dict:
        return {
            "online_metrics": self.online_monitor.summary(),
            "continual_learning_pool": len(self.continual_loop._pool),
        }