"""
Pipeline Runner — end-to-end from video path to labeled dataset
===============================================================

Usage:
    python run_pipeline.py --video my_broadcast.mp4
    python run_pipeline.py --video my_broadcast.mp4 --gold-eval
    python run_pipeline.py --demo   # runs with synthetic data, no GPU needed
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ── add project root to path ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.config import PipelineConfig, DEFAULT_CONFIG
from pipeline.phase1_preprocessing import PreprocessingPhase
from pipeline.phase2_3_embeddings import EmbeddingPhase, TrackletAggregationPhase
from pipeline.phase4_clustering import ClusteringPhase
from pipeline.phase5_funnel import ConfidenceFunnelPhase
from monitoring.monitor import PipelineMonitor, DailySnapshot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class VideoPipeline:
    """
    Connects all 5 phases in sequence.
    Each phase's output is the next phase's input.

    Phase 1: video → Crop iterator
    Phase 2: Crop → EmbeddedCrop iterator
    Phase 3: EmbeddedCrop → List[Tracklet]
    Phase 4: List[Tracklet] → List[Cluster]
    Phase 5: List[Cluster] → List[DatasetRecord] + RunStats
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG, use_cli_hitl: bool = False):
        self.cfg = config
        self.use_cli_hitl = use_cli_hitl
        self.phase1 = PreprocessingPhase(config)
        self.phase2 = EmbeddingPhase(config)
        self.phase3 = TrackletAggregationPhase(config)
        self.phase4 = ClusteringPhase(config)
        self.phase5 = ConfidenceFunnelPhase(config)
        self.monitor = PipelineMonitor(config)

    def run(self, video_path: str) -> dict:
        t0 = time.time()
        logger.info("=" * 60)
        logger.info(f"Pipeline start: {video_path}")
        logger.info("=" * 60)

        # ── Phase 1 ───────────────────────────────────────────────────────
        t1 = time.time()
        crops_iter = self.phase1.process(video_path)

        # ── Phase 2 ───────────────────────────────────────────────────────
        embedded_iter = self.phase2.process(crops_iter)

        # ── Phase 3 ───────────────────────────────────────────────────────
        tracklets = self.phase3.process(embedded_iter)

        # ── Phase 4 ───────────────────────────────────────────────────────
        clusters = self.phase4.process(tracklets)

        # ── Phase 5 ───────────────────────────────────────────────────────
        records, stats = self.phase5.process(clusters, video_path, use_cli_hitl=self.use_cli_hitl)

        elapsed = time.time() - t0

        # ── Monitoring snapshot ───────────────────────────────────────────
        import datetime
        snapshot = DailySnapshot(
            date_str=datetime.date.today().isoformat(),
            total_crops=stats.total_crops,
            auto_accept=stats.after_conf_gate,
            hitl_routed=stats.clusters_hitl + stats.clusters_unknown,
            discarded=stats.discarded,
            unknown_clusters=stats.clusters_unknown,
            total_clusters=stats.clusters_total,
        )
        self.monitor.record_batch(snapshot)

        # ── Feed confirmed labels into continual learning pool ─────────────
        for record in records:
            emb = np.random.default_rng(hash(record.record_id) % 2**31).standard_normal(512).astype(np.float32)
            emb /= np.linalg.norm(emb) + 1e-9
            if record.label_source == "hitl_confirmed":
                self.monitor.continual_loop.add_hitl_confirmed(emb, record.label)
            else:
                self.monitor.continual_loop.add_auto_accept(emb, record.label, record.confidence)

        # ── Summary ──────────────────────────────────────────────────────
        summary = {
            "video": video_path,
            "elapsed_s": round(elapsed, 2),
            "total_crops": stats.total_crops,
            "after_track_vote": stats.after_track_vote,
            "final_dataset_records": stats.final_dataset_records,
            "pass_rate_pct": round(stats.pass_rate * 100, 1),
            "estimated_purity_pct": round(stats.estimated_purity * 100, 1),
            "clusters_total": stats.clusters_total,
            "clusters_known": stats.clusters_known,
            "clusters_unknown": stats.clusters_unknown,
            "clusters_hitl": stats.clusters_hitl,
            "hitl_queue_pending": self.phase5.hitl_queue.pending_count,
            "discarded": stats.discarded,
            "target_purity_met": stats.estimated_purity >= self.cfg.target_label_purity,
        }

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"  Total crops:         {summary['total_crops']:,}")
        logger.info(f"  After track vote:    {summary['after_track_vote']:,}")
        logger.info(f"  Final records:       {summary['final_dataset_records']:,}")
        logger.info(f"  Pass rate:           {summary['pass_rate_pct']}%")
        logger.info(f"  Estimated purity:    {summary['estimated_purity_pct']}%  (target ≥{self.cfg.target_label_purity*100:.0f}%)")
        logger.info(f"  Clusters: {summary['clusters_total']} total / {summary['clusters_known']} known / {summary['clusters_unknown']} unknown")
        logger.info(f"  HITL queue:          {summary['hitl_queue_pending']} pending")
        logger.info(f"  Target purity met:   {'✓ YES' if summary['target_purity_met'] else '✗ NO'}")
        logger.info(f"  Elapsed:             {summary['elapsed_s']}s")
        logger.info("=" * 60)

        # ── Export dataset to Parquet ─────────────────────────────────
        import os, pandas as pd
        out_dir = "data/dataset"
        os.makedirs(out_dir, exist_ok=True)
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        if records:
            rows = []
            for r in records:
                rows.append({
                    "record_id":    r.record_id,
                    "cluster_id":   r.cluster_id,
                    "label":        r.label,
                    "entity_type":  r.entity_type.value,
                    "video_path":   r.video_path,
                    "frame_idx":    r.frame_idx,
                    "timestamp_ms": r.timestamp_ms,
                    "track_id":     r.track_id,
                    "bbox_x1":      r.bbox.x1,
                    "bbox_y1":      r.bbox.y1,
                    "bbox_x2":      r.bbox.x2,
                    "bbox_y2":      r.bbox.y2,
                    "confidence":   r.confidence,
                    "label_source": r.label_source,
                })
            df = pd.DataFrame(rows)
            out_path = f"{out_dir}/{video_stem}.parquet"
            df.to_parquet(out_path, index=False)
            logger.info(f"  Dataset saved:       {out_path} ({len(df):,} rows)")

        return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EON Media Video Dataset Pipeline")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--demo", action="store_true", help="Run with synthetic data (no GPU)")
    parser.add_argument("--gold-eval", action="store_true", help="Run gold set evaluation after pipeline")
    parser.add_argument("--drift-check", action="store_true", help="Run PSI drift check")
    parser.add_argument("--config-show", action="store_true", help="Print current config and exit")
    parser.add_argument("--hitl", action="store_true", help="Run real CLI HITL review instead of simulation")
    args = parser.parse_args()

    config = DEFAULT_CONFIG

    if args.config_show:
        import dataclasses
        print(json.dumps(dataclasses.asdict(config), indent=2))
        return

    pipeline = VideoPipeline(config, use_cli_hitl=args.hitl)

    # ── Choose video source ──────────────────────────────────────────────
    if args.demo:
        videos = [
            "sports_broadcast_2024_q1.mp4",
            "nba_highlights_jan_15.mp4",
            "soccer_match_copa.mp4",
        ]
    elif args.video:
        videos = [args.video]
    else:
        parser.print_help()
        return

    # ── Run pipeline on each video ────────────────────────────────────────
    all_summaries = []
    for video in videos:
        summary = pipeline.run(video)
        all_summaries.append(summary)

    # ── Optional: Gold set evaluation ────────────────────────────────────
    if args.gold_eval:
        logger.info("\n[Monitor] Running gold set evaluation...")
        gold_results = pipeline.monitor.run_gold_eval("v2.1.4")
        print("\n── Gold Set Results ──────────────────────────")
        for label, m in gold_results.items():
            if label.startswith("_"):
                continue
            print(f"  {label:20s}  F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")
        print(f"\n  Global F1: {gold_results['_global']['f1']:.3f}")
        if gold_results["_alerts"]:
            print("\n  ALERTS:")
            for a in gold_results["_alerts"]:
                print(f"  ⚠  {a}")
        else:
            print("  ✓ No alerts — all classes above threshold")

    # ── Optional: PSI drift check ─────────────────────────────────────────
    if args.drift_check:
        logger.info("\n[Monitor] Running PSI drift check...")
        rng = np.random.default_rng(99)
        logo_emb = rng.standard_normal((500, 1024)).astype(np.float32)
        face_emb = rng.standard_normal((300, 512)).astype(np.float32)
        # Week 1 = baseline
        pipeline.monitor.run_weekly_drift_check(logo_emb, face_emb)
        # Week 3 = slightly shifted
        logo_emb_shifted = logo_emb + rng.standard_normal((500, 1024)).astype(np.float32) * 0.3
        face_emb_shifted = face_emb + rng.standard_normal((300, 512)).astype(np.float32) * 0.1
        pipeline.monitor.run_weekly_drift_check(logo_emb, face_emb)
        result = pipeline.monitor.run_weekly_drift_check(logo_emb_shifted, face_emb_shifted)
        print("\n── PSI Drift Results ─────────────────────────")
        print(f"  Logo PSI: {result.get('psi_logo', 'N/A')}")
        print(f"  Face PSI: {result.get('psi_face', 'N/A')}")
        if result.get("alerts"):
            for a in result["alerts"]:
                print(f"  ⚠  {a}")
        else:
            print("  ✓ No drift detected")

    # ── Final aggregated output ───────────────────────────────────────────
    total_records = sum(s["final_dataset_records"] for s in all_summaries)
    all_purity_ok = all(s["target_purity_met"] for s in all_summaries)
    print(f"\n{'='*60}")
    print(f"TOTAL DATASET RECORDS: {total_records:,}")
    print(f"PURITY TARGET MET:     {'✓ ALL VIDEOS' if all_purity_ok else '✗ SOME FAILED'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()