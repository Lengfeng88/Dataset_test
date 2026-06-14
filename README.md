# Eon Media — Video Dataset Pipeline

Automated pipeline for extracting labeled brand logo and face datasets from sports broadcast footage. Processes raw video into a high-purity annotated dataset using a 5-phase detection, embedding, clustering, and confidence-filtering architecture.

**Tested result:** 5-minute NBA 1080p broadcast → 2,545 labeled records at **99.0% estimated label purity**.

---

## Pipeline Overview

```
Video
  │
  ▼ Phase 1 — Preprocessing & Detection
  │   TransNetV2 scene splitting → optical flow keyframe extraction
  │   Grounding DINO (open-vocab logo detection) + RetinaFace (face detection)
  │   ByteTrack multi-object tracking → stable track_ids
  │
  ▼ Phase 2+3 — Embeddings & Tracklet Aggregation
  │   DINOv2 ViT-L/14 → 1024-dim logo embeddings
  │   AdaFace IR-100   → 512-dim face embeddings
  │   Mean embedding per track_id → tracklet vectors
  │
  ▼ Phase 4 — Clustering & Labeling
  │   Leiden community detection (γ=0.72, cosine > 0.75)
  │   CLIP ViT-L/14 zero-shot labeling → known brand/person or brand_unknown_NNN
  │   Temporal Stability Score (TSS) per cluster
  │
  ▼ Phase 5 — Confidence Funnel
  │   Track vote ≥ 70%  → assign label
  │   Score ≥ 0.85      → auto-accept → dataset
  │   Score 0.60–0.85   → HITL queue → human review → dataset
  │   Score < 0.60      → discard
  │
  ▼ Output
      Parquet dataset — label, entity_type, bbox, confidence, label_source
```

Human cost is **O(clusters)** not O(samples): one HITL review labels all crops in a cluster simultaneously.

---

## Results

| Metric | Value |
|--------|-------|
| Input | `nba_1080p.mp4` — 5 min, 1080p, 25fps |
| Total crops detected | 3,109 |
| Final dataset records | 2,545 |
| Pass rate | 81.9% |
| Estimated label purity | **99.0%** (target ≥ 95%) |
| Clusters found | 911 (896 known, 15 unknown) |
| HITL queue | 0 pending |
| Processing time | 30 min (RTX 4080 12GB) |

---

## Project Structure

```
Eon_dataset_pipeline/
├── run_pipeline.py              # Entry point
├── requirement.txt
├── .gitignore
│
├── pipeline/
│   ├── config.py                # All thresholds in one place
│   ├── models.py                # Crop → EmbeddedCrop → Tracklet → Cluster → DatasetRecord
│   ├── phase1_preprocessing.py  # TransNetV2 + Grounding DINO + RetinaFace + ByteTrack
│   ├── phase2_3_embeddings.py   # DINOv2 + AdaFace + tracklet aggregation
│   ├── phase4_clustering.py     # Leiden + CLIP labeling + TSS
│   ├── phase5_funnel.py         # Confidence funnel + HITL queue
│   ├── cache.py                 # Phase 1+2 result caching (skip re-runs)
│   └── hitl_cli.py              # CLI for human-in-the-loop review
│
├── monitoring/
│   └── monitor.py               # 3-layer monitoring stack
│
├── tests/
│   └── test_pipeline.py         # 23 tests including purity guarantee tests
│
└── data/
    ├── videos/                  # Input videos (gitignored)
    ├── dataset/                 # Output parquet files (gitignored)
    └── cache/                   # Phase cache (gitignored)
```

---

## Setup

**Requirements:** Python 3.11, CUDA 12.1, cuDNN 8.x

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/Eon_dataset_pipeline.git
cd Eon_dataset_pipeline

# Virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirement.txt

# cuDNN 8 (required for onnxruntime-gpu — system cuDNN 9 is not compatible)
pip install nvidia-cudnn-cu12==8.9.7.29
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# Download Grounding DINO weights
mkdir -p weights
wget -O weights/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# InsightFace buffalo_l model downloads automatically on first run
```

**Known dependency conflict:** `opencv-python>=4.10` requires `numpy>=2`, but `onnxruntime-gpu==1.18.0` requires `numpy<2`. Fix:

```bash
pip install "numpy==1.26.4" "opencv-python==4.9.0.80" "onnxruntime-gpu==1.18.0" --no-deps \
  --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

---

## Usage

```bash
# Run on a video
python -W ignore run_pipeline.py --video data/videos/your_video.mp4

# Run with gold set evaluation
python -W ignore run_pipeline.py --video data/videos/your_video.mp4 --gold-eval

# Run with PSI drift check
python -W ignore run_pipeline.py --video data/videos/your_video.mp4 --drift-check

# Print current config
python run_pipeline.py --config-show
```

Output is written to `data/dataset/<video_name>.parquet` with columns:

```
record_id, cluster_id, label, entity_type, video_path,
frame_idx, timestamp_ms, track_id,
bbox_x1, bbox_y1, bbox_x2, bbox_y2,
confidence, label_source
```

---

## How 95%+ Purity Is Maintained

Purity is a **design constraint**, not a model accuracy target. Three mechanisms enforce it:

**1. Track-level voting** — A label is only assigned to a tracklet if ≥70% of its frames agree. Single-frame noise is rejected by majority vote.

**2. Confidence gating** — Only samples with score ≥ 0.85 enter the dataset automatically. The 0.60–0.85 band goes to human review. Below 0.60 is discarded. Uncertain samples never enter the dataset.

**3. Cluster-level HITL** — Humans review representative frames once per cluster. Confirming a cluster label propagates to all samples in it simultaneously. This keeps human cost at O(clusters) regardless of video length.

---

## Monitoring

Three layers of ongoing quality assurance (see `monitoring/monitor.py`):

| Layer | Frequency | What it catches |
|-------|-----------|-----------------|
| Gold set F1 evaluation | Per model update | Regression in per-class accuracy |
| Online proxy metrics | Daily | HITL queue growth, auto-accept rate drop |
| PSI embedding drift | Weekly | Distribution shift in DINOv2/AdaFace embeddings |

Alert thresholds in `pipeline/config.py`:

```python
psi_warn_threshold     = 0.25   # PSI > 0.25 → schedule fine-tune
gold_f1_min            = 0.85   # per-class F1 floor
gold_f1_drop_alert     = 0.02   # 2pp drop → alert
hitl_growth_alert      = 0.20   # 20% WoW HITL growth → alert
```

---

## Configuration

All thresholds are in `pipeline/config.py`. Key parameters:

```python
track_vote_threshold  = 0.70   # min frame agreement to assign label
auto_accept_score     = 0.85   # above → write to dataset
hitl_low_score        = 0.60   # below → discard
leiden_resolution     = 0.72   # cluster granularity (higher = finer)
similarity_threshold  = 0.75   # cosine similarity edge threshold
clip_known_threshold  = 0.60   # below → brand_unknown_NNN
target_label_purity   = 0.95   # pipeline asserts this is met
```

---

## Tests

```bash
python -m pytest tests/ -v
```

23 tests covering:
- Track vote logic
- Confidence gate boundary values
- HITL queue operations
- TSS computation
- PSI drift detection
- **End-to-end purity guarantee** — asserts estimated purity ≥ 95% on synthetic clusters

---

## Environment Notes

Tested configuration:
- Ubuntu 24.04
- Python 3.11
- PyTorch 2.5.1+cu121
- CUDA 12.1 / Driver 595.71.05
- onnxruntime-gpu 1.18.0 + nvidia-cudnn-cu12 8.9.7.29
- RTX 4080 12GB (Laptop) — 30 min per 5-min 1080p video

Phase 1 (Grounding DINO detection) accounts for ~87% of total runtime. Adding the phase cache (`pipeline/cache.py`) eliminates re-processing on repeated runs of the same video.