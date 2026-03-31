# YOLO (You Only Look Once) — Minimal Implementation for Study

## Fast R-CNN vs YOLO: Key Differences

| Aspect | Fast R-CNN (Two-Stage) | YOLO (One-Stage) |
|--------|----------------------|-------------------|
| **Architecture** | Two stages: (1) Region Proposal Network generates candidate boxes, (2) Classifier refines them | Single stage: one unified network predicts boxes + classes in one pass |
| **Speed** | Slower — must run RPN then per-region classification | Much faster — single forward pass, real-time capable |
| **How it works** | "Where might objects be?" → "What are they?" | Divides image into grid; each cell directly predicts bounding boxes and class probabilities |
| **Accuracy** | Generally higher, especially for small objects | Lower on small/dense objects (improved in later versions) |
| **Loss** | Separate losses for RPN and ROI heads | Single combined loss (localization + confidence + classification) |
| **Region proposals** | Yes — anchors + RPN generate ~300 proposals | No region proposals — directly regresses boxes from grid cells |
| **Feature reuse** | Features extracted per-proposal via ROI pooling/align | Features shared globally; each grid cell uses the full feature map |
| **Post-processing** | NMS on per-class detections | NMS on per-class detections (same) |

### The Core Insight

**Fast R-CNN** asks: *"Let me first find interesting regions, then classify each one carefully."*

**YOLO** asks: *"Let me look at the whole image once and predict everything simultaneously."*

Fast R-CNN's two-stage approach gives it more chances to refine detections, leading to higher
accuracy. YOLO's single-stage approach is dramatically faster because it avoids the expensive
per-region feature extraction step.

---

## Evolution of YOLO Versions

### YOLOv1 (2016) — "You Only Look Once"
- **Paper**: Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection"
- **Key idea**: Divide image into S×S grid. Each cell predicts B bounding boxes + C class probabilities.
- **Output tensor**: S × S × (B × 5 + C), where 5 = (x, y, w, h, confidence)
- **Loss**: Sum-squared error over coordinates, confidence, and class probabilities.
- **Speed**: 45 FPS on Titan X (real-time!).
- **Weakness**: Struggles with small objects; at most one class per grid cell; coarse spatial predictions.

### YOLOv2 / YOLO9000 (2017) — "Better, Faster, Stronger"
- **Paper**: Redmon & Farhadi
- **Key improvements**:
  - **Batch normalization** on all conv layers (removed dropout).
  - **Anchor boxes** (borrowed from Faster R-CNN) — predict offsets relative to anchors instead of raw coordinates.
  - **Dimension clusters** — use k-means on training boxes to pick better anchor priors.
  - **Multi-scale training** — randomly resize input during training for robustness.
  - **Passthrough layer** — concatenate fine-grained features from earlier layers.
  - **Darknet-19** backbone (19 conv layers).
- **Result**: Higher accuracy, still real-time.

### YOLOv3 (2018) — "An Incremental Improvement"
- **Paper**: Redmon & Farhadi
- **Key improvements**:
  - **Multi-scale detection** — predictions at 3 different scales (like FPN), helping small objects.
  - **Darknet-53** backbone — 53 conv layers with residual connections.
  - **Independent logistic classifiers** per class (multi-label, not softmax) — objects can belong to overlapping classes.
  - **3 anchor boxes per scale** (9 total anchors).
- **Result**: Competitive with SSD and RetinaNet; still very fast.

### YOLOv4 (2020) — "Optimal Speed and Accuracy"
- **Authors**: Bochkovskiy, Wang, Liao (Redmon had left the field)
- **Key improvements**:
  - **Bag of Freebies**: CutMix, Mosaic augmentation, DropBlock, label smoothing.
  - **Bag of Specials**: Mish activation, CSPNet backbone, SPP, PAN neck.
  - **CSPDarknet53** backbone.
  - Extensive empirical study of what training tricks actually help.
- **Result**: State-of-the-art speed-accuracy tradeoff.

### YOLOv5 (2020) — Ultralytics (no paper, code-only release)
- **Key changes**:
  - PyTorch-native implementation (previous versions used Darknet/C).
  - Auto-anchor learning, mosaic augmentation, focus layer.
  - Multiple model sizes: YOLOv5n/s/m/l/x.
  - Easy-to-use training pipeline with YAML configs.
- **Controversial**: No academic paper; some debated whether it was a true "v5".

### YOLOv6 (2022) — Meituan
- Efficient decoupled head, anchor-free approach, RepVGG-style backbone.
- Optimized for industrial deployment.

### YOLOv7 (2022) — Wang, Bochkovskiy, Liao
- **E-ELAN** (Extended Efficient Layer Aggregation Network).
- Model re-parameterization & planned re-parameterized convolution.
- Auxiliary head for training (coarse-to-fine lead head).

### YOLOv8 (2023) — Ultralytics
- **Anchor-free** detection (no predefined anchor boxes).
- Decoupled head (separate branches for classification and regression).
- C2f module (improved CSP bottleneck).
- Unified framework for detection, segmentation, classification, and pose.

### YOLOv9 (2024) — Wang et al.
- **Programmable Gradient Information (PGI)** — addresses information bottleneck.
- **GELAN** (Generalized Efficient Layer Aggregation Network).
- Focuses on preserving information through deep networks.

### YOLOv10 (2024) — Tsinghua University
- **NMS-free** training via consistent dual assignments.
- Efficiency-accuracy driven design with spatial-channel decoupled downsampling.
- Large-kernel convolutions and partial self-attention.

### YOLOv11 / YOLO-World / YOLO-NAS (2024-2025)
- Open-vocabulary detection (YOLO-World).
- Neural Architecture Search (YOLO-NAS).
- Continued trend toward anchor-free, transformer-augmented, multi-task architectures.

### Evolution Summary
```
YOLOv1 (grid-based, raw regression)
  ↓ + anchor boxes, batch norm, multi-scale training
YOLOv2 (anchor-based, Darknet-19)
  ↓ + multi-scale detection, residual backbone, multi-label
YOLOv3 (FPN-like, Darknet-53)
  ↓ + advanced augmentation, CSP backbone, SPP, PAN
YOLOv4 (bag of freebies/specials, CSPDarknet53)
  ↓ + PyTorch-native, auto-anchor, easy pipeline
YOLOv5 (practical engineering, multiple sizes)
  ↓ + anchor-free experiments, rep-conv
YOLOv6/v7 (efficiency + re-parameterization)
  ↓ + fully anchor-free, decoupled head, unified tasks
YOLOv8 (anchor-free, multi-task)
  ↓ + information preservation, NMS-free
YOLOv9/v10 (PGI, GELAN, dual assignments)
```

---

## This Implementation

This folder contains a **minimal YOLOv1-style** detector written from scratch in PyTorch.
It follows the original YOLOv1 design for clarity:

- **S×S grid** divides the image
- Each cell predicts **B bounding boxes** (each with x, y, w, h, confidence) + **C class probabilities**
- **Single-pass** forward through a CNN backbone → reshape → detection output
- **Combined loss** = λ_coord × localization_loss + confidence_loss + λ_noobj × no-object_loss + classification_loss

Files:
- `model.py` — YOLOv1 network architecture with detailed comments
- `loss.py` — YOLOv1 loss function with all components explained
- `box_ops.py` — IoU and box conversion utilities
- `train_demo.py` — Minimal training script with synthetic data
- `MATH_AND_ALGO_README.md` — Detailed mathematical derivations
