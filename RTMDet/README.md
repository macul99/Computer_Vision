# RTMDet — Minimal Implementation for Study

## YOLOv1 vs RTMDet: Key Differences

| Aspect | YOLOv1 (see ../yolo/) | RTMDet |
|--------|----------------------|--------|
| **Year** | 2016 | 2022 |
| **Feature scales** | Single scale (7×7) | Multi-scale (3 levels via FPN) |
| **Box format** | (x, y, w, h) per grid cell | (l, t, r, b) distances from point |
| **Anchors** | B fixed boxes per cell | Anchor-free, point-based |
| **Head** | Shared FC layers (global) | Decoupled conv heads (cls + reg) |
| **Backbone** | VGG-style plain CNN | CSPNeXt with depthwise separable convs |
| **Neck** | None | PAFPN (bidirectional feature fusion) |
| **Label assignment** | Static (grid cell ownership) | Dynamic (SimOTA) |
| **Classification loss** | MSE or CrossEntropy | Quality Focal Loss |
| **Box regression loss** | MSE on coordinates | GIoU loss |

### The Core Insight

**YOLOv1** asks: *"Divide the image into a grid and let each cell predict objects."*

**RTMDet** asks: *"Put a prediction point at every feature-map location across multiple scales, dynamically assign targets during training, and let the model learn quality-aware predictions."*

RTMDet improves on YOLOv1 in almost every dimension: multi-scale features catch small objects, dynamic assignment adapts to the model's learning, and quality-aware losses produce better-calibrated confidence scores.

---

## Architecture Overview

```
Input [B, 3, 320, 320]
      │
      ▼
┌─────────────────────┐
│  CSPNeXt Backbone    │
│  Stem: stride 4      │
│  Stage1: stride 8    │── P3 [128, 40, 40]  (small objects)
│  Stage2: stride 16   │── P4 [256, 20, 20]  (medium objects)
│  Stage3: stride 32   │── P5 [512, 10, 10]  (large objects)
└───────┬──────┬───────┘
        │      │      │
        ▼      ▼      ▼
┌─────────────────────┐
│    CSPAFPN Neck      │
│  Top-down: P5→P4→P3  │  (semantics flow down)
│  Bottom-up: P3→P4→P5 │  (detail flows up)
│  All → 128 channels   │
└───────┬──────┬───────┘
        │      │      │
        ▼      ▼      ▼
┌─────────────────────┐
│  Decoupled Head      │  (shared weights across levels)
│  cls_branch → [C]    │  Classification scores
│  reg_branch → [4]    │  Box distances (l,t,r,b)
└─────────────────────┘

Total: 1600 + 400 + 100 = 2100 predictions per image
```

---

## Key Components

### 1. CSPNeXt Backbone
Cross Stage Partial network with large-kernel depthwise separable convolutions. Splits features into two paths (processed + shortcut), concatenates, and mixes. Uses 5×5 depthwise convs for larger receptive field without heavy compute.

### 2. CSPAFPN (Path Aggregation FPN)
Bidirectional feature fusion:
- **Top-down**: High-level semantics from P5 flow down to P3
- **Bottom-up**: Fine-grained spatial detail from P3 flows back up to P5

After fusion, all levels have the same channel count and rich multi-scale information.

### 3. Decoupled Head
Separate branches for classification and regression. The same head weights are shared across all feature levels. Uses SiLU activation and proper initialization (bias trick for focal loss).

### 4. SimOTA Assignment
Dynamic label assignment that considers the model's current predictions. Assigns more positives to well-predicted objects, creating a positive feedback loop during training.

### 5. Quality Focal Loss
Soft-target focal loss where classification targets are IoU values (not binary 0/1). Unifies classification confidence with localization quality.

### 6. GIoU Loss
Generalized IoU for box regression. Provides gradients even when boxes don't overlap, unlike plain IoU.

---

## Files

| File | Contents |
|------|----------|
| `model.py` | CSPNeXtBackbone, CSPAFPN, RTMDetHead, RTMDet |
| `loss.py` | QualityFocalLoss, GIoULoss, SimOTAAssigner, RTMDetLoss |
| `box_ops.py` | Box format conversions, IoU, GIoU, distance↔bbox |
| `train_demo.py` | Training demo with synthetic data |
| `MATH_AND_ALGO_README.md` | Detailed mathematical foundations |

---

## Running the Demo

```bash
cd /path/to/Computer_Vision
python -m RTMDet.train_demo
```

---

## What to Study in the Code

1. **`model.py` — CSPNeXtBlock**: How CSP splits and merges feature paths. Compare with plain residual blocks.

2. **`model.py` — CSPAFPN**: How top-down and bottom-up paths work. Trace the data flow through `forward()`.

3. **`model.py` — RTMDetHead.get_points()`**: How anchor-free points are generated. Why `+ 0.5` offset?

4. **`loss.py` — SimOTAAssigner`**: The dynamic assignment algorithm. This is the most complex and important part. Trace through the 5 steps.

5. **`loss.py` — QualityFocalLoss`**: Compare with standard BCE. Notice the soft targets and focusing factor.

6. **`box_ops.py` — distance2bbox/bbox2distance`**: The ltrb format is central to anchor-free detection. Draw it on paper.

7. **`box_ops.py` — giou`**: Compare with plain IoU. Why does GIoU give gradients when boxes don't overlap?

---

## Evolution Context

RTMDet represents the convergence of many ideas:

```
YOLO (2016) → grid-based detection, single stage
    ↓
YOLOv2 (2017) → anchor boxes, multi-scale
    ↓
FCOS (2019) → anchor-free, point-based, ltrb distances
    ↓
YOLOX (2021) → decoupled head, dynamic assignment (OTA)
    ↓
RTMDet (2022) → CSPNeXt backbone, SimOTA, quality focal loss
```

RTMDet combines the best engineering ideas from YOLO (efficient backbone), FCOS (anchor-free), YOLOX (dynamic assignment), and focal loss research into one clean, fast detector.
