# DINO — Minimal Implementation for Study

A from-scratch implementation of DINO (DETR with Improved deNoising anchOr boxes) in PyTorch, designed for learning and understanding.

This is **not** a production implementation. It is simplified and heavily commented to help you study how DINO-style detectors work.

## Files

| File | Purpose |
|---|---|
| `model.py` | Complete DINO architecture: backbone, encoder, decoder, detection heads, denoising generator, iterative refinement |
| `loss.py` | Hungarian matching, focal loss, L1 + GIoU box losses, denoising loss, auxiliary losses |
| `box_ops.py` | Box format conversion (cxcywh ↔ xyxy), IoU, Generalized IoU |
| `train_demo.py` | Synthetic data generation and full training loop |
| `MATH_AND_ALGO_README.md` | Detailed math derivations and algorithm explanations |

## Install

```bash
pip install torch scipy
```

`scipy` is needed for `linear_sum_assignment` (Hungarian algorithm).

## Run Demo

```bash
python -m DINO.train_demo
```

Expected output: training loss decreasing over epochs, then inference results showing detected objects (class, score, box).

## Key Concepts Demonstrated

1. **Hungarian Matching** — Bipartite assignment between predictions and ground truth (no NMS)
2. **Multi-Scale Features** — Backbone features from 2 resolution levels
3. **Anchor-Based Queries** — Each object query has a learned spatial reference point
4. **Iterative Box Refinement** — Each decoder layer refines boxes from the previous layer
5. **Denoising Training** — Noisy GT queries accelerate convergence
6. **Auxiliary Losses** — Detection loss applied at every decoder layer
7. **Focal Loss** — Down-weights easy negatives to focus on hard examples

## Expected Inputs and Outputs

### Training Mode

**Input:**
- `images`: `[B, 3, H, W]` float tensor (normalized)
- `targets`: list of B dicts, each with:
  - `"labels"`: `[N_gt]` LongTensor (class indices)
  - `"boxes"`: `[N_gt, 4]` FloatTensor (cx, cy, w, h normalized to [0,1])

**Output:** dict with:
- `"pred_logits"`: `[B, Q, C]` class logits (final decoder layer)
- `"pred_boxes"`: `[B, Q, 4]` predicted boxes (final decoder layer)
- `"aux_outputs"`: list of `{"pred_logits", "pred_boxes"}` from intermediate layers
- `"dn_pred_logits"`: `[B, Q_dn, C]` denoising logits
- `"dn_pred_boxes"`: `[B, Q_dn, 4]` denoising boxes
- `"dn_targets"`: denoising ground truth

### Inference Mode

**Input:**
- `images`: `[B, 3, H, W]` float tensor

**Output (from `model.predict()`):** list of B dicts, each with:
- `"scores"`: `[K]` confidence scores
- `"labels"`: `[K]` predicted class indices
- `"boxes"`: `[K, 4]` predicted boxes (cx, cy, w, h normalized)

## Important Notes

- This uses **standard multi-head attention** instead of deformable attention for simplicity. The full DINO requires custom CUDA kernels for deformable attention.
- The backbone is a tiny CNN, not ResNet/Swin. Swap in a real backbone for real datasets.
- Training on synthetic data shows the pipeline works but won't produce competitive detection results.
- For study purposes, read `model.py` top-to-bottom, following the forward pass from backbone through decoder to detection heads.
- Then read `loss.py` to understand how matching and losses work.
- See `MATH_AND_ALGO_README.md` for the full mathematical background.

## Comparison with Other Implementations in This Repo

| | Faster R-CNN | YOLO | DINO |
|---|---|---|---|
| Assignment | IoU-based anchor matching | Grid cell overlap | Hungarian matching |
| Proposals | RPN → ~300 regions | Grid cells | Object queries |
| NMS | Required | Required | Not needed |
| Architecture | CNN + RPN + ROI Head | CNN → dense grid | CNN + Transformer |
| Box prediction | Delta from anchor | Absolute in cell | Iterative refinement |
| Training signal | Anchor-GT IoU | Cell-GT overlap | Bipartite matching |
